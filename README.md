# CaMonitor â Edge AI Nursery Safety Guard

Real-time child safety monitoring using pose estimation on a Raspberry Pi 4.
No cloud, no subscription, no data leaving the device.

Built as a personal project and engineering portfolio piece â deployed to monitor
a 20-month-old toddler at home.

---

## What It Does

CaMonitor runs a full computer vision pipeline on a Raspberry Pi 4 to detect
potentially dangerous situations involving a young child in real time.

**Alert types:**

| Alert | Category | Severity | Trigger |
|---|---|---|---|
| `ZONE_ENTRY` | Spatial | MEDIUM/HIGH | Child enters a mapped danger zone (bed edge, furniture, kitchen) |
| `INVERSION` | Posture | HIGH | Head drops below foot level (tumbling, hanging) |
| `CLIMBING` | Posture | HIGH | Wrists and knees simultaneously above hip level |
| `AIRBORNE` | Posture | MEDIUM | Both ankles above floor baseline |
| `RAPID_DESCENT` | Motion | HIGH | Hip Y-coordinate drops > 4% of frame height per frame |

On detection, the system sends an email notification with alert type, severity,
timestamp, and an attached JPEG frame showing the skeleton overlay.

**Zone severity levels:**
- `MEDIUM` â furniture zones (bed edge, sofa, chairs)
- `HIGH` â kitchen zone (fridge, water tap/basin, and oven used as spatial anchors for zone boundary definition)

Kitchen zone is defined by the convex hull of three anchor objects detected by
YOLOv8: fridge, water tap/basin, and oven. Any keypoint inside this hull triggers
a HIGH-severity alert.

---

## System Output â Real Frames

All frames below are actual system output captured during live testing.

### CLEAR â Adult Detected, Alerts Suppressed

![CLEAR adult squat](docs/clear_adult_squat.png)

*FPS:6.6 Frame:1359 â Status: CLEAR. Adult in squat/crouch position near desk.
Green border = no active alerts. Orange skeleton overlay with green keypoints tracked correctly.
Adult filter correctly identifies adult body (bbox area > 0.02) and suppresses all alerts
despite low crouching posture â demonstrating orientation-invariant adult classification.*

---

### ZONE_ENTRY â Toddler Enters Bed Zone

![ZONE_ENTRY alert](docs/alert_zone_entry.jpg)

*FPS:4.4 Frame:2183 â [MEDIUM] ZONE_ENTRY bed. Toddler standing and playing near
the bed edge zone. Blue zone boxes for bed (top-left) and chair (bottom) visible.
Full toddler skeleton tracked at 0.55 confidence threshold. Adult legs partially
visible on right edge of frame â adult filter correctly suppresses adult body,
alerting only on the toddler.*

---

### CLIMBING â Toddler Reaches Above Hip Level

![CLIMBING alert](docs/alert_climbing.jpg)

*FPS:5.2 Frame:2297 â [HIGH] CLIMBING (wrists+knees above hip_y=0.33). Red border
= HIGH severity alert. Toddler reaching upward with arm fully extended above head
level while seated at toy desk. Green keypoints on shoulder, elbow, and wrist chain.
System correctly classifies elevated wrist position as climbing behaviour.*

---

### Room Scan â YOLOv8 Zone Mapping

![Room scan result](docs/room_scan_result.jpg)

*Room scan output from `room_scan.py`. YOLOv8 detected 3 zones: bed (magenta filled
polygon, left), chair Ã2 (cyan bounding boxes, centre). Zone definitions saved to
`zones_config.yaml` and loaded by `alerts.py` at startup. Re-run whenever camera
position changes.*

---

## System Architecture

```
Camera (Logitech StreamCam, 720p/MJPG mode, top-down ~2m height)
    â
    â¼
OpenCV frame capture (MJPG set before resolution â USB 2.0 bandwidth requirement)
    â
    â¼
MediaPipe BlazePose (TFLite XNNPACK delegate, model_complexity=0)
    â
    âââº Adult filter (bbox area > 0.02 â adult â skip alerts)
    â   Bounding box of 11 upper-body keypoints
    â   Orientation-invariant for fixed overhead camera
    â
    âââº Zone alert  (hip/ankle keypoints vs YAML zone boxes)
    â   15-frame persistence filter â requires ~2s sustained detection
    â
    âââº Posture alerts (inversion, climbing, airborne)
    â
    âââº Motion alert  (rapid descent â inter-frame hip delta)
    â
    â¼
CSV logging + JPEG frame save + Gmail email notification
```

---

## Benchmark Results

### Phase 1 â Resolution vs Baseline FPS

| Resolution | Avg FPS | Avg CPU | Avg Temp | Peak Temp |
|---|---|---|---|---|
| 480p | 63.12 | 14.3% | 50.2Â°C | 52.6Â°C |
| 720p | 62.74 | 25.9% | 52.3Â°C | 55.0Â°C |
| 1080p | 37.11 | 30.7% | 54.3Â°C | 56.5Â°C |

**Decision:** 1080p consumed 30% CPU before any AI. 720p held 63 FPS capture
at 26% CPU, leaving full headroom for inference.

### Phase 1b â XNNPACK Delegate Verification

TFLite automatically selects the XNNPACK delegate on Raspberry Pi ARM Cortex-A72/A76,
using NEON SIMD to vectorise neural network multiply-accumulate operations â
4 floats processed per CPU instruction instead of 1.

Confirmed active via startup log:
```
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
```

| Backend | FPS | CPU | Cost | Status |
|---|---|---|---|---|
| Naive CPU (no delegate) | ~2â3 FPS | ~65% | Free | Too slow |
| **XNNPACK / ARM NEON** | **8.63 FPS** | **36%** | **Free, auto-selected** | **â Production** |
| GPU delegate (VideoCore) | Not supported | â | Free | â Not available on Pi |
| Coral Edge TPU (USB) | ~25â30 FPS | ~15% | ~S$80 | Future upgrade |

XNNPACK provides ~3Ã throughput over estimated naive CPU path at zero cost.
Coral Edge TPU evaluated â 8.63 FPS sufficient for monitoring a toddler,
S$80 BOM addition unjustified at current stage.

### Phase 1c â PTQ Accuracy / Latency Sweep

Google ships BlazePose in three quantization operating points (all int8):

| Complexity | Model Size | FPS | Detection Rate | Confidence | Decision |
|---|---|---|---|---|---|
| 0 â lite | 2.7MB | **8.63** | **100%** | 0.997 | â Production |
| 1 â full | 6.2MB | 6.11 | 100% | 0.998 | Reference |
| 2 â heavy | ~26MB | 1.73 | 98% | 0.993 | â Not viable |

Key finding: complexity=0 gives **41% more throughput** than complexity=1 at only
**0.1% confidence drop**. Complexity=2 performed worst overall â inference latency
exceeds frame arrival rate, causing tracking continuity loss (98% detection rate
vs 100% for lighter models). complexity=0 is the unambiguous choice for edge deployment.

---

## Adult / Child Differentiation

### Design Evolution

The system must suppress alerts when a parent is in frame (CHILD_MONITOR mode).
Three approaches were evaluated for a fixed top-down overhead camera â each
failing a distinct edge case:

**Attempt 1 â height_span (nose-to-ankle ratio)**
Works upright but fails when ankles are not visible (sitting, crouching).
Defaults to child classification â triggers false alerts on parent.
Calibrated value: `height_span mean=0.5999, std=0.0429`

**Attempt 2 â head/shoulder width ratio**
Ankles-independent, but fails when parent is side-on to camera â shoulder
width collapses in top-down projection, misclassifying adult as toddler.

**Attempt 3 â Bounding box area (current)**
Orientation-invariant for a fixed overhead camera. An adult body always
occupies more 2D frame area than a toddler regardless of pose or facing direction.
Uses bounding box of 11 upper-body keypoints (nose, shoulders, elbows,
wrists, hips, knees).

### Validated Thresholds

| Subject / Pose | bbox Area | Classification |
|---|---|---|
| Adult standing / moving | 0.087 â 0.189 | â adult=True |
| Adult sitting, side-on | 0.063 â 0.074 | â adult=True |
| Adult crouching low | 0.021 â 0.055 | â adult=True (~90% accuracy) |
| Toddler (20 months) standing | < 0.020 | â child=True (100%) |
| Toddler crawling | < 0.020 | â child=True (100%) |

**Validated threshold:** `bbox_area_threshold: 0.02`
Validated 2026-03-10 with real child subject in frame.

At threshold=0.02, the adult/child gap is well-separated: even a deeply crouching
adult correctly classifies at ~90% accuracy, while toddler detection is 100%.
The 15-frame persistence filter further prevents any misclassified frame from
triggering an alert.

---

## Synthetic Data Strategy

`RAPID_DESCENT` (child falling) is an edge case that cannot be safely collected
in real life. The correct approach is geometric augmentation of real landmark
sequences â scale, flip, and jitter 20 real captures to generate 1000+ synthetic
variants for threshold validation and future classifier training.

```python
def augment_landmarks(landmarks_array, n_augments=50):
    """landmarks_array: shape (n_frames, 33, 3) â x, y, visibility"""
    augmented = []
    for _ in range(n_augments):
        aug = landmarks_array.copy()
        aug[:, :, :2] *= np.random.uniform(0.8, 1.2)     # scale
        if np.random.random() > 0.5:
            aug[:, :, 0] = 1.0 - aug[:, :, 0]            # horizontal flip
        aug[:, :, :2] += np.random.normal(0, 0.01,
                         aug[:, :, :2].shape)              # jitter
        aug[:, :, :2] = np.clip(aug[:, :, :2], 0, 1)
        augmented.append(aug)
    return augmented
```

---

## Project Structure

```
camonitor/
âââ scripts/
â   âââ alerts.py                    # Main monitoring loop
â   âââ benchmark.py                 # Baseline benchmark harness
â   âââ ptq_benchmark.py             # PTQ complexity sweep benchmark
â   âââ room_scan.py                 # YOLOv8 zone detection
â   âââ calibrate_adult.py           # Adult profile calibration
â   âââ email_notifier.py            # Gmail SMTP notification
â   âââ config.yaml                  # Master config
â   âââ adult_profile.yaml           # Saved calibration ratios
â   âââ zones_config.yaml            # Room zone definitions
â   âââ email_config_template.yaml   # Credential template
âââ logs/
â   âââ benchmark_baseline.csv
â   âââ benchmark_inference.csv
â   âââ ptq_benchmark.csv
â   âââ alerts_log.csv
âââ data/
â   âââ room_scan_result.jpg
âââ docs/
â   âââ screenshots/
â       âââ clear_adult_squat.png    # Adult crouching â CLEAR, alerts suppressed
â       âââ alert_zone_entry.jpg     # Toddler ZONE_ENTRY bed alert
â       âââ alert_climbing.jpg       # Toddler CLIMBING HIGH alert
â       âââ room_scan_result.jpg     # YOLOv8 zone mapping output
âââ README.md
```

---

## Setup

Tested on Raspberry Pi 4 (4GB), Raspberry Pi OS 64-bit, Python 3.11/3.12.

```bash
conda create -n camonitor python=3.11
conda activate camonitor
pip install -r requirements.txt
```

### Camera Setup Note

Set MJPG format **before** resolution â required for USB 2.0 bandwidth:

```python
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # FIRST
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
```

If reversed, camera falls back to YUYV â 3Ã USB bandwidth, causing frame drops.

---

## Running

### Step 1 â Room Scan (once per camera position)
```bash
python scripts/room_scan.py
```
YOLOv8 maps furniture into zone boxes. Saves `zones_config.yaml`.
For kitchen zone: ensure fridge, water tap/basin, and oven are visible in frame
during scan â YOLOv8 uses these as spatial anchors for the kitchen exclusion zone.

### Step 2 â Adult Calibration (once)
```bash
python scripts/calibrate_adult.py
```
Stand in frame for ~8 seconds. Saves `adult_profile.yaml`.

### Step 3 â Configure
```yaml
# config.yaml
mode: CHILD_MONITOR        # ADULT_TEST during development
debug: false               # true prints bbox area values for threshold tuning
adult_filter:
  bbox_area_threshold: 0.02  # validated 2026-03-10
```

### Step 4 â Monitor
```bash
# Foreground
python scripts/alerts.py

# Background (24/7)
nohup python scripts/alerts.py >> logs/monitor.log 2>&1 &
```

### Step 5 â PTQ Benchmark (optional)
```bash
python scripts/ptq_benchmark.py
```
Stand in frame during each complexity level run (~5 min total).

---

## Email Notifications

```bash
cp scripts/email_config_template.yaml scripts/email_config.yaml
chmod 600 scripts/email_config.yaml
nano scripts/email_config.yaml        # add Gmail App Password
```

Gmail â Security â 2-Step Verification â App Passwords â generate for CaMonitor.

---

## Configuration Reference

```yaml
mode: CHILD_MONITOR     # ADULT_TEST or CHILD_MONITOR
debug: false            # true = print bbox area values for threshold tuning

adult_filter:
  enabled: true
  bbox_area_threshold: 0.02   # validated â 100% child detection, ~90% adult at crouch

mediapipe:
  min_detection_confidence: 0.55
  min_tracking_confidence: 0.55
  model_complexity: 0         # 8.63 FPS, 100% detection, 0.997 confidence

alerts:
  zone_entry_frames: 15       # ~2 seconds persistence before alert fires
  inversion_buffer: 0.05
  descent_threshold: 0.04

email:
  enabled: true
  cooldown_sec: 300

performance:
  idle_sleep_sec: 0.3
```

---

## Known Limitations

**Adult presence = child safe:** The system suppresses all alerts when an adult
body is detected in frame. This is intentional â an adult present means the child
is supervised. The monitor is designed for unsupervised periods only.

**BlazePose on toddlers:** Model trained predominantly on adults. Lower keypoint
confidence expected for a 20-month toddler's body proportions.
`min_detection_confidence: 0.55` set lower than default 0.60 to compensate.
In practice, toddler skeleton tracked reliably at this threshold as shown in
the ZONE_ENTRY and CLIMBING frames above.

**RAPID_DESCENT camera height dependency:** Requires camera at 1.8â2m height for
sufficient hip Y range per frame at 8 FPS. At monitor-level mounting, hip vertical
range is insufficient to exceed the per-frame delta threshold. Known hardware
deployment constraint â not a logic bug.

---

## Roadmap

- [ ] LivenessChecker â reject static false detections (stuffed animals, sheet creases)
- [ ] RAPID_DESCENT validation at current camera height
- [ ] Kitchen zone implementation â fridge/basin/oven anchor detection in room_scan.py
- [ ] Spatial floor calibration â user-defined floor plane for posture alerts
- [ ] Dynamic thresholding â confidence gate based on frame brightness
- [ ] Threading pipeline â producer/consumer for capture, inference, alert logic
- [ ] PWA dashboard â Flask server + mobile live view
- [ ] Synthetic data augmentation script for RAPID_DESCENT training data

---

## Hardware

- Raspberry Pi 4 (4GB RAM)
- Logitech StreamCam USB webcam (720p/MJPG, top-down ~2m height)
- Camera angled 15â20Â° downward for full room coverage
- Fan case recommended for sustained all-day operation

---

## License

MIT â personal project, provided as-is.
Not a substitute for direct supervision or commercial child safety products.
