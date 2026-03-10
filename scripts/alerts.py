import cv2
import mediapipe as mp
import time
import psutil
import csv
import os
import yaml
import datetime
import numpy as np
from email_notifier import AlertEmailer
import time

os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ.setdefault('DISPLAY', ':0')

THERMAL_PATH = '/sys/class/thermal/thermal_zone0/temp'
YAML_PATH    = os.path.expanduser('~/Documents/camonitor/scripts/zones_config.yaml')
LOG_PATH     = os.path.expanduser('~/Documents/camonitor/logs/alerts_log.csv')
IMG_DIR      = os.path.expanduser('~/Documents/camonitor/data/alerts')

CONFIDENCE_THRESHOLD = 0.60

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
LMKS    = mp_pose.PoseLandmark

os.makedirs(IMG_DIR, exist_ok=True)

ALERT_DIR      = os.path.expanduser('~/Documents/camonitor/data/alerts')
RETENTION_DAYS = 3

def cleanup_old_alerts():
    """Delete alert images older than RETENTION_DAYS."""
    if not os.path.exists(ALERT_DIR):
        return
    cutoff = time.time() - RETENTION_DAYS * 86400
    deleted = 0
    for fname in os.listdir(ALERT_DIR):
        fpath = os.path.join(ALERT_DIR, fname)
        if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
            os.remove(fpath)
            deleted += 1
    if deleted:
        print(f"  [CLEANUP] Deleted {deleted} alert images older than {RETENTION_DAYS} days")


def get_temp():
    return round(float(open(THERMAL_PATH).read()) / 1000, 1)

def load_zones():
    if not os.path.exists(YAML_PATH):
        print("ERROR: zones_config.yaml not found. Run room_scan.py first.")
        return {}
    with open(YAML_PATH, 'r') as f:
        zones = yaml.safe_load(f)
    frame_info = zones.pop('_frame_info', {})
    print(f"Zones loaded: {list(zones.keys())}")
    print(f"Calibrated for: {frame_info.get('width')}x{frame_info.get('height')}")
    return zones

def get_kp(landmarks, name):
    """Return (x_px, y_px, visibility) for a named keypoint."""
    lm = landmarks.landmark[LMKS[name]]
    return lm.x, lm.y, lm.visibility  # x,y normalised 0–1

def kp_confident(vis):
    return vis >= CONFIDENCE_THRESHOLD

def point_in_box(nx, ny, box, frame_w, frame_h):
    """Check if normalised point (nx,ny) is inside pixel box [x1,y1,x2,y2]."""
    px = nx * frame_w
    py = ny * frame_h
    return box[0] <= px <= box[2] and box[1] <= py <= box[3]

# ── Category A: Zone-based ───────────────────────────────────────────────────

def check_zone_alerts(landmarks, zones, frame_w, frame_h):
    alerts = []
    check_kps = ['LEFT_HIP', 'RIGHT_HIP',
                 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX',
                 'LEFT_ANKLE', 'RIGHT_ANKLE']

    for zone_name, zone_data in zones.items():
        box = zone_data['box']
        for kp_name in check_kps:
            x, y, vis = get_kp(landmarks, kp_name)
            if not kp_confident(vis):
                continue
            if point_in_box(x, y, box, frame_w, frame_h):
                alerts.append({
                    'type'    : 'ZONE_ENTRY',
                    'zone'    : zone_name,
                    'keypoint': kp_name,
                    'severity': 'HIGH' if zone_name == 'kitchen' else 'MEDIUM'
                })
                break  # one alert per zone per frame is enough
    return alerts

# ── Category B: Posture-based ────────────────────────────────────────────────

def check_inversion(landmarks):
    """Head lower than feet = inversion / falling."""
    _, head_y, head_v  = get_kp(landmarks, 'NOSE')
    _, lf_y,  lf_v     = get_kp(landmarks, 'LEFT_FOOT_INDEX')
    _, rf_y,  rf_v     = get_kp(landmarks, 'RIGHT_FOOT_INDEX')

    if not (kp_confident(head_v) and
            (kp_confident(lf_v) or kp_confident(rf_v))):
        return None

    foot_ys = [y for y, v in [(lf_y, lf_v), (rf_y, rf_v)] if kp_confident(v)]
    avg_foot_y = sum(foot_ys) / len(foot_ys)

    # In image coords y increases downward
    # head_y > avg_foot_y means head is LOWER than feet in frame
    if head_y > avg_foot_y:
        return {'type': 'INVERSION', 'severity': 'HIGH',
                'detail': f'head_y={head_y:.2f} foot_y={avg_foot_y:.2f}'}
    return None

def check_climbing(landmarks):
    """Wrist AND knee both elevated above hip = climbing posture."""
    _, lw_y, lw_v  = get_kp(landmarks, 'LEFT_WRIST')
    _, rw_y, rw_v  = get_kp(landmarks, 'RIGHT_WRIST')
    _, lk_y, lk_v  = get_kp(landmarks, 'LEFT_KNEE')
    _, rk_y, rk_v  = get_kp(landmarks, 'RIGHT_KNEE')
    _, lh_y, lh_v  = get_kp(landmarks, 'LEFT_HIP')
    _, rh_y, rh_v  = get_kp(landmarks, 'RIGHT_HIP')

    if not (kp_confident(lh_v) or kp_confident(rh_v)):
        return None

    hip_ys  = [y for y,v in [(lh_y,lh_v),(rh_y,rh_v)] if kp_confident(v)]
    avg_hip = sum(hip_ys) / len(hip_ys)

    wrist_elevated = any(
        y < avg_hip and kp_confident(v)
        for y, v in [(lw_y,lw_v),(rw_y,rw_v)]
    )
    knee_elevated  = any(
        y < avg_hip and kp_confident(v)
        for y, v in [(lk_y,lk_v),(rk_y,rk_v)]
    )

    if wrist_elevated and knee_elevated:
        return {'type': 'CLIMBING', 'severity': 'HIGH',
                'detail': f'wrists+knees above hip_y={avg_hip:.2f}'}
    return None

def check_airborne(landmarks, floor_y_baseline):
    """Both ankles above floor baseline = possible jump/airborne."""
    _, la_y, la_v = get_kp(landmarks, 'LEFT_ANKLE')
    _, ra_y, ra_v = get_kp(landmarks, 'RIGHT_ANKLE')

    if not (kp_confident(la_v) and kp_confident(ra_v)):
        return None

    # y < floor_baseline means higher in frame = off floor
    margin = 0.08  # must be 8% of frame above baseline to count
    if la_y < (floor_y_baseline - margin) and ra_y < (floor_y_baseline - margin):
        return {'type': 'AIRBORNE', 'severity': 'MEDIUM',
                'detail': f'ankles at y={la_y:.2f},{ra_y:.2f} '
                          f'floor={floor_y_baseline:.2f}'}
    return None

# ── Category C: Motion-based ─────────────────────────────────────────────────

def check_rapid_descent(landmarks, prev_hip_y):
    """Hip Y drops fast between frames = rapid downward movement."""
    if prev_hip_y is None:
        return None

    _, lh_y, lh_v = get_kp(landmarks, 'LEFT_HIP')
    _, rh_y, rh_v = get_kp(landmarks, 'RIGHT_HIP')

    hip_ys = [y for y,v in [(lh_y,lh_v),(rh_y,rh_v)] if kp_confident(v)]
    if not hip_ys:
        return None

    curr_hip_y = sum(hip_ys) / len(hip_ys)
    delta      = curr_hip_y - prev_hip_y  # positive = moving down in frame

    DESCENT_THRESHOLD = 0.08  # 8% of frame height per frame
    if delta > DESCENT_THRESHOLD:
        return {'type': 'RAPID_DESCENT', 'severity': 'HIGH',
                'detail': f'hip_y delta={delta:.3f} threshold={DESCENT_THRESHOLD}'}
    return curr_hip_y

# ── Master evaluator ─────────────────────────────────────────────────────────

def evaluate_frame(pose_result, zones, frame_w, frame_h,
                   floor_y_baseline, prev_hip_y):
    """
    Run all alert checks. Returns:
      alerts       : list of alert dicts
      curr_hip_y   : updated hip y for next frame motion check
    """
    if not pose_result.pose_landmarks:
        return [], prev_hip_y

    lmks   = pose_result.pose_landmarks
    alerts = []

    # Category A
    alerts += check_zone_alerts(lmks, zones, frame_w, frame_h)

    # Category B
    for check_fn in [check_inversion, check_climbing]:
        result = check_fn(lmks)
        if result:
            alerts.append(result)

    airborne = check_airborne(lmks, floor_y_baseline)
    if airborne:
        alerts.append(airborne)

    # Category C
    descent = check_rapid_descent(lmks, prev_hip_y)
    if isinstance(descent, dict):       # it's an alert
        alerts.append(descent)
        curr_hip_y = prev_hip_y
    elif descent is not None:           # it's the updated hip_y float
        curr_hip_y = descent
    else:
        curr_hip_y = prev_hip_y

    return alerts, curr_hip_y

# ── Visualisation ─────────────────────────────────────────────────────────────

ALERT_COLORS = {
    'HIGH'  : (0,   0, 220),
    'MEDIUM': (0, 140, 255),
    'LOW'   : (0, 200, 200),
}

def draw_zones_on_frame(frame, zones):
    for zone_name, zone_data in zones.items():
        box   = zone_data['box']
        color = (0, 180, 80) if zone_name == 'kitchen' else (200, 140, 0)
        cv2.rectangle(frame, (box[0],box[1]), (box[2],box[3]), color, 1)
        cv2.putText(frame, zone_name, (box[0]+4, box[1]+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return frame

def draw_alert_overlay(frame, alerts, fps, frame_idx, timestamp_str):
    h, w = frame.shape[:2]

    # Header bar
    bar_h = 36 + len(alerts) * 26
    cv2.rectangle(frame, (0,0), (w, bar_h), (0,0,0), -1)

    cv2.putText(frame, f"FPS:{fps:.1f}  Frame:{frame_idx:04d}  {timestamp_str}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160,160,160), 1)

    if alerts:
        for i, alert in enumerate(alerts):
            severity = alert.get('severity', 'MEDIUM')
            color    = ALERT_COLORS[severity]
            text     = f"[{severity}] {alert['type']}"
            if 'zone' in alert:
                text += f" — {alert['zone']}"
            if 'detail' in alert:
                text += f"  ({alert['detail']})"
            cv2.putText(frame, text, (10, 40 + i*26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Red border for any HIGH alert
        if any(a.get('severity') == 'HIGH' for a in alerts):
            cv2.rectangle(frame, (0,0), (w-1,h-1), (0,0,220), 6)
    else:
        cv2.putText(frame, "CLEAR", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0), 2)
        cv2.rectangle(frame, (0,0), (w-1,h-1), (0,180,0), 3)

    return frame

# ── Main monitor loop ─────────────────────────────────────────────────────────
import yaml

CONFIG_PATH  = os.path.expanduser('~/Documents/camonitor/scripts/config.yaml')
PROFILE_PATH = os.path.expanduser('~/Documents/camonitor/scripts/adult_profile.yaml')

def load_config():
    if not os.path.exists(CONFIG_PATH):
        # Defaults if no config file
        return {'mode': 'ADULT_TEST', 'adult_filter': {'height_span_threshold': 0.65,
                'match_tolerance': 0.20},
                'alerts': {'zone_entry_frames': 5,
                           'inversion_buffer': 0.05,
                           'descent_threshold': 0.05}}
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def load_adult_profile():
    if not os.path.exists(PROFILE_PATH):
        return None
    with open(PROFILE_PATH) as f:
        return yaml.safe_load(f)

def is_adult(landmarks, config, profile, debug=False):
    """
    Orientation-independent adult filter for top-down camera.

    Uses bounding box area of visible keypoints — an adult body always
    occupies more 2D frame area than a toddler regardless of orientation
    (standing, sitting, crouching, side-on, lying down).

    Threshold tuning:
      Run in ADULT_TEST mode and watch [DEBUG] bbox lines in terminal.
      Note your area values in various poses, then note toddler values.
      Set bbox_area_threshold between the two — halfway is a good start.

    Typical values from top-down ~2m height:
      Adult standing/sitting : 0.08 – 0.20
      Toddler standing       : 0.02 – 0.05
      Toddler crawling       : 0.02 – 0.04
    """
    area_threshold = config['adult_filter'].get('bbox_area_threshold', 0.07)

    # Keypoints to use — upper body reliable from top-down view
    kp_names = [
        'NOSE',
        'LEFT_SHOULDER',  'RIGHT_SHOULDER',
        'LEFT_ELBOW',     'RIGHT_ELBOW',
        'LEFT_WRIST',     'RIGHT_WRIST',
        'LEFT_HIP',       'RIGHT_HIP',
        'LEFT_KNEE',      'RIGHT_KNEE',
    ]

    visible_x = []
    visible_y = []
    for name in kp_names:
        x, y, v = get_kp(landmarks, name)
        if v > 0.5:
            visible_x.append(x)
            visible_y.append(y)

    # Need at least 4 keypoints for a reliable bounding box
    if len(visible_x) < 4:
        return True   # too few keypoints — default to adult (safe)

    width  = max(visible_x) - min(visible_x)
    height = max(visible_y) - min(visible_y)
    area   = width * height

    if debug:
        print(f"  [DEBUG] bbox area={area:.4f}  w={width:.3f}  h={height:.3f}  "
              f"threshold={area_threshold}  adult={area > area_threshold}")

    return area > area_threshold
    
def run_monitor(duration_sec=None):
    zones   = load_zones()
    emailer = AlertEmailer()
    cfg     = load_config()
    profile = load_adult_profile()
    mode    = cfg.get('mode', 'ADULT_TEST')
    debug   = cfg.get('debug', False)

    if not zones:
        return

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 10)

    for _ in range(15):
        cap.read()

    frame_h, frame_w = 720, 1280
    floor_y_baseline = 0.90

    prev_hip_y   = None
    results_log  = []
    frame_count  = 0
    alert_count  = 0
    last_cleanup = time.time()

    window_ok = False
    try:
        cv2.namedWindow('CaMonitor — Alert Mode', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('CaMonitor — Alert Mode', 960, 540)
        window_ok = True
    except Exception:
        pass

    if duration_sec:
        print(f"\nMonitoring started — {duration_sec}s run")
    else:
        print(f"\nMonitoring started — continuous (mode: {mode})")
    print("Press Q to quit early\n")

    mp_cfg = cfg.get('mediapipe', {})
    with mp_pose.Pose(
        min_detection_confidence=mp_cfg.get('min_detection_confidence', 0.55),
        min_tracking_confidence =mp_cfg.get('min_tracking_confidence',  0.55),
        model_complexity        =mp_cfg.get('model_complexity', 0)
    ) as pose:

        t_start = time.time()

        while True:
            # Stop if duration reached (only when duration_sec is set)
            if duration_sec and (time.time() - t_start) >= duration_sec:
                break

            t0  = time.time()
            ret, frame = cap.read()
            cap.grab()

            if not ret:
                continue

            now           = datetime.datetime.now()
            timestamp_str = now.strftime('%Y%m%d_%H%M%S_%f')[:-3]
            timestamp_iso = now.isoformat(timespec='milliseconds')

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            fps  = 1.0 / max(time.time() - t0, 0.001)
            temp = get_temp()
            cpu  = psutil.cpu_percent(interval=None)

            detected = result.pose_landmarks is not None

            # Adult filter — skip alerts if adult detected in CHILD_MONITOR mode
            skip_alerts = False
            if detected and mode == 'CHILD_MONITOR':
                if is_adult(result.pose_landmarks, cfg, profile, debug=debug):
                    skip_alerts = True

            # Idle sleep when no person — reduces CPU heat during quiet periods
            if not detected:
                time.sleep(0.3)

            # Draw skeleton
            if detected:
                mp_draw.draw_landmarks(
                    frame, result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0,255,0),
                                        thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=(0,180,255), thickness=2)
                )

            # Draw zone outlines
            frame = draw_zones_on_frame(frame, zones)

            # Evaluate alerts
            if detected and not skip_alerts:
                alerts, prev_hip_y = evaluate_frame(
                    result, zones, frame_w, frame_h,
                    floor_y_baseline, prev_hip_y)
            else:
                alerts = []

            # Draw alert overlay
            frame = draw_alert_overlay(
                frame, alerts, fps, frame_count, timestamp_str)

            # Save frame only on MEDIUM or HIGH alert
            has_alert = len(alerts) > 0
            if has_alert:
                severities = [a['severity'] for a in alerts]
                if any(s in ('MEDIUM', 'HIGH') for s in severities):
                    filename  = (f"frame{frame_count:04d}_{timestamp_str}"
                                 f"_ALERT_{'_'.join(a['type'] for a in alerts)}.jpg")
                    save_path = os.path.join(ALERT_DIR, filename)
                    cv2.imwrite(save_path, frame)

                alert_count += len(alerts)
                for alert in alerts:
                    print(f"  [{alert['severity']}] {alert['type']}"
                          f"{' — '+alert.get('zone','')}"
                          f"  {timestamp_iso}")

                # Email notification — path matches saved frame
                last_img = os.path.join(IMG_DIR,
                    f"frame{frame_count:04d}_{timestamp_str}"
                    f"_ALERT_{'_'.join(a['type'] for a in alerts)}.jpg")
                emailer.send_alert(alerts, last_img, timestamp_iso)

            results_log.append({
                'frame'      : frame_count,
                'timestamp'  : timestamp_iso,
                'fps'        : round(fps, 2),
                'cpu_temp'   : temp,
                'cpu_pct'    : cpu,
                'detected'   : detected,
                'alert_count': len(alerts),
                'alert_types': '+'.join(a['type'] for a in alerts) if alerts else 'none',
                'severity'   : '+'.join(a['severity'] for a in alerts) if alerts else 'none'
            })

            # Hourly cleanup of old alert images
            if time.time() - last_cleanup > 3600:
                cleanup_old_alerts()
                last_cleanup = time.time()

            if window_ok:
                if frame_count % 3 == 0:
                    display_frame = cv2.resize(frame, (640, 360))
                    cv2.imshow('CaMonitor — Alert Mode', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_count += 1

    cap.release()
    if window_ok:
        cv2.destroyAllWindows()

    # Append to CSV (not overwrite) so logs accumulate across runs
    file_exists = os.path.exists(LOG_PATH)
    fieldnames  = ['frame','timestamp','fps','cpu_temp','cpu_pct',
                   'detected','alert_count','alert_types','severity']
    with open(LOG_PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(results_log)

    print(f"\nMonitoring complete.")
    print(f"Total frames   : {frame_count}")
    print(f"Total alerts   : {alert_count}")
    print(f"CSV saved      : {LOG_PATH}")
    print(f"Images saved   : {IMG_DIR}")


if __name__ == '__main__':
    run_monitor()
