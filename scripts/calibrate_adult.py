import cv2
import mediapipe as mp
import numpy as np
import yaml
import os
import time

os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ.setdefault('DISPLAY', ':0')

PROFILE_PATH = os.path.expanduser(
    '~/Documents/camonitor/scripts/adult_profile.yaml')

mp_pose = mp.solutions.pose
LMKS    = mp_pose.PoseLandmark

def get_kp(landmarks, name):
    lm = landmarks.landmark[LMKS[name]]
    return lm.x, lm.y, lm.visibility

def compute_ratios(landmarks):
    """
    Compute scale-invariant body proportion ratios.
    All values are normalised 0-1 coordinates so ratios
    are distance-independent.
    """
    # Keypoints needed
    _, nose_y,   nose_v   = get_kp(landmarks, 'NOSE')
    lsx, _,      ls_v     = get_kp(landmarks, 'LEFT_SHOULDER')
    rsx, _,      rs_v     = get_kp(landmarks, 'RIGHT_SHOULDER')
    lhx, lhy,   lh_v     = get_kp(landmarks, 'LEFT_HIP')
    rhx, rhy,   rh_v     = get_kp(landmarks, 'RIGHT_HIP')
    _,   lank_y, lank_v  = get_kp(landmarks, 'LEFT_ANKLE')
    _,   rank_y, rank_v  = get_kp(landmarks, 'RIGHT_ANKLE')
    _,   lear_y, lear_v  = get_kp(landmarks, 'LEFT_EAR')
    _,   rear_y, rear_v  = get_kp(landmarks, 'RIGHT_EAR')

    # Require key landmarks visible
    required = [nose_v, ls_v, rs_v, lh_v, rh_v]
    if any(v < 0.6 for v in required):
        return None

    shoulder_width = abs(lsx - rsx)
    hip_width      = abs(lhx - rhx)
    avg_hip_y      = (lhy + rhy) / 2
    torso_length   = avg_hip_y - nose_y   # positive = torso goes down

    ankle_ys = [y for y,v in [(lank_y,lank_v),(rank_y,rank_v)] if v > 0.5]
    avg_ankle_y = sum(ankle_ys)/len(ankle_ys) if ankle_ys else None

    total_height = (avg_ankle_y - nose_y) if avg_ankle_y else None

    ear_ys  = [y for y,v in [(lear_y,lear_v),(rear_y,rear_v)] if v > 0.5]
    head_size = (avg_hip_y - min(ear_ys)) if ear_ys else None

    ratios = {}

    if shoulder_width > 0.01:
        ratios['hip_shoulder_ratio'] = round(hip_width / shoulder_width, 4)

    if torso_length > 0.01:
        ratios['shoulder_torso_ratio'] = round(shoulder_width / torso_length, 4)
        if head_size:
            ratios['head_torso_ratio'] = round(head_size / torso_length, 4)

    if total_height and total_height > 0.01:
        ratios['height_span'] = round(total_height, 4)
        ratios['leg_height_ratio'] = round(
            (avg_ankle_y - avg_hip_y) / total_height, 4)

    return ratios if len(ratios) >= 3 else None

def calibrate():
    print("Adult Profile Calibration")
    print("=" * 40)
    print("Stand naturally in front of camera.")
    print("Collecting 60 frames over ~8 seconds...")
    print("Stay at your normal monitoring distance.\n")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    for _ in range(15):
        cap.read()

    all_ratios = []
    collected  = 0
    target     = 60

    with mp_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        model_complexity=0
    ) as pose:

        while collected < target:
            ret, frame = cap.read()
            if not ret:
                continue

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if result.pose_landmarks:
                ratios = compute_ratios(result.pose_landmarks)
                if ratios:
                    all_ratios.append(ratios)
                    collected += 1
                    if collected % 10 == 0:
                        print(f"  Collected {collected}/{target} frames...")

            # Visual feedback
            if result.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS)
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0,0), (w, 50), (0,0,0), -1)
            cv2.putText(frame,
                        f"Calibrating: {collected}/{target} frames",
                        (10, 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0,220,0), 2)
            try:
                cv2.imshow('Adult Calibration', frame)
                cv2.waitKey(1)
            except Exception:
                pass

    cap.release()
    cv2.destroyAllWindows()

    if not all_ratios:
        print("ERROR: No valid frames collected. Check camera and lighting.")
        return

    # Average all ratio keys across collected frames
    all_keys = set(k for r in all_ratios for k in r)
    profile  = {}
    for key in all_keys:
        vals = [r[key] for r in all_ratios if key in r]
        profile[key] = {
            'mean' : round(float(np.mean(vals)), 4),
            'std'  : round(float(np.std(vals)),  4)
        }

    print(f"\nCalibration complete — {len(all_ratios)} frames used")
    print("\nAdult profile ratios:")
    for k, v in profile.items():
        print(f"  {k:30s}: mean={v['mean']:.4f}  std={v['std']:.4f}")

    with open(PROFILE_PATH, 'w') as f:
        yaml.dump(profile, f, default_flow_style=False)
    print(f"\nProfile saved: {PROFILE_PATH}")
    print("\nNext: set mode: CHILD_MONITOR in config.yaml to activate child monitoring.")

if __name__ == '__main__':
    calibrate()
