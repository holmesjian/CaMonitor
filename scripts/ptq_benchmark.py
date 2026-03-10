import cv2
import mediapipe as mp
import time
import csv
import numpy as np

mp_pose = mp.solutions.pose

def benchmark_complexity(complexity, n_frames=100):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    #cap.set(cv2.CAP_PROP_FPS, 10)

    for _ in range(10):
        cap.read()

    fps_list        = []
    confidence_list = []
    detection_count = 0

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=complexity
    ) as pose:
        for i in range(n_frames):
            t0 = time.time()
            ret, frame = cap.read()
            cap.grab()
            if not ret:
                continue

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            elapsed = time.time() - t0
            fps_list.append(1.0 / max(elapsed, 0.001))

            if result.pose_landmarks:
                detection_count += 1
                key_lmks = [0, 11, 12, 23, 24]  # nose, shoulders, hips
                vis = np.mean([
                    result.pose_landmarks.landmark[i].visibility
                    for i in key_lmks
                ])
                confidence_list.append(vis)

            if i % 20 == 0:
                print(f"  complexity={complexity}  "
                      f"frame={i}/{n_frames}  "
                      f"fps={fps_list[-1]:.1f}")

    cap.release()

    return {
        'complexity'     : complexity,
        'mean_fps'       : round(np.mean(fps_list), 2),
        'detection_rate' : round(detection_count / n_frames * 100, 1),
        'mean_confidence': round(np.mean(confidence_list) if confidence_list else 0, 3),
        'n_frames'       : n_frames
    }


def run():
    results = []
    for c in [0, 1, 2]:
        print(f"\nBenchmarking complexity={c}")
        print("Stand in frame and move naturally...")
        input("Press Enter when ready...")
        r = benchmark_complexity(c, n_frames=100)
        results.append(r)
        print(f"  → FPS={r['mean_fps']}  "
              f"detection={r['detection_rate']}%  "
              f"confidence={r['mean_confidence']}")

    with open('../logs/ptq_benchmark.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\n── Summary ───────────────────────────────────────────")
    print(f"{'Complexity':<12} {'FPS':<8} {'Detection%':<14} {'Confidence'}")
    print(f"{'─'*10:<12} {'─'*6:<8} {'─'*10:<14} {'─'*10}")
    for r in results:
        print(f"{r['complexity']:<12} {r['mean_fps']:<8} "
              f"{r['detection_rate']:<14} {r['mean_confidence']}")
    print("\nSaved to logs/ptq_benchmark.csv")


if __name__ == '__main__':
    run()
