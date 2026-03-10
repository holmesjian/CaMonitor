import cv2
import time
import psutil
import csv
import os

THERMAL_PATH = '/sys/class/thermal/thermal_zone0/temp'
LOG_PATH = os.path.expanduser('~/Documents/camonitor/logs/benchmark_baseline.csv')

def get_temp():
    return round(float(open(THERMAL_PATH).read()) / 1000, 1)

def run_benchmark(width, height, duration_sec=30, label=''):
    # FIX 1: wait for camera to fully release before reopening
    time.sleep(3)

    cap = cv2.VideoCapture(0)

    # FIX 2: set MJPG format FIRST before setting resolution
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 60)

    # Warm up camera — discard first 10 frames
    for _ in range(10):
        cap.read()

    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"\n--- {label} ---")
    print(f"Requested: {width}x{height} | Actual: {int(actual_w)}x{int(actual_h)}")

    results = []
    t_start = time.time()
    frame_count = 0

    while time.time() - t_start < duration_sec:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            continue
        elapsed = time.time() - t0
        fps = 1.0 / elapsed if elapsed > 0 else 0

        temp = get_temp()
        cpu  = psutil.cpu_percent(interval=None)

        results.append({
            'label'    : label,
            'frame'    : frame_count,
            'fps'      : round(fps, 2),
            'cpu_temp' : temp,
            'cpu_pct'  : cpu
        })
        frame_count += 1

    cap.release()

    avg_fps   = round(sum(r['fps'] for r in results) / len(results), 2)
    avg_temp  = round(sum(r['cpu_temp'] for r in results) / len(results), 1)
    avg_cpu   = round(sum(r['cpu_pct'] for r in results) / len(results), 1)
    peak_temp = max(r['cpu_temp'] for r in results)

    print(f"Avg FPS   : {avg_fps}")
    print(f"Avg Temp  : {avg_temp}°C  |  Peak Temp: {peak_temp}°C")
    print(f"Avg CPU   : {avg_cpu}%")

    return results

def save_csv(all_results):
    with open(LOG_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['label','frame','fps','cpu_temp','cpu_pct'])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nLog saved to: {LOG_PATH}")

if __name__ == '__main__':
    all_results = []
    all_results += run_benchmark(640,  480,  label='480p')
    all_results += run_benchmark(1280, 720,  label='720p')
    all_results += run_benchmark(1920, 1080, label='1080p')
    save_csv(all_results)
    print("\nPhase 1 complete. Check logs/benchmark_baseline.csv for results.")
