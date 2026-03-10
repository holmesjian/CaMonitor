import cv2
import os
import yaml
import time
import numpy as np
from ultralytics import YOLO

os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ.setdefault('DISPLAY', ':0')

YAML_PATH = os.path.expanduser('~/Documents/camonitor/scripts/zones_config.yaml')
IMG_PATH  = os.path.expanduser('~/Documents/camonitor/data/room_scan_result.jpg')

# Objects treated as individual danger zones
FURNITURE_OBJECTS = ['sofa', 'bed', 'chair', 'dining table']

# Objects used as kitchen anchors
KITCHEN_ANCHORS   = ['refrigerator', 'sink', 'oven', 'microwave']
KITCHEN_MIN_ANCHORS = 2   # minimum anchors needed to create kitchen zone
KITCHEN_PADDING     = 0.15  # 15% expansion around hull

def expand_box(box, padding, frame_w, frame_h):
    """Expand a bounding box by padding fraction, clamped to frame bounds."""
    x1, y1, x2, y2 = box
    pw = (x2 - x1) * padding
    ph = (y2 - y1) * padding
    return [
        max(0,       int(x1 - pw)),
        max(0,       int(y1 - ph)),
        min(frame_w, int(x2 + pw)),
        min(frame_h, int(y2 + ph))
    ]

def compute_kitchen_zone(anchor_boxes, frame_w, frame_h):
    """
    Given a list of anchor bounding boxes, compute the unified kitchen zone.
    Returns a single bounding box enclosing all anchors + padding.
    """
    all_x1 = [b[0] for b in anchor_boxes]
    all_y1 = [b[1] for b in anchor_boxes]
    all_x2 = [b[2] for b in anchor_boxes]
    all_y2 = [b[3] for b in anchor_boxes]

    hull_box = [min(all_x1), min(all_y1), max(all_x2), max(all_y2)]
    return expand_box(hull_box, KITCHEN_PADDING, frame_w, frame_h)

def draw_zone(frame, box, label, color, alpha=0.25):
    """Draw a filled semi-transparent zone rectangle with label."""
    x1, y1, x2, y2 = box
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(frame, (x1, y1 - 24), (x1 + len(label)*11 + 8, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame

def scan_room():
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8s.pt')

    print("Opening camera for room scan...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Warm up camera
    for _ in range(15):
        cap.read()

    print("Capturing room scan frame...")
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("ERROR: Could not capture frame.")
        return

    frame_h, frame_w = frame.shape[:2]
    print(f"Frame captured: {frame_w}x{frame_h}")

    # Run YOLO inference on the room scan frame
    print("Running YOLO object detection...")
    results = model(frame, conf=0.07, verbose=False)
    detections = results[0].boxes

    # Colour map for zone types
    zone_colors = {
        'sofa'        : (255, 100,   0),
        'bed'         : (255,   0, 150),
        'chair'       : (255, 180,   0),
        'dining table': (200, 100, 255),
        'kitchen'     : (0,   200, 100),
    }

    zones         = {}   # final zones dict to save
    kitchen_found = {}   # anchor_name → box

    vis_frame = frame.copy()

    for box in detections:
        cls_id    = int(box.cls[0])
        cls_name  = model.names[cls_id]
        conf      = float(box.conf[0])
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        det_box   = [x1, y1, x2, y2]

        print(f"  Detected: {cls_name:20s} conf={conf:.2f}  box={det_box}")

        # Furniture — direct danger zone
        if cls_name in FURNITURE_OBJECTS:
            expanded = expand_box(det_box, 0.05, frame_w, frame_h)
            # Count existing zones of same type to avoid overwriting
            base_key = cls_name.replace(' ', '_')
            existing = [k for k in zones if k.startswith(base_key)]
            zone_key = base_key if not existing else f"{base_key}_{len(existing)}"
            zones[zone_key] = {
                'box'    : expanded,
                'type'   : 'furniture',
                'source' : cls_name,
                'conf'   : round(conf, 3)
            }
            color = zone_colors.get(cls_name, (200, 200, 0))
            vis_frame = draw_zone(vis_frame, expanded, cls_name, color)

        # Kitchen anchors — collect for hull computation
        if cls_name in KITCHEN_ANCHORS:
            kitchen_found[cls_name] = det_box
            # Draw anchor markers
            cv2.rectangle(vis_frame, (x1,y1), (x2,y2), (0,200,100), 2)
            cv2.putText(vis_frame, f"[kitchen anchor: {cls_name}]",
                        (x1+4, y1-8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,200,100), 1)

    # Kitchen zone — compute from anchors
    print(f"\nKitchen anchors found: {list(kitchen_found.keys())}")

    if len(kitchen_found) >= KITCHEN_MIN_ANCHORS:
        kitchen_box = compute_kitchen_zone(
            list(kitchen_found.values()), frame_w, frame_h)
        zones['kitchen'] = {
            'box'     : kitchen_box,
            'type'    : 'kitchen',
            'anchors' : list(kitchen_found.keys()),
            'conf'    : 'multi-anchor'
        }
        vis_frame = draw_zone(vis_frame, kitchen_box,
                              f"kitchen ({len(kitchen_found)} anchors)",
                              zone_colors['kitchen'])
        print(f"  Kitchen zone created from: {list(kitchen_found.keys())}")
        print(f"  Kitchen box: {kitchen_box}")

    elif len(kitchen_found) == 1:
        print(f"  WARNING: Only 1 kitchen anchor detected — need {KITCHEN_MIN_ANCHORS}.")
        print(f"  No kitchen zone created. Reposition camera to include more appliances.")

    else:
        print("  No kitchen anchors detected. No kitchen zone created.")

    # Summary
    print(f"\nZones created: {list(zones.keys())}")

    if not zones:
        print("WARNING: No zones detected. Check camera view or lower conf threshold.")
        return

    # Add frame dimensions to YAML for coordinate validation at runtime
    zones['_frame_info'] = {
        'width' : frame_w,
        'height': frame_h
    }

    # Save to YAML
    with open(YAML_PATH, 'w') as f:
        yaml.dump(zones, f, default_flow_style=False)
    print(f"\nZones saved to: {YAML_PATH}")

    # Add scan info overlay to visualisation
    cv2.rectangle(vis_frame, (0, frame_h-40), (frame_w, frame_h), (0,0,0), -1)
    cv2.putText(vis_frame,
                f"Room scan complete — {len(zones)-1} zones detected. "
                f"Press any key to close.",
                (12, frame_h-12), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200,200,200), 1)

    # Save visualisation
    cv2.imwrite(IMG_PATH, vis_frame)
    print(f"Scan visualisation saved to: {IMG_PATH}")

    # Show result
    try:
        cv2.imshow('Room Scan Result', vis_frame)
        cv2.waitKey(10)
        cv2.destroyAllWindows()
    except Exception:
        print("No display — check saved image at data/room_scan_result.jpg")

if __name__ == '__main__':
    scan_room()
