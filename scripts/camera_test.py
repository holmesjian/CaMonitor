# For camonitor to test camera
import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)

ret, frame = cap.read()

if not cap.isOpened():
    print("Error: Camera not found")
    
else:
    ret, frame = cap.read()
    if ret:
        print(f"Camera OK - Frame shape: {frame.shape}")
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"FPS setting: {actual_fps}")
        
    cap.release()
