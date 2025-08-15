import cv2
import numpy as np
import time

CAMERA_INDEX = 0
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)

SHARPNESS_THRESHOLD = 80
MOTION_THRESHOLD = 500  # Tune for hand movement
CAPTURE_DELAY = 1.0  # Seconds to wait after hand removed

last_frame = None
motion_end_time = 0
captured = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    motion_detected = False
    if last_frame is not None:
        diff = cv2.absdiff(gray, last_frame)
        motion_level = np.mean(diff)
        if motion_level > MOTION_THRESHOLD:
            motion_detected = True
            motion_end_time = time.time() + CAPTURE_DELAY
            captured = False

    # Decide what to display and capture
    if motion_detected:
        status_text = "Hand detected!"
        color = (0, 0, 255)  # Red
    elif time.time() > motion_end_time and laplacian_var > SHARPNESS_THRESHOLD and not captured:
        timestamp = int(time.time())
        filename = f"page_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Captured: {filename}")
        status_text = "Captured!"
        color = (0, 255, 0)  # Green
        captured = True
    else:
        status_text = "Waiting..."
        color = (0, 255, 255)  # Yellow

    cv2.putText(frame, f"{status_text}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    cv2.imshow("Preview", frame)
    last_frame = gray.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
