"""
using opencv to capture video frames and return each frame

"""


import cv2
from threading import Thread, Lock


frames = []
lock = Lock()


def capture_frames():
    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            with lock:
                frames.append(frame)

    finally:
        cap.release()


def get_frames():
    Thread(target=capture_frames, daemon=True).start()

    while True:
        with lock:
            frame = frames.pop(0) if frames else None

        if frame is not None:
            yield frame


# Testing
"""for frame in get_frames():
    cv2.imshow("Camera", frame)
    print(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break"""

#cv2.destroyAllWindows()
