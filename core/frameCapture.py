"""
using opencv to capture video frames and return each frame

"""


import os
from threading import Lock, Thread

import cv2


frames = []
lock = Lock()


DEFAULT_CAMERA_URL = "http://10.11.222.250:8080/stream"


def camera_source() -> str:
    """Return the configured MJPEG camera URL.

    Set ``FRUIT_SORTER_CAMERA_URL`` to point the sorter at a different camera.
    """
    return os.getenv("FRUIT_SORTER_CAMERA_URL", DEFAULT_CAMERA_URL)


def capture_frames(source: str | None = None):
    cap = cv2.VideoCapture(source or camera_source())

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            with lock:
                frames.append(frame)

    finally:
        cap.release()


def get_frames(source: str | None = None):
    Thread(target=capture_frames, args=(source,), daemon=True).start()

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
