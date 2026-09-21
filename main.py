"""
This file:
- Main file for the whole process
- System health check
- System Engine
"""

import logging

import cv2

import FruitStatusDetector
from frameCapture import get_frames


logging.basicConfig(level=logging.INFO)


def log(text: str):
    logging.info(text)
    return text


def main():
    log("Starting system...")

    # Start capturing frames
    frames = get_frames()

    # Load trained model
    model = FruitStatusDetector.ImageClassifier(
        r"C:\Users\TCFL\PycharmProjects\AutomatedFruitSorter\Fruit Ripeness V1.pth"
    )

    log("System started.")

    # Process each captured frame
    for frame in frames:



        for frame in get_frames():
            cv2.imshow("Camera", frame)
            decision = model.predict(frame)
            log(f"Decision: {decision}")
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    main()
