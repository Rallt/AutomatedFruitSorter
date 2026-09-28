"""
This file:
- Main file for the whole process
- System health check
- System Engine
"""

import logging
from pathlib import Path

import cv2

from FruitClassifier import FruitClassifier
from frameCapture import get_frames


logging.basicConfig(level=logging.INFO)


def log(text: str):
    logging.info(text)
    return text


def main():
    log("Starting system...")

    # Start capturing frames
    frames = get_frames()

    # Load the current Keras fruit/vegetable classifier.
    root = Path(__file__).resolve().parents[1]
    model = FruitClassifier(
        str(root / "models" / "FruitClassifier.keras"),
        str(root / "models" / "FruitClassifier.json"),
    )

    log("System started.")

    # Process each captured frame
    for frame in frames:
        cv2.imshow("Camera", frame)
        decision = model.predict(frame)
        log(f"Decision: {decision}")
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
