"""Keras fruit and vegetable classifier used by the web sorter."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


class FruitClassifier:
    """Loads ``FruitClassifier.keras`` and returns its most likely label."""

    def __init__(self, model_path: str, classes_path: str):
        self.class_names = json.loads(Path(classes_path).read_text(encoding="utf-8"))
        self.model = tf.keras.models.load_model(model_path, compile=False)

    def predict(self, image: np.ndarray) -> str:
        """Classify an OpenCV BGR frame using the model's 224px RGB input."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_image, (224, 224), interpolation=cv2.INTER_AREA)
        batch = np.expand_dims(resized.astype(np.float32), axis=0)
        probabilities = self.model.predict(batch, verbose=0)
        index = int(np.argmax(probabilities[0]))
        return self.class_names[index]
