"""
This file loads the trained PyTorch model.

Args:
    Frame of an image.

Returns:
    Predicted fruit ripeness decision.
"""

import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision import models, transforms


class ImageClassifier:
    """
    Loads the trained MobileNetV3-Small model.

    Example:
        model = ImageClassifier("Fruit Ripeness V1.pth")

        image = cv2.imread("image.jpg")

        decision = model.predict(image)

        print(decision)
    """

    def __init__(self, model_path: str):

        # These MUST match dataset.classes from training
        self.class_names = [
    "freshapples",     # 0
    "freshbanana",     # 1
    "freshoranges",    # 2
    "rottenapples",    # 3
    "rottenbanana",    # 4
    "rottenoranges",   # 5
    "unripeapple",     # 6
    "unripe banana",   # 7
    "unripe orange",   # 8
]

        # Recreate the same model architecture
        self.model = models.mobilenet_v3_small(weights=None)

        # The trained model has 9 output classes
        self.model.classifier[3] = nn.Linear(
            self.model.classifier[3].in_features,
            len(self.class_names)
        )

        # Load trained weights
        state_dict = torch.load(
            model_path,
            map_location="cpu",
            weights_only=False
        )

        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Same preprocessing used during training
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess an OpenCV BGR image."""

        return self.transform(image).unsqueeze(0)

    def predict(self, image: np.ndarray) -> str:
        """Take an OpenCV image and return the predicted class."""

        input_tensor = self.preprocess(image)

        with torch.no_grad():
            output = self.model(input_tensor)

        predicted_idx = torch.argmax(output, dim=1).item()

        return self.class_names[predicted_idx]
