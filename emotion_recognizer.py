import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from typing import Tuple
from emotion_classifier import EmotionClassifier, EMOTION_CLASSES

class EmotionRecognizer:
    """
    Wraps the trained EmotionClassifier and preprocessing for a face image.
    """
    def __init__(self,
                 model_path: str = "emotion_classifier.pth",
                 device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        print("EmotionRecognizer using device:", self.device)

        # Load model
        self.model = EmotionClassifier(num_classes=len(EMOTION_CLASSES))
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print(f"Loaded emotion model from {model_path}")

        # Preprocessing to match training pipeline
        self.transform = transforms.Compose([
            transforms.ToPILImage(),             # expects H x W x C (RGB)
            transforms.Resize((75, 75)),        # ensure 75x75 to match training
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)
            ),
        ])

    def predict(self, face_bgr) -> Tuple[str, float, np.ndarray]:
        """
        Given a BGR face patch, returns:
        (predicted_label, confidence, probs_array)
        where probs_array has shape [num_classes].
        """
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

        # Preprocess
        face_tensor = self.transform(face_rgb)            # [3, 75, 75]
        face_tensor = face_tensor.unsqueeze(0).to(self.device)  # [1, 3, 75, 75]

        with torch.no_grad():
            logits = self.model(face_tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx = int(np.argmax(probs))
        label = EMOTION_CLASSES[pred_idx]
        conf = float(probs[pred_idx])
        return label, conf, probs