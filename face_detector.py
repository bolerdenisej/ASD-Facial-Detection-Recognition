import cv2
from typing import List, Optional, Tuple
from helpers import non_max_suppression

# Detection tuple: (score, left, top, right, bottom)
Detection = Tuple[float, int, int, int, int]

class FaceDetector:
    """
    Simple face detector wrapper around OpenCV Haar cascades.
    Uses helpers.non_max_suppression to clean up detections.
    """
    def __init__(self,
                 cascade_path: str = None,
                 scale_factor: float = 1.3,
                 min_neighbors: int = 5,
                 min_size=(60, 60),
                 iou_threshold: float = 0.3):
        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f"Could not load Haar cascade from {cascade_path}")

        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.iou_threshold = iou_threshold

    def detect_detections(self, frame_bgr) -> List[Detection]:
        """
        Detect faces in a BGR frame and return a list of detection tuples
        (score, left, top, right, bottom). Haar doesn't provide a real score,
        so we set score=1.0 for all detections.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )

        detections: List[Detection] = []
        for (x, y, w, h) in faces:
            left, top = int(x), int(y)
            right, bottom = int(x + w), int(y + h)
            detections.append((1.0, left, top, right, bottom))
        return detections

    def detect_best_detection(self, frame_bgr) -> Optional[Detection]:
        """
        Detect faces, apply NMS using helpers.non_max_suppression,
        and return the best detection (highest score after NMS).
        Returns None if no face is found.
        """
        detections = self.detect_detections(frame_bgr)
        if not detections:
            return None

        # Apply NMS (uses IoU helper internally).
        detections_nms = non_max_suppression(detections, iou_threshold=self.iou_threshold)
        if not detections_nms:
            return None

        # Since scores are all 1.0, non_max_suppression already sorted them;
        # we'll just take the first (best) detected face.
        return detections_nms[0]