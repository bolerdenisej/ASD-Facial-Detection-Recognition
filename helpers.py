"""
Helpers shared by detectors:
- Intersection over Union (IoU) and Non-Maximum Suppression (NMS)
- Evaluation against ground truth
- Drawing detections

Detections are tuples: (score, left, top, right, bottom[, scale])
Ground-truth boxes are (left, top, right, bottom)
"""

import cv2
import numpy as np


def iou(box1, box2):
    """
    Compute Intersection over Union (IoU) for two boxes.
    IoU is the area of overlap divided by the area of union between the boxes.
    Boxes are (left, top, right, bottom). Returns a float in [0, 1].
    """
    # --- YOUR CODE HERE ---
    if len(box1) > 4:
        box1 = box1[-4:]
    if len(box2) > 4:
        box2 = box2[-4:]

    left1, top1, right1, bottom1 = box1
    left2, top2, right2, bottom2 = box2

    intersectionLeft = max(left1, left2)
    intersectionTop = max(top1, top2)
    intersectionRight = min(right1, right2)
    intersectionBottom = min(bottom1, bottom2)
    intersectionWidth = max(0, intersectionRight - intersectionLeft)
    intersectionHeight = max(0, intersectionBottom - intersectionTop)
    intersectionArea = intersectionWidth * intersectionHeight

    area1 = max(0, right1 - left1) * max(0, bottom1 - top1)
    area2 = max(0, right2 - left2) * max(0, bottom2 - top2)

    union = area1 + area2 - intersectionArea
    if union <= 0:
        return 0.0;

    return intersectionArea / union


def non_max_suppression(detections, iou_threshold=0.3):
    """
    Greedy Non-Maximum Suppression (NMS) on detection tuples (score, left, top, right, bottom[, scale]).
    Sorts by score descending and removes boxes with Intersection over Union (IoU) >= iou_threshold
    relative to any higher-scoring box. Returns filtered detections.
    """
    if not detections:
        return []
    detections = sorted(detections, key=lambda x: x[0], reverse=True)
    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)
        detections = [d for d in detections if iou(best[1:5], d[1:5]) < iou_threshold]
    return keep


def evaluate_detection(detections, ground_truth):
    """
    Evaluate detections against ground truth with 1-to-1 greedy matching.
    A detection matches a ground-truth box if Intersection over Union (IoU) >= 0.5
    and that ground-truth box is not already matched. Returns (num_correct, precision, recall).
    """
    correct = 0
    matched = set()
    for det in detections:
        for i, gt in enumerate(ground_truth):
            box = det[1:5]
            if i not in matched and iou(box, gt) >= 0.5:
                correct += 1
                matched.add(i)
                break
    precision = correct / len(detections) if detections else 0
    recall = correct / len(ground_truth) if ground_truth else 0
    return correct, precision, recall


def draw_detections(image, detections, ground_truth):
    """
    Draw ground truth (green) and detections (blue) on an image.
    Each detection is labeled with confidence and optional scale (s=...).
    Accepts grayscale or BGR input; returns a BGR visualization.
    """
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    for (l, t, r, b) in ground_truth:
        cv2.rectangle(vis, (l, t), (r, b), (0, 255, 0), 3)
    for det in detections:
        score, l, t, r, b = det[:5]
        scale_txt = ''
        if len(det) >= 6:
            scale_txt = f' s={det[5]:.2f}'
        cv2.rectangle(vis, (l, t), (r, b), (255, 0, 0), 2)
        cv2.putText(vis, f'{score:.2f}{scale_txt}', (l, t-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return vis


