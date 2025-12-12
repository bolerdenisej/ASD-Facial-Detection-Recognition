# webcam_emotion_game.py
import cv2
import random
import time

from face_detector import FaceDetector
from emotion_recognizer import EmotionRecognizer
from helpers import draw_detections
from emotion_classifier import EMOTION_CLASSES


def pick_new_target(current_target=None):
    """Pick a new random emotion, optionally different from current_target."""
    choices = EMOTION_CLASSES
    if current_target in choices and len(choices) > 1:
        choices = [e for e in choices if e != current_target]
    return random.choice(choices)


def main():
    # --- Initialize face detector and emotion recognizer ---
    face_detector = FaceDetector()
    recognizer = EmotionRecognizer(model_path="emotion_classifier.pth", device="auto")

    # --- Open webcam ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    print("Press 'q' to quit.")

    # --- Game state ---
    target_emotion = pick_new_target()
    score = 0
    last_match_time = 0.0
    last_score_increase_time = 0.0
    match_display_duration = 1.0  # seconds to show "Matched!" message
    score_flash_duration = 0.5  # seconds to show score in green when it increases
    confidence_threshold = 0.4   # only count match if conf >= 40%
    no_face_frame_count = 0
    no_face_threshold = 10  # Show "No face detected" after 10 consecutive frames

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        vis = frame.copy()

        # --- 1. Detect a single best face ---
        det = face_detector.detect_best_detection(frame)

        if det is None:
            no_face_frame_count += 1
            # Only show "No face detected" after threshold number of frames
            if no_face_frame_count >= no_face_threshold:
                cv2.putText(
                    vis,
                    "No face detected",
                    (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
        else:
            # Reset counter when face is detected
            no_face_frame_count = 0
            score_det, left, top, right, bottom = det

            # Draw box using helper
            detections = [det]
            vis = draw_detections(vis, detections, ground_truth=[])

            # Crop face and predict emotion
            face_bgr = frame[top:bottom, left:right]
            if face_bgr.size != 0:
                label, conf, _ = recognizer.predict(face_bgr)

                # Show current prediction on screen
                pred_text = f"{label} ({conf*100:.1f}%)"
                cv2.putText(
                    vis,
                    pred_text,
                    (left, bottom + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

                # --- 2. Check if prediction matches target emotion ---
                now = time.time()
                if label == target_emotion and conf >= confidence_threshold:
                    score += 1
                    last_match_time = now
                    last_score_increase_time = now
                    # Pick a new target emotion
                    target_emotion = pick_new_target(target_emotion)

        # --- 3. Draw game HUD (target, score, match notification) ---
        # Target emotion
        target_text = f"Target: {target_emotion.upper()}"
        cv2.putText(
            vis,
            target_text,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        # Score (green when just increased, black otherwise)
        score_text = f"Score: {score}"
        now = time.time()
        if now - last_score_increase_time < score_flash_duration:
            score_color = (0, 255, 0)  # Green when just increased
        else:
            score_color = (0, 0, 0)  # Black otherwise
        cv2.putText(
            vis,
            score_text,
            (30, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            score_color,
            2,
            cv2.LINE_AA,
        )

        # If we recently matched, show a "Matched!" banner
        if time.time() - last_match_time < match_display_duration:
            cv2.putText(
                vis,
                "Matched!",
                (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 200, 0),
                2,
                cv2.LINE_AA,
            )

        # --- 4. Show frame ---
        cv2.imshow("ASD Learning Tool: Emotion", vis)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
