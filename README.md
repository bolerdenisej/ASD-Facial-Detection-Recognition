# ASD-Facial-Detection-Recognition

A real-time facial emotion recognition system designed as an interactive learning tool for ASD (Autism Spectrum Disorder). This project combines face detection with emotion classification to create an engaging game where users practice expressing different emotions.

## Features

- **Real-time Face Detection**: Uses OpenCV Haar Cascades for robust face detection
- **7 Emotion Classification**: Recognizes anger, disgust, fear, happy, neutral, sad, and surprise
- **Interactive Game Mode**: Gamified emotion recognition with scoring system
- **Deep Learning Model**: Custom CNN architecture trained on the RAF-DB dataset
- **Webcam Integration**: Real-time processing from webcam feed

## Project Structure

```
ASD-Facial-Detection-Recognition/
├── app_webcam.py              # Main application with interactive game
├── emotion_classifier.py      # CNN model definition and training script
├── emotion_classifier.pth     # Trained model weights
├── emotion_recognizer.py      # Emotion recognition wrapper
├── face_detector.py           # Face detection using Haar Cascades
├── helpers.py                 # Utility functions (IoU, NMS, visualization)
├── requirements.txt           # Python dependencies
├── raf_data/                  # Training dataset (needs to be extracted from raf_data.zip)
└── README.md                  # This file
```

## Installation

### Prerequisites

- Python 3.7+
- Webcam/camera access
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ASD-Facial-Detection-Recognition
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Extract the training data (if not already extracted):
```bash
unzip raf_data.zip
```

## Usage

### Running the Interactive Application

Launch the webcam-based emotion recognition game:

```bash
python app_webcam.py
```

**Game Instructions:**
- A target emotion will be displayed at the top of the screen
- Express the target emotion with your face
- When your emotion matches the target with ≥60% confidence, you score a point
- A new target emotion is selected after each match
- Press 'q' to quit

### Training the Model

Train or retrain the emotion classifier:

```bash
python emotion_classifier.py --data-dir ./raf_data --epochs 10 --device auto
```

**Arguments:**
- `--data-dir`: Path to the dataset directory (default: `./raf_data`)
- `--epochs`: Number of training epochs (default: 10)
- `--output-model`: Output path for saved model (default: `emotion_classifier.pth`)
- `--device`: Device to use for training (`auto`, `cuda`, or `cpu`)

The script will:
- Load training and test data from `raf_data/train/` and `raf_data/test/`
- Train the model with data augmentation
- Evaluate on the test set
- Save the trained model weights

## Technical Details

### Model Architecture

The `EmotionClassifier` uses a CNN with:
- **3 Convolutional Layers**: 32, 64, and 128 filters with BatchNorm
- **3 MaxPooling Layers**: 2x2 pooling for dimensionality reduction
- **2 Fully Connected Layers**: 256 hidden units with dropout (0.5)
- **Input Size**: 75x75 RGB images
- **Output**: 7 emotion classes

### Face Detection

- Uses OpenCV's Haar Cascade classifier for frontal face detection
- Includes Non-Maximum Suppression (NMS) to filter overlapping detections
- Configurable parameters for scale factor, min neighbors, and minimum face size

### Data Preprocessing

- Images are resized to 75x75 pixels
- Normalized to [-1, 1] range using mean=0.5, std=0.5
- Training includes augmentation: random crops, horizontal flips, rotation, and random erasing

## Emotion Classes

The model recognizes 7 emotions:
1. **Anger** 
2. **Disgust** 
3. **Fear** 
4. **Happy** 
5. **Neutral** 
6. **Sad** 
7. **Surprise** 

## Requirements

See `requirements.txt` for full list. Key dependencies:
- `torch` - PyTorch for deep learning
- `torchvision` - Image transforms and utilities
- `opencv-python-headless` - Computer vision operations
- `numpy` - Numerical operations
- `tqdm` - Progress bars
- `scikit-learn` - Evaluation metrics

## Notes

- The model requires `emotion_classifier.pth` to be present in the project root
- If the model file is missing, train it first using `emotion_classifier.py`
- Webcam index defaults to 0; modify `cv2.VideoCapture(0)` if needed
- For best results, ensure good lighting and face the camera directly

## Acknowledgments

- [RAF-DB (Balanced RAF-DB Dataset 75x75 Grayscale)](https://www.kaggle.com/datasets/dollyprajapati182/balanced-raf-db-dataset-7575-grayscale) for training dataset
