"""
Emotion classifier for 7 emotion classes using facial expressions.

Usage:
  python emotion_classifier.py --data-dir ./data --epochs 10 --device auto
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# --- 1. DATASET and AUGMENTATION ---

# Define the 7 emotion classes
EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASSES = len(EMOTION_CLASSES)

class EmotionDataset(Dataset):
    """Dataset wrapper for emotion classification with optional augmentation.
    Images are expected to be 100x100 RGB color pixels."""
    def __init__(self, images, labels, augment: bool = False):
        self.images = images
        self.labels = labels

        # Simple normalization for 3-channel images in [0,1]
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        
        # If augment is True, create a transform pipeline using transforms.Compose.
        # If augment is False, just convert to tensor + normalize.
        if augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.transform(self.images[idx])
        return image, self.labels[idx]

# --- 2. MODEL ARCHITECTURE ---

class EmotionClassifier(nn.Module):
    def __init__(self, num_classes: int = 7):
        super().__init__()
        # 3 conv layers, 3 pooling layers, 2 fully connected layers, 1 dropout layer.
        # Input images are 100x100 RGB (3 channels).
        self.conv1 = nn.Conv2d(3, 32, 5)   # 3 channels now
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)

        # Calculate the size after convolutions for the fully connected layer
        # Input: 100x100
        # conv1 (5x5, no padding): 100 - 5 + 1 = 96  -> pool -> 48
        # conv2 (5x5):             48  - 5 + 1 = 44  -> pool -> 22
        # conv3 (3x3):             22  - 3 + 1 = 20  -> pool -> 10
        # So we get 128 * 10 * 10 features.
        self.fc1 = nn.Linear(128 * 10 * 10, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Conv -> ReLU -> Pool (3x), then Flatten -> FC -> Dropout -> FC
        x = self.pool(F.relu(self.conv1(x)))  # -> [B, 32, 48, 48]
        x = self.pool(F.relu(self.conv2(x)))  # -> [B, 64, 22, 22]
        x = self.pool(F.relu(self.conv3(x)))  # -> [B, 128, 10, 10]

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# --- 3. DATA LOADING ---

def load_data(data_dir: str = './data'):
    """Load emotion images from train and test folders with 7 emotion classes."""
    def load_images_from_folder(folder_path: Path):
        """Load all images from a folder. Images are expected to be 100x100 RGB."""
        images = []
        image_files = sorted(folder_path.glob('*.jpg'))
        for img_path in image_files:
            try:
                img = Image.open(img_path)
                # Ensure RGB format
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images.append(np.array(img))
            except Exception as e:
                print(f"Warning: Could not load {img_path}: {e}")
                continue
        return np.array(images)
    
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    
    # Load training data
    train_dir = Path(data_dir) / 'train'
    for emotion_idx, emotion in enumerate(EMOTION_CLASSES):
        emotion_dir = train_dir / emotion
        if emotion_dir.exists():
            images = load_images_from_folder(emotion_dir)
            train_images.append(images)
            train_labels.append(np.full(len(images), emotion_idx, dtype=np.int64))
            print(f'Loaded {len(images)} {emotion} images from training set')
    
    # Load test data (check for 'test' folder, fallback to 'validation' for compatibility)
    test_dir = Path(data_dir) / 'test'
    if not test_dir.exists():
        test_dir = Path(data_dir) / 'validation'
    
    for emotion_idx, emotion in enumerate(EMOTION_CLASSES):
        emotion_dir = test_dir / emotion
        if emotion_dir.exists():
            images = load_images_from_folder(emotion_dir)
            test_images.append(images)
            test_labels.append(np.full(len(images), emotion_idx, dtype=np.int64))
            print(f'Loaded {len(images)} {emotion} images from test set')
    
    # Concatenate all classes
    train_data = np.concatenate(train_images, axis=0) if train_images else np.array([])
    train_labels = np.concatenate(train_labels, axis=0) if train_labels else np.array([], dtype=np.int64)
    test_data = np.concatenate(test_images, axis=0) if test_images else np.array([])
    test_labels = np.concatenate(test_labels, axis=0) if test_labels else np.array([], dtype=np.int64)
    
    return train_data, train_labels, test_data, test_labels


# --- 4. TRAINING ---

def train(model,
          train_images,
          train_labels,
          epochs: int = 10,
          batch_size: int = 64,
          lr: float = 1e-3,
          device: str = 'auto'):
    if device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Device: {device}")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dataset = EmotionDataset(train_images, train_labels, augment=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}', ncols=120)
        for batch_images, batch_labels in pbar:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pred = outputs.argmax(1)
            correct += (pred == batch_labels).sum().item()
            total += batch_labels.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{100*correct/total:.1f}%'})
        print(f'Epoch {epoch+1}: loss={running_loss/len(loader):.3f}, acc={100*correct/total:.1f}%')


# --- 5. EVALUATION ---
def evaluate(model, test_images, test_labels, device: str = 'auto'):
    """Evaluate model on test set and show per-class accuracy."""
    if device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    model.to(device)
    model.eval()
    
    dataset = EmotionDataset(test_images, test_labels, augment=False)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(device)
            outputs = model(batch_images)
            predictions = outputs.argmax(1).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(batch_labels.numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Overall accuracy
    accuracy = (all_predictions == all_labels).mean()
    print(f'\nTest accuracy: {accuracy*100:.2f}%')
    
    # Per-class accuracy
    print('\nPer-class accuracy:')
    for emotion_idx, emotion in enumerate(EMOTION_CLASSES):
        class_mask = all_labels == emotion_idx
        if class_mask.sum() > 0:
            class_acc = (all_predictions[class_mask] == all_labels[class_mask]).mean()
            print(f'  {emotion:10s}: {class_acc*100:.2f}% ({class_mask.sum()} samples)')
    
    return accuracy


# --- 6. MAIN ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train emotion classifier for 7 emotion classes')
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--output-model', type=str, default='emotion_classifier.pth')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    print('Loading data...')
    train_images, train_labels, test_images, test_labels = load_data(args.data_dir)
    print(f'\nTrain: {len(train_images)} images')
    print(f'Test: {len(test_images)} images\n')

    model = EmotionClassifier(num_classes=NUM_CLASSES)
    train(model, train_images, train_labels, epochs=args.epochs, device=args.device)
    evaluate(model, test_images, test_labels, device=args.device)
    torch.save(model.state_dict(), args.output_model)
    print(f'\nModel saved to {args.output_model}')