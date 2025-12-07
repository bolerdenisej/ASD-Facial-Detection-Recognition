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
    Images are expected to be 48x48 grayscale pixels."""
    def __init__(self, images, labels, augment=False):
        self.images = images
        self.labels = labels
        
        # If augment is True, create a transform pipeline using transforms.Compose.
        # If augment is False, the transform should only convert the image to a tensor.
        # Note: Images are already 48x48 grayscale, so no resizing or grayscale conversion is needed.
        if augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.transform(self.images[idx])
        return image, self.labels[idx]

# --- 2. MODEL ARCHITECTURE ---

class EmotionClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        # Define the layers of your CNN: 3 conv layers, 3 pooling layers, 2 fully connected layers, 1 dropout layer.
        # Input images are 48x48 pixels.
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)

        # Calculate the size after convolutions for the fully connected layer
        # Input: 48x48
        # After conv1 (kernel 5): (48-5+1) = 44, after pool: 44/2 = 22
        # After conv2 (kernel 5): (22-5+1) = 18, after pool: 18/2 = 9
        # After conv3 (kernel 3): (9-3+1) = 7, after pool: 7/2 = 3
        # So we get 128 * 3 * 3 = 1152
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Implement the forward pass: Conv -> ReLU -> Pool, repeated 3 times,
        # then Flatten, then the dense layers.
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# --- 3. DATA LOADING ---

def load_data(data_dir='./data'):
    """Load emotion images from train and test folders with 7 emotion classes."""
    def load_images_from_folder(folder_path):
        """Load all images from a folder. Images are expected to be grayscale."""
        images = []
        image_files = sorted(Path(folder_path).glob('*.jpg'))
        for img_path in image_files:
            try:
                img = Image.open(img_path)
                # Ensure grayscale format (L mode) - images should already be grayscale
                if img.mode != 'L':
                    img = img.convert('L')
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


def train(model, train_images, train_labels, epochs=10, batch_size=64, lr=1e-3, device: str = 'auto'):
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