"""
Emotion classifier for 7 emotion classes using facial expressions.

Usage:
  python emotion_classifier.py --data-dir ./raf_data --epochs 10 --device auto
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
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import copy

# --- 1. DATASET and AUGMENTATION ---

# Define the 7 emotion classes
EMOTION_CLASSES = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASSES = len(EMOTION_CLASSES)

class EmotionDataset(Dataset):
    """Dataset wrapper for emotion classification with optional augmentation.
    Images are 75x75 grayscale (converted to RGB 3-channel format for the model)."""
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
                transforms.RandomResizedCrop(75, scale=(0.9, 1.1)),  # slight scale/crop
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.RandomErasing(p=0.25) 
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((75, 75)),
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
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)

        # Calculate the size after convolutions for the fully connected layer
        # Input: 100x100
        # conv1 (5x5, no padding): 100 - 5 + 1 = 96  -> pool -> 48
        # conv2 (5x5):             48  - 5 + 1 = 44  -> pool -> 22
        # conv3 (3x3):             22  - 3 + 1 = 20  -> pool -> 10
        # So we get 128 * 10 * 10 features.
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Conv -> ReLU -> Pool (3x), then Flatten -> FC -> Dropout -> FC
        x = self.pool(F.relu(self.batchnorm1(self.conv1(x))))  # -> [B, 32, 48, 48]
        x = self.pool(F.relu(self.batchnorm2(self.conv2(x))))  # -> [B, 64, 22, 22]
        x = self.pool(F.relu(self.conv3(x)))  # -> [B, 128, 10, 10]

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# --- 3. DATA LOADING ---

def load_data(data_dir: str = './raf_data'):
    """Load emotion images from train, validation, and test folders with 7 emotion classes."""
    def load_images_from_folder(folder_path: Path):
        """Load all images from a folder. Images are 75x75 grayscale but converted to RGB (3 channels)."""
        images = []
        # Load both .jpg and .png files
        image_files = sorted(list(folder_path.glob('*.jpg')) + list(folder_path.glob('*.png')))
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
    val_images = []
    val_labels = []
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
            print(f'Loaded {len(images)} {emotion} images from train set')
    
    # Load validation data (separate from test)
    val_dir = Path(data_dir) / 'validation'
    if val_dir.exists():
        for emotion_idx, emotion in enumerate(EMOTION_CLASSES):
            emotion_dir = val_dir / emotion
            if emotion_dir.exists():
                images = load_images_from_folder(emotion_dir)
                val_images.append(images)
                val_labels.append(np.full(len(images), emotion_idx, dtype=np.int64))
                print(f'Loaded {len(images)} {emotion} images from validation set')
    
    # Load test data
    test_dir = Path(data_dir) / 'test'
    if test_dir.exists():
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
    val_data = np.concatenate(val_images, axis=0) if val_images else np.array([])
    val_labels = np.concatenate(val_labels, axis=0) if val_labels else np.array([], dtype=np.int64)
    test_data = np.concatenate(test_images, axis=0) if test_images else np.array([])
    test_labels = np.concatenate(test_labels, axis=0) if test_labels else np.array([], dtype=np.int64)
    
    return train_data, train_labels, val_data, val_labels, test_data, test_labels


# --- 4. TRAINING ---

def evaluate_validation(model, val_images, val_labels, criterion, device, batch_size=64):
    """Evaluate model on validation set and return loss and accuracy."""
    model.eval()
    dataset = EmotionDataset(val_images, val_labels, augment=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_images, batch_labels in loader:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            val_loss += loss.item()
            pred = outputs.argmax(1)
            correct += (pred == batch_labels).sum().item()
            total += batch_labels.size(0)
    
    avg_loss = val_loss / len(loader)
    accuracy = 100 * correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def train(model,
          train_images,
          train_labels,
          val_images=None,
          val_labels=None,
          epochs: int = 10,
          batch_size: int = 64,
          lr: float = 1e-3,
          device: str = 'auto',
          early_stopping_patience: int = 2,
          val_split: float = 0.2,
          min_delta: float = 0.0001):
    """
    Train the model with validation-based early stopping.
    
    Args:
        model: The model to train
        train_images: Training images
        train_labels: Training labels
        val_images: Validation images (if None, will split from training data)
        val_labels: Validation labels (if None, will split from training data)
        epochs: Maximum number of epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device to use
        early_stopping_patience: Number of epochs to wait before early stopping
        val_split: Fraction of training data to use for validation if val_images is None
        min_delta: Minimum change in validation loss/accuracy to qualify as an improvement
    """
    if device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Device: {device}")
    model.to(device)

    # Check if we have validation data
    has_validation = val_images is not None and val_labels is not None and len(val_images) > 0 and len(val_labels) > 0
    
    # Only split if we don't have validation data and val_split > 0
    if not has_validation and val_split > 0:
        train_images, val_images, train_labels, val_labels = train_test_split(
            train_images, train_labels, test_size=val_split, random_state=42, stratify=train_labels
        )
        has_validation = True
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    train_dataset = EmotionDataset(train_images, train_labels, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Early stopping variables
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    if has_validation:
        print(f"\nTraining with early stopping (patience={early_stopping_patience})...")
    else:
        print(f"\nTraining without validation (no early stopping)...")
    print("=" * 80)

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', ncols=120)
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
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total if total > 0 else 0.0
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase (only if we have validation data)
        if has_validation:
            val_loss, val_acc = evaluate_validation(model, val_images, val_labels, criterion, device, batch_size)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # Print epoch summary
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
            print(f'  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
            
            # Check for improvement (with minimum delta threshold)
            improved = False
            # For loss: improvement means lower loss (by at least min_delta)
            if val_loss < (best_val_loss - min_delta):
                best_val_loss = val_loss
                improved = True
            # For accuracy: improvement means higher accuracy (by at least min_delta percentage points)
            if val_acc > (best_val_acc + min_delta):
                best_val_acc = val_acc
                improved = True
            
            # Save best model
            if improved:
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                print(f'  ✓ Best model updated (val_loss: {best_val_loss:.4f}, val_acc: {best_val_acc:.2f}%)')
            else:
                patience_counter += 1
                print(f'  No improvement ({patience_counter}/{early_stopping_patience})')
            
            # Check for overfitting
            if epoch > 0:
                train_val_gap = train_acc - val_acc
                if train_val_gap > 15:  # Large gap indicates overfitting
                    print(f'  ⚠ Warning: Large train-val gap ({train_val_gap:.2f}%) - possible overfitting!')
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f'\nEarly stopping triggered after {epoch+1} epochs (no improvement for {early_stopping_patience} epochs)')
                break
        else:
            # No validation data - just print training metrics
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        
        print("-" * 80)
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'\nBest model restored (val_loss: {best_val_loss:.4f}, val_acc: {best_val_acc:.2f}%)')
    
    # Print training summary
    print("\nTraining Summary:")
    print(f"  Final train loss: {train_losses[-1]:.4f}, acc: {train_accs[-1]:.2f}%")
    print(f"  Final val loss: {val_losses[-1]:.4f}, acc: {val_accs[-1]:.2f}%")
    print(f"  Best val loss: {best_val_loss:.4f}, acc: {best_val_acc:.2f}%")
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc
    }


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
    
    cm = confusion_matrix(all_labels, all_predictions, labels=list(range(NUM_CLASSES)))
    print('\nConfusion Matrix (rows = true, columns = predicted):\n')
    print(cm)
    
    return accuracy


# --- 6. MAIN ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train emotion classifier for 7 emotion classes')
    parser.add_argument('--data-dir', type=str, default='./raf_data')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--output-model', type=str, default='emotion_classifier.pth')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--early-stopping-patience', type=int, default=2,
                        help='Number of epochs to wait before early stopping (default: 2)')
    parser.add_argument('--min-delta', type=float, default=0.0001,
                        help='Minimum change in validation metric to qualify as improvement (default: 0.0001)')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Fraction of training data to use for validation (default: 0.2)')
    args = parser.parse_args()

    print('Loading data...')
    train_images, train_labels, val_images, val_labels, test_images, test_labels = load_data(args.data_dir)
    
    if len(train_images) == 0:
        raise ValueError(f'No training data found in {args.data_dir}. Please check the data directory path.')
    
    print(f'\nData Summary:')
    print(f'  Train:      {len(train_images)} images')
    print(f'  Validation: {len(val_images)} images')
    print(f'  Test:       {len(test_images)} images\n')

    model = EmotionClassifier(num_classes=NUM_CLASSES)
    train_history = train(
        model, 
        train_images, 
        train_labels,
        val_images=val_images,
        val_labels=val_labels,
        epochs=args.epochs, 
        device=args.device,
        early_stopping_patience=args.early_stopping_patience,
        val_split=args.val_split,
        min_delta=args.min_delta
    )
    
    print('\n' + '=' * 80)
    print('Evaluating on test set...')
    print('=' * 80)
    evaluate(model, test_images, test_labels, device=args.device)
    torch.save(model.state_dict(), args.output_model)
    print(f'\nModel saved to {args.output_model}')