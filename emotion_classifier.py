"""
Minimal face/non-face classifier for 65×60 (H×W) grayscale patches.

Usage:
  python train_classifier.py --data-dir ./processed_data --epochs 10 --device auto
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

class FaceDataset(Dataset):
    """Tiny dataset wrapper with optional light augmentation."""
    def __init__(self, images, labels, augment=False):
        self.images = images
        self.labels = labels
        
        # --- YOUR CODE HERE ---
        # If augment is True, create a transform pipeline using transforms.Compose.
        # If augment is False, the transform should only convert the image to a tensor.
        if augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.transform(self.images[idx])
        return image, self.labels[idx]

# --- 2. MODEL ARCHITECTURE ---

class FaceClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # --- YOUR CODE HERE ---
        # Define the layers of your CNN here as specified in the README.
        # 3 conv layers, 3 pooling layers, 2 fully connected layers, 1 dropout layer.
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 5 * 5, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 2)
        # --- END YOUR CODE ---

    def forward(self, x):
        # --- YOUR CODE HERE ---
        # Implement the forward pass of your network.
        # The sequence should be Conv -> ReLU -> Pool, repeated 3 times,
        # then Flatten, then the dense layers.
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

        pass
        # --- END YOUR CODE ---


# --- THE REST OF THIS FILE IS PROVIDED COMPLETE ---


def load_data(data_dir='./data'):
    """Load 65×60 (H×W) grayscale BMP patches for train/test."""
    def load_images(path):
        return np.array([np.array(Image.open(f)) for f in sorted(Path(path).glob('*.bmp'))])
    
    train_faces = load_images(f'{data_dir}/training_data/faces')
    train_nonfaces = load_images(f'{data_dir}/training_data/nonfaces')
    test_faces = load_images(f'{data_dir}/test_data/faces')
    test_nonfaces = load_images(f'{data_dir}/test_data/nonfaces')
    return train_faces, train_nonfaces, test_faces, test_nonfaces


def train(model, train_faces, train_nonfaces, epochs=10, batch_size=64, lr=1e-3, device: str = 'auto'):
    if device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Device: {device}")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    images = np.concatenate([train_faces, train_nonfaces])
    labels = np.concatenate([
        np.ones(len(train_faces), dtype=np.int64),
        np.zeros(len(train_nonfaces), dtype=np.int64)
    ])

    dataset = FaceDataset(images, labels, augment=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}', ncols=120)
        for batch_images, batch_labels in pbar:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            correct += (outputs.argmax(1) == batch_labels).sum().item()
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})
        print(f'Epoch {epoch+1}: loss={running_loss/len(loader):.3f}, acc={100*correct/len(dataset):.1f}%')


def evaluate(model, faces, nonfaces, device: str = 'auto'):
    """Simple accuracy on test patches."""
    data = np.concatenate([faces, nonfaces])
    labels = np.concatenate([
        np.ones(len(faces), dtype=np.int64),
        np.zeros(len(nonfaces), dtype=np.int64)
    ])
    
    if device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        x = torch.FloatTensor(data).unsqueeze(1).to(device) / 255.0
        outputs = model(x)
        predictions = outputs.argmax(1).cpu().numpy()
    
    accuracy = (predictions == labels).mean()
    print(f'\nTest accuracy: {accuracy*100:.2f}%')
    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train face/non-face classifier (65x60)')
    parser.add_argument('--data-dir', type=str, default='./data/training_data')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--output-model', type=str, default='face_classifier.pth')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    print('Loading data...')
    train_faces, train_nonfaces, test_faces, test_nonfaces = load_data(args.data_dir)
    print(f'Train: {len(train_faces)} faces + {len(train_nonfaces)} non-faces')
    print(f'Test:  {len(test_faces)} faces + {len(test_nonfaces)} non-faces\n')

    model = FaceClassifier()
    train(model, train_faces, train_nonfaces, epochs=args.epochs, device=args.device)
    evaluate(model, test_faces, test_nonfaces, device=args.device)
    torch.save(model.state_dict(), args.output_model)
    print(f'\nModel saved to {args.output_model}')