import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import json
from pathlib import Path
from tqdm import tqdm
import argparse

# Import your ResNet18 model here
# from your_model_file import make_resnet18k, PreActBlock, PreActResNet


class NoisyDataset(Dataset):
    """Wrapper to add label noise to a dataset."""
    def __init__(self, dataset, noise_rate=0.15, num_classes=10, seed=42):
        self.dataset = dataset
        self.noise_rate = noise_rate
        self.num_classes = num_classes
        
        # Generate noisy labels deterministically
        np.random.seed(seed)
        self.noisy_labels = []
        
        for idx in range(len(dataset)):
            original_label = dataset[idx][1]
            if np.random.random() < noise_rate:
                # Choose a random different label
                noisy_label = np.random.randint(0, num_classes)
                while noisy_label == original_label:
                    noisy_label = np.random.randint(0, num_classes)
                self.noisy_labels.append(noisy_label)
            else:
                self.noisy_labels.append(original_label)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        label = self.noisy_labels[idx]
        return img, label


def get_cifar10_loaders(batch_size=128, noise_rate=0.15, num_workers=4):
    """Create CIFAR-10 data loaders with label noise and augmentation."""
    
    # Data augmentation for training (applied before noise)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # No augmentation for test set
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load datasets
    trainset_base = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    
    # Add label noise to training set
    trainset = NoisyDataset(trainset_base, noise_rate=noise_rate, num_classes=10)
    
    # Create data loaders
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True)
    
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    
    return trainloader, testloader


def train_epoch(model, trainloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return train_loss / len(trainloader), 100. * correct / total


def test(model, testloader, criterion, device):
    """Evaluate on test set."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return test_loss / len(testloader), 100. * correct / total


def save_checkpoint(model, optimizer, epoch, train_acc, test_acc, save_dir, k):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_acc': train_acc,
        'test_acc': test_acc,
        'k': k
    }
    save_path = os.path.join(save_dir, f'checkpoint_k{k}_epoch{epoch}.pt')
    torch.save(checkpoint, save_path)
    return save_path


def train(k, num_epochs=4000, batch_size=128, lr=1e-4, noise_rate=0.15, 
          save_dir='./checkpoints', log_interval=100, save_interval=500,
          device='cuda'):
    """
    Train a ResNet18 with width parameter k.
    
    Args:
        k: Width multiplier for ResNet18
        num_epochs: Number of training epochs (default: 4000)
        batch_size: Batch size (default: 128)
        lr: Learning rate (default: 1e-4)
        noise_rate: Label noise rate (default: 0.15)
        save_dir: Directory to save checkpoints and logs
        log_interval: Log metrics every N epochs
        save_interval: Save checkpoint every N epochs
        device: Device to train on
    """
    
    # Create save directory
    save_dir = Path(save_dir) / f'resnet18_k{k}'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create data loaders
    print('Loading CIFAR-10 dataset...')
    trainloader, testloader = get_cifar10_loaders(
        batch_size=batch_size, noise_rate=noise_rate)
    
    # Create model
    print(f'Creating ResNet18 with k={k}...')
    model = make_resnet18k(k=k, num_classes=10).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {num_params:,}')
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'epochs': [],
        'k': k,
        'num_params': num_params,
        'noise_rate': noise_rate,
        'lr': lr,
        'batch_size': batch_size
    }
    
    # Training loop
    print(f'Starting training for {num_epochs} epochs...')
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
        test_loss, test_acc = test(model, testloader, criterion, device)
        
        # Log metrics
        if epoch % log_interval == 0 or epoch == 1:
            print(f'Epoch {epoch}/{num_epochs} | '
                  f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
                  f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')
            
            history['epochs'].append(epoch)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            
            # Save history
            with open(save_dir / 'history.json', 'w') as f:
                json.dump(history, f, indent=2)
        
        # Save checkpoint
        if epoch % save_interval == 0 or epoch == num_epochs:
            checkpoint_path = save_checkpoint(
                model, optimizer, epoch, train_acc, test_acc, save_dir, k)
            print(f'Saved checkpoint: {checkpoint_path}')
    
    print('Training completed!')
    return history


def main():
    parser = argparse.ArgumentParser(description='Train ResNet18 for Deep Double Descent')
    parser.add_argument('--k', type=int, default=64, help='Width parameter for ResNet18')
    parser.add_argument('--epochs', type=int, default=4000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--noise_rate', type=float, default=0.15, help='Label noise rate')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Save directory')
    parser.add_argument('--log_interval', type=int, default=100, help='Log every N epochs')
    parser.add_argument('--save_interval', type=int, default=500, help='Save every N epochs')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Train model
    history = train(
        k=args.k,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        noise_rate=args.noise_rate,
        save_dir=args.save_dir,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        device=args.device
    )
    
    print(f'\nFinal Results:')
    print(f'Train Accuracy: {history["train_acc"][-1]:.2f}%')
    print(f'Test Accuracy: {history["test_acc"][-1]:.2f}%')


if __name__ == '__main__':
    main()