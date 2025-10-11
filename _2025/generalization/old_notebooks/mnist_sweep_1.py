import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
class Config:
    # Data settings
    num_train_samples = 4000  # Fixed training set size
    batch_size = 64
    
    # Model sweep settings - extended range for better double descent
    # hidden_units_list = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 200]
    hidden_units_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 400]
    num_epochs = 3000
    
    # Training settings
    learning_rate = 0.001
    optimizer_name = 'Adam'
    
    # Parallel processing
    num_workers = 12 ##min(4, mp.cpu_count() - 1)  # Adjust based on your system
    
    # Output settings
    results_file = f'double_descent_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    checkpoint_freq = 50  # Save intermediate results every N epochs

# Model definition
class SimpleNN(nn.Module):
    def __init__(self, num_hidden_units=128):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, num_hidden_units)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(num_hidden_units, 10)
        
        # Calculate number of parameters
        self.num_params = (28 * 28 + 1) * num_hidden_units + (num_hidden_units + 1) * 10
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Dataset class
class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def prepare_data(config):
    """Prepare MNIST dataset with specified number of training samples"""
    print(f"Loading MNIST dataset with {config.num_train_samples} training samples...")
    
    # Load dataset
    dataset = load_dataset('mnist')
    
    # Define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Create train and test datasets
    train_subset = dataset['train'].select(range(config.num_train_samples))
    train_dataset = MNISTDataset(train_subset, transform=transform)
    test_dataset = MNISTDataset(dataset['test'], transform=transform)
    
    return train_dataset, test_dataset

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def test(model, loader, criterion, device):
    """Evaluate model on test set"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def save_results_safe(results_df, filepath):
    """Save results to CSV, thread-safe"""
    # Use file locking or temporary file for thread safety
    temp_file = filepath + f'.tmp_{os.getpid()}'
    results_df.to_csv(temp_file, index=False)
    
    # Atomic rename
    try:
        os.replace(temp_file, filepath)
    except:
        # If replace fails, try alternative approach
        if os.path.exists(filepath):
            existing_df = pd.read_csv(filepath)
            combined_df = pd.concat([existing_df, results_df], ignore_index=True)
            combined_df.to_csv(filepath, index=False)
        else:
            results_df.to_csv(filepath, index=False)
        
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)

def train_model_wrapper(args):
    """Wrapper function for training a single model (picklable for multiprocessing)"""
    hidden_units, device_id, config_dict, train_data, test_data = args
    
    # Reconstruct config from dictionary
    config = Config()
    for key, value in config_dict.items():
        setattr(config, key, value)
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{device_id % torch.cuda.device_count()}')
    else:
        device = torch.device('cpu')
    
    print(f"Starting training for width={hidden_units} on {device}")
    
    # Recreate transform for dataloaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Create datasets
    train_dataset = MNISTDataset(train_data, transform=transform)
    test_dataset = MNISTDataset(test_data, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                            shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1000, 
                           shuffle=False, num_workers=0)
    
    # Initialize model
    model = SimpleNN(hidden_units).to(device)
    num_params = model.num_params
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Training loop with periodic saving
    results = []
    
    for epoch in range(1, config.num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)
        
        # Store results
        result = {
            'Hidden_Units': hidden_units,
            'Parameters': num_params,
            'Epoch': epoch,
            'Learning_Rate': config.learning_rate,
            'Batch_Size': config.batch_size,
            'Optimizer': config.optimizer_name,
            'Train_Samples': config.num_train_samples,
            'Train_Loss': train_loss,
            'Train_Accuracy': train_acc,
            'Test_Loss': test_loss,
            'Test_Accuracy': test_acc
        }
        results.append(result)
        
        # Periodic checkpoint saving
        if epoch % config.checkpoint_freq == 0 or epoch == config.num_epochs:
            print(f"Width={hidden_units}, Epoch={epoch}/{config.num_epochs}: "
                  f"Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")
            
            # Save intermediate results
            df_new = pd.DataFrame(results)
            
            # Append to existing file
            if os.path.exists(config.results_file):
                df_existing = pd.read_csv(config.results_file)
                # Remove any existing data for this width/epoch combination to avoid duplicates
                mask = ~((df_existing['Hidden_Units'] == hidden_units) & 
                        (df_existing['Epoch'].isin(df_new['Epoch'].values)))
                df_existing = df_existing[mask]
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df_combined = df_new
            
            save_results_safe(df_combined, config.results_file)
            results = []  # Clear buffer after saving
    
    print(f"Completed training for width={hidden_units}")
    return hidden_units

def run_parallel_experiments(config):
    """Run experiments in parallel across different model widths"""
    # Prepare data once
    print("Preparing dataset...")
    dataset = load_dataset('mnist')
    train_data = dataset['train'].select(range(config.num_train_samples))
    test_data = dataset['test']
    
    # Convert config to dictionary for pickling
    config_dict = {
        'num_train_samples': config.num_train_samples,
        'batch_size': config.batch_size,
        'num_epochs': config.num_epochs,
        'learning_rate': config.learning_rate,
        'optimizer_name': config.optimizer_name,
        'results_file': config.results_file,
        'checkpoint_freq': config.checkpoint_freq
    }
    
    # Create results file with headers
    initial_df = pd.DataFrame(columns=[
        'Hidden_Units', 'Parameters', 'Epoch', 'Learning_Rate', 
        'Batch_Size', 'Optimizer', 'Train_Samples',
        'Train_Loss', 'Train_Accuracy', 'Test_Loss', 'Test_Accuracy'
    ])
    save_results_safe(initial_df, config.results_file)
    
    print(f"\nStarting parallel training with {config.num_workers} workers")
    print(f"Training {len(config.hidden_units_list)} models with widths: {config.hidden_units_list}")
    print(f"Results will be saved to: {config.results_file}\n")
    
    # Prepare arguments for each task
    tasks = [
        (width, i % config.num_workers, config_dict, train_data, test_data)
        for i, width in enumerate(config.hidden_units_list)
    ]
    
    # Run parallel training
    with mp.Pool(processes=config.num_workers) as pool:
        results = pool.map(train_model_wrapper, tasks)
    
    print(f"\nAll experiments completed! Results saved to {config.results_file}")
    return config.results_file

def run_sequential_experiments(config):
    """Alternative: Run experiments sequentially (useful for debugging)"""
    # Prepare data once
    print("Preparing dataset...")
    dataset = load_dataset('mnist')
    train_data = dataset['train'].select(range(config.num_train_samples))
    test_data = dataset['test']
    
    # Convert config to dictionary
    config_dict = {
        'num_train_samples': config.num_train_samples,
        'batch_size': config.batch_size,
        'num_epochs': config.num_epochs,
        'learning_rate': config.learning_rate,
        'optimizer_name': config.optimizer_name,
        'results_file': config.results_file,
        'checkpoint_freq': config.checkpoint_freq
    }
    
    # Create results file
    initial_df = pd.DataFrame(columns=[
        'Hidden_Units', 'Parameters', 'Epoch', 'Learning_Rate', 
        'Batch_Size', 'Optimizer', 'Train_Samples',
        'Train_Loss', 'Train_Accuracy', 'Test_Loss', 'Test_Accuracy'
    ])
    save_results_safe(initial_df, config.results_file)
    
    print(f"\nStarting sequential training")
    print(f"Training {len(config.hidden_units_list)} models with widths: {config.hidden_units_list}")
    print(f"Results will be saved to: {config.results_file}\n")
    
    # Run training sequentially
    for width in config.hidden_units_list:
        args = (width, 0, config_dict, train_data, test_data)
        train_model_wrapper(args)
    
    print(f"\nAll experiments completed! Results saved to {config.results_file}")
    return config.results_file

def plot_double_descent(results_file):
    """Create visualization of double descent phenomenon"""
    df = pd.read_csv(results_file)
    
    # Remove any duplicate entries
    df = df.drop_duplicates(subset=['Hidden_Units', 'Epoch'])
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Select specific epochs to visualize
    epochs_to_plot = [50, 100, 250, 500, 1000]
    available_epochs = df['Epoch'].unique()
    epochs_to_plot = [e for e in epochs_to_plot if e in available_epochs]
    
    # Plot 1: Test Accuracy vs Model Width for different epochs
    ax = axes[0, 0]
    for epoch in epochs_to_plot:
        epoch_data = df[df['Epoch'] == epoch].sort_values('Hidden_Units')
        if len(epoch_data) > 0:
            ax.plot(epoch_data['Hidden_Units'], epoch_data['Test_Accuracy'], 
                    marker='o', label=f'Epoch {epoch}', markersize=4)
    ax.set_xlabel('Hidden Units (Model Width)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Test Accuracy vs Model Width')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # Plot 2: Test Loss vs Model Width
    ax = axes[0, 1]
    for epoch in epochs_to_plot:
        epoch_data = df[df['Epoch'] == epoch].sort_values('Hidden_Units')
        if len(epoch_data) > 0:
            ax.plot(epoch_data['Hidden_Units'], epoch_data['Test_Loss'], 
                    marker='o', label=f'Epoch {epoch}', markersize=4)
    ax.set_xlabel('Hidden Units (Model Width)')
    ax.set_ylabel('Test Loss')
    ax.set_title('Test Loss vs Model Width')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # Plot 3: Train vs Test Accuracy
    ax = axes[1, 0]
    last_epoch = df['Epoch'].max()
    last_epoch_data = df[df['Epoch'] == last_epoch].sort_values('Hidden_Units')
    if len(last_epoch_data) > 0:
        ax.plot(last_epoch_data['Hidden_Units'], last_epoch_data['Train_Accuracy'], 
                marker='o', label='Train Accuracy', markersize=4)
        ax.plot(last_epoch_data['Hidden_Units'], last_epoch_data['Test_Accuracy'], 
                marker='s', label='Test Accuracy', markersize=4)
        
        # Add parameter count on secondary x-axis
        ax2 = ax.twiny()
        ax2.plot(last_epoch_data['Parameters'], last_epoch_data['Test_Accuracy'], alpha=0)
        ax2.set_xlabel('Number of Parameters', color='gray')
        ax2.tick_params(axis='x', labelcolor='gray')
        
    ax.set_xlabel('Hidden Units (Model Width)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Train vs Test Accuracy at Epoch {last_epoch}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # Plot 4: Interpolation ratio (params/samples)
    ax = axes[1, 1]
    if len(last_epoch_data) > 0:
        interpolation_ratio = last_epoch_data['Parameters'] / last_epoch_data['Train_Samples']
        ax.plot(interpolation_ratio, last_epoch_data['Test_Accuracy'], 
                marker='o', color='red', markersize=4)
        ax.axvline(x=1.0, color='gray', linestyle='--', label='Interpolation Threshold')
        ax.set_xlabel('Parameters / Training Samples')
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title('Double Descent: Test Accuracy vs Interpolation Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('double_descent_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Plots saved to 'double_descent_results.png'")

if __name__ == "__main__":
    # Initialize configuration
    config = Config()
    
    # Choose between parallel and sequential execution
    use_parallel = True  # Set to False for debugging
    
    if use_parallel:
        try:
            results_file = run_parallel_experiments(config)
        except Exception as e:
            print(f"Parallel execution failed: {e}")
            print("Falling back to sequential execution...")
            results_file = run_sequential_experiments(config)
    else:
        results_file = run_sequential_experiments(config)
    
    # Generate plots
    print("\nGenerating visualization...")
    plot_double_descent(results_file)
    
    # Print summary statistics
    df = pd.read_csv(results_file)
    df = df.drop_duplicates(subset=['Hidden_Units', 'Epoch'])
    
    print("\n=== Experiment Summary ===")
    print(f"Total experiments run: {len(df['Hidden_Units'].unique())} models × {df['Epoch'].max()} epochs")
    print(f"Model widths tested: {sorted(df['Hidden_Units'].unique())}")
    print(f"Parameter counts: {sorted(df['Parameters'].unique())}")
    
    # Find best model
    last_epoch = df['Epoch'].max()
    last_epoch_df = df[df['Epoch'] == last_epoch]
    if len(last_epoch_df) > 0:
        best = last_epoch_df.nlargest(1, 'Test_Accuracy').iloc[0]
        print(f"\nBest model at epoch {last_epoch}:")
        print(f"  Width: {best['Hidden_Units']} units ({int(best['Parameters'])} parameters)")
        print(f"  Test Accuracy: {best['Test_Accuracy']:.2f}%")
        print(f"  Train Accuracy: {best['Train_Accuracy']:.2f}%")
        print(f"  Interpolation ratio: {best['Parameters']/best['Train_Samples']:.2f}")