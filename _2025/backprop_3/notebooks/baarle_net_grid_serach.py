import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
import cv2
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import os
import pandas as pd
from datetime import datetime
import itertools
import json
import warnings

# Suppress NumPy compatibility warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*NumPy.*')

# Set random seed for reproducibility
random_seed = 25
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def denorm(p, size=960):
    result = np.zeros_like(p)
    result[..., 0] = (p[..., 0] + 1) * (size / 2)
    result[..., 1] = size - ((p[..., 1] + 1) * (size / 2))
    return result

def viz_map_with_predictions(ax, map, X, y, model):
    probe = np.zeros((100, 100, 2))
    for j, xx in enumerate(np.linspace(-1, 1, 100)):
        for k, yy in enumerate(np.linspace(-1, 1, 100)):
            probe[j, k] = [yy, xx]
    probe = probe.reshape(100**2, -1)
    
    # Move model to CPU for visualization
    model_cpu = model.cpu()
    probe_logits = model_cpu(torch.tensor(probe).float())
    probe_logits = probe_logits.detach().numpy().reshape(100, 100, 2)
    probe_softmax = torch.nn.Softmax(dim=1)(torch.tensor(probe_logits.reshape(-1, 2)))
    
    ax.imshow(map.mean(2), cmap='gray')
    ax.imshow(np.flipud(probe_softmax[:, 1].reshape(100, 100)), 
               extent=[0, 960, 960, 0],
               alpha=0.7,
               cmap='viridis')
    
    X_denorm = denorm(X[:1000, :])
    labels = y[:1000]
    y_hat = torch.argmax(model_cpu(torch.tensor(X[:1000, :]).float()).detach(), 1).numpy()
    
    tp_mask = np.logical_and(labels == 1, y_hat == 1)
    fp_mask = np.logical_and(labels == 0, y_hat == 1)
    fn_mask = np.logical_and(labels == 1, y_hat == 0)
    tn_mask = np.logical_and(labels == 0, y_hat == 0)
    
    ax.scatter(X_denorm[tp_mask, 0], X_denorm[tp_mask, 1], c='g', s=3, alpha=0.8, label='TP')
    ax.scatter(X_denorm[fp_mask, 0], X_denorm[fp_mask, 1], c='r', marker='x', s=5, alpha=0.8, label='FP')
    ax.scatter(X_denorm[fn_mask, 0], X_denorm[fn_mask, 1], c='orange', marker='x', s=5, alpha=0.8, label='FN')
    ax.scatter(X_denorm[tn_mask, 0], X_denorm[tn_mask, 1], c='purple', marker='o', s=5, alpha=0.8, label='TN')
    
    # Move model back to original device
    model.to(device)

class BaarleNet(nn.Module):
    def __init__(self, hidden_layers=[64]):
        super(BaarleNet, self).__init__()
        layers = [nn.Linear(2, hidden_layers[0]), nn.ReLU()]
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_layers[-1], 2))
        self.layers = layers
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def generate_network_configs(max_depth=8, max_width=512):
    """Generate network configurations from shallow to deep, narrow to wide"""
    configs = []
    
    # Single layer networks
    for width in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
        configs.append([width])
    
    # Multi-layer networks
    for depth in range(2, max_depth + 1):
        # Start with uniform width networks
        for width in [2, 4, 8, 16, 32, 64, 128, 256]:
            if width ** depth <= 512 ** 2:  # Limit total parameters
                configs.append([width] * depth)
        
        # Add some pyramidal architectures (narrowing)
        if depth <= 4:
            for start_width in [256, 128, 64]:
                config = []
                width = start_width
                for _ in range(depth):
                    config.append(width)
                    width = max(2, width // 2)
                configs.append(config)
        
        # Add some expanding architectures
        if depth <= 4:
            for start_width in [8, 16, 32]:
                config = []
                width = start_width
                for _ in range(depth):
                    config.append(width)
                    width = min(512, width * 2)
                configs.append(config[:depth])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_configs = []
    for config in configs:
        config_tuple = tuple(config)
        if config_tuple not in seen:
            seen.add(config_tuple)
            unique_configs.append(config)
    
    return unique_configs

def train_model(X_tensor, y_tensor, hidden_layers, lr, num_epochs=20000, print_freq=None):
    """Train a single model configuration"""
    torch.manual_seed(random_seed)
    
    model = BaarleNet(hidden_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    X_tensor_gpu = X_tensor.to(device)
    y_tensor_gpu = y_tensor.to(device)
    
    losses = []
    accuracies = []
    
    for epoch in range(num_epochs):
        outputs = model(X_tensor_gpu)
        loss = criterion(outputs, y_tensor_gpu)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if print_freq and (epoch + 1) % print_freq == 0:
            with torch.no_grad():
                outputs_batch = model(X_tensor_gpu)
                accuracy = (torch.argmax(outputs_batch, dim=1) == y_tensor_gpu).sum().item() / len(y_tensor)
                losses.append(loss.item())
                accuracies.append(accuracy)
    
    # Final evaluation
    with torch.no_grad():
        outputs_final = model(X_tensor_gpu)
        final_accuracy = (torch.argmax(outputs_final, dim=1) == y_tensor_gpu).sum().item() / len(y_tensor)
        final_loss = criterion(outputs_final, y_tensor_gpu).item()
    
    return model, final_loss, final_accuracy, losses, accuracies

def main():
    # Create output directories
    os.makedirs('grid_search_results', exist_ok=True)
    os.makedirs('grid_search_results/visualizations', exist_ok=True)
    
    # Load and preprocess data
    print("Loading data...")
    map_img = cv2.imread('Baarle-Nassau_-_Baarle-Hertog-en.svg.png')[:, :, (2, 1, 0)]
    
    belgium_color = np.array([251, 234, 81])
    netherlands_color = np.array([255, 255, 228])
    
    netherlands_region = ((map_img - netherlands_color)**2).sum(-1) < 10
    belgium_region = ((map_img - belgium_color)**2).sum(-1) < 10000
    
    num_points_to_sample = 10000
    
    b_coords = np.array(np.where(belgium_region)).T.astype('float')
    n_coords = np.array(np.where(netherlands_region)).T.astype('float')
    
    belgium_coords_all = np.zeros_like(b_coords)
    netherlands_coords_all = np.zeros_like(n_coords)
    
    belgium_coords_all[:, 0] = b_coords[:, 1] / (960/2) - 1
    belgium_coords_all[:, 1] = (960 - b_coords[:, 0]) / (960/2) - 1
    netherlands_coords_all[:, 0] = n_coords[:, 1] / (960/2) - 1
    netherlands_coords_all[:, 1] = (960 - n_coords[:, 0]) / (960/2) - 1
    
    belgium_coords = belgium_coords_all[np.random.choice(len(belgium_coords_all), num_points_to_sample), :]
    netherlands_coords = netherlands_coords_all[np.random.choice(len(netherlands_coords_all), num_points_to_sample), :]
    
    X = np.vstack((netherlands_coords, belgium_coords))
    y = np.concatenate((np.zeros(len(netherlands_coords)), np.ones(len(belgium_coords)))).astype(np.int32)
    
    rI = np.arange(len(y))
    np.random.shuffle(rI)
    X = X[rI, :]
    y = y[rI]
    
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    # Generate configurations
    network_configs = generate_network_configs(max_depth=8, max_width=512)
    learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
    
    print(f"Testing {len(network_configs)} network configurations with {len(learning_rates)} learning rates")
    print(f"Total experiments: {len(network_configs) * len(learning_rates)}")
    
    # Initialize results DataFrame
    results_file = 'grid_search_results/results.csv'
    if os.path.exists(results_file):
        results_df = pd.read_csv(results_file)
    else:
        results_df = pd.DataFrame(columns=['timestamp', 'config', 'config_str', 'num_layers', 
                                         'total_params', 'learning_rate', 'final_loss', 
                                         'final_accuracy', 'training_time'])
    
    # Grid search
    experiment_count = 0
    total_experiments = len(network_configs) * len(learning_rates)
    
    for config in tqdm(network_configs, desc="Network configs"):
        for lr in learning_rates:
            experiment_count += 1
            print(f"\nExperiment {experiment_count}/{total_experiments}")
            print(f"Config: {config}, LR: {lr}")
            
            # Calculate number of parameters
            num_params = 2 * config[0] + config[0]  # Input layer
            for i in range(len(config) - 1):
                num_params += config[i] * config[i+1] + config[i+1]
            num_params += config[-1] * 2 + 2  # Output layer
            
            # Train model
            start_time = datetime.now()
            model, final_loss, final_accuracy, _, _ = train_model(
                X_tensor, y_tensor, config, lr, num_epochs=20000, print_freq=1000
            )
            training_time = (datetime.now() - start_time).total_seconds()
            
            print(f"Final accuracy: {final_accuracy:.4f}, Final loss: {final_loss:.4f}")
            
            # Save visualization
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
            viz_map_with_predictions(ax, map_img, X, y, model)
            
            config_str = '_'.join(map(str, config))
            viz_filename = f'grid_search_results/visualizations/config_{config_str}_lr_{lr:.4f}_acc_{final_accuracy:.4f}.png'
            ax.set_title(f'Config: {config}, LR: {lr}, Acc: {final_accuracy:.4f}')
            ax.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(viz_filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Update results
            new_result = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'config': str(config),
                'config_str': config_str,
                'num_layers': len(config),
                'total_params': num_params,
                'learning_rate': lr,
                'final_loss': final_loss,
                'final_accuracy': final_accuracy,
                'training_time': training_time
            }
            
            results_df = pd.concat([results_df, pd.DataFrame([new_result])], ignore_index=True)
            
            # Save results after each experiment
            results_df.to_csv(results_file, index=False)
            
            # Also save as JSON for easier loading
            results_df.to_json('grid_search_results/results.json', orient='records', indent=2)
    
    # Final summary
    print("\n" + "="*50)
    print("GRID SEARCH COMPLETE")
    print("="*50)
    
    # Find best configuration
    best_idx = results_df['final_accuracy'].idxmax()
    best_result = results_df.loc[best_idx]
    
    print(f"\nBest configuration:")
    print(f"  Architecture: {best_result['config']}")
    print(f"  Learning rate: {best_result['learning_rate']}")
    print(f"  Accuracy: {best_result['final_accuracy']:.4f}")
    print(f"  Loss: {best_result['final_loss']:.4f}")
    print(f"  Parameters: {best_result['total_params']}")
    
    # Save summary statistics
    summary_stats = results_df.groupby('num_layers')['final_accuracy'].agg(['mean', 'std', 'max'])
    summary_stats.to_csv('grid_search_results/summary_by_depth.csv')
    
    print(f"\nResults saved to: {results_file}")
    print(f"Visualizations saved to: grid_search_results/visualizations/")

if __name__ == "__main__":
    main()