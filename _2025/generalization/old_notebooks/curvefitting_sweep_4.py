import numpy as np 
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime
import json

# ============================================================================
# Configuration
# ============================================================================
class ExperimentConfig:
    # Data parameters
    data_random_seed = 5
    n_points = 32
    noise_level = 0.2
    train_fraction = 0.5
    data_dimension = 2  # Input dimension (1, 2, 3, etc.)
    
    # Training parameters
    num_epochs = 100000
    checkpoints = 5
    
    # Sweep parameters
    hidden_units = [4, 5, 6, 7, 8, 10, 12, 14, 15, 16, 18, 20, 22, 24, 32, 
                    64, 128, 256, 512, 1024, 2048, 4096]
    random_seeds = [5, 42, 123, 456, 789]  # Multiple seeds per config
    
    # Learning rate strategy
    base_lr = 1e-2
    large_model_lrs = [1e-2, 1e-3]  # For models > 1000 units
    large_model_threshold = 1000
    
    # Output directories
    output_dir = Path("double_descent_results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ============================================================================
# Data Generation
# ============================================================================
def f(x):
    """True function to fit - generalizes to arbitrary dimensions
    f(x) = 0.5 * sum(x_i^2) for i in dimensions
    """
    if x.ndim == 1:
        # Single point, potentially multi-dimensional
        return 0.5 * np.sum(x**2)
    else:
        # Multiple points (rows), sum over columns (dimensions)
        return 0.5 * np.sum(x**2, axis=1)

def generate_data(config):
    """Generate train/test data splits"""
    d = config.data_dimension
    
    # For visualization, create grid over input space
    if d == 1:
        all_x = np.linspace(-2, 2, 128).reshape(-1, 1)
    elif d == 2:
        # Create 2D grid for visualization
        grid_1d = np.linspace(-2, 2, 32)
        xx, yy = np.meshgrid(grid_1d, grid_1d)
        all_x = np.c_[xx.ravel(), yy.ravel()]
    else:
        # For higher dimensions, sample points for visualization
        all_x = np.random.uniform(-2, 2, (128, d))
    
    all_y = f(all_x)
    
    n_train = int(np.floor(config.n_points * config.train_fraction))
    n_test = config.n_points - n_train
    
    np.random.seed(config.data_random_seed) 
    x = np.random.uniform(-2, 2, (config.n_points, d))
    y = f(x) + config.noise_level * np.random.randn(config.n_points)
    
    x_train = x[:n_train]
    y_train = y[:n_train]
    x_test = x[n_train:]
    y_test = y[n_train:]
    
    # Convert to tensors
    x_train_tensor = torch.FloatTensor(x_train)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    x_test_tensor = torch.FloatTensor(x_test)
    y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)
    
    return {
        'x_train': x_train, 'y_train': y_train,
        'x_test': x_test, 'y_test': y_test,
        'x_train_tensor': x_train_tensor, 'y_train_tensor': y_train_tensor,
        'x_test_tensor': x_test_tensor, 'y_test_tensor': y_test_tensor,
        'all_x': all_x, 'all_y': all_y,
        'n_train': n_train, 'n_test': n_test,
        'dimension': d
    }

# ============================================================================
# Model Definition
# ============================================================================
class TwoLayerNet(nn.Module):
    def __init__(self, input_dim=1, hidden_size=20):
        super(TwoLayerNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# ============================================================================
# Training Function
# ============================================================================
def train_model(data, hidden_units=4, num_epochs=100000, lr=1e-2, 
                seed=5, checkpoints=5):
    """Train a single model and return training history"""
    torch.manual_seed(seed)
    
    train_losses = []
    test_losses = []
    epochs_recorded = []
    
    input_dim = data['dimension']
    model = TwoLayerNet(input_dim=input_dim, hidden_size=hidden_units)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    checkpoint_interval = num_epochs // checkpoints
    
    for epoch in range(num_epochs):
        # Training step
        outputs = model(data['x_train_tensor'])
        train_loss = criterion(outputs, data['y_train_tensor'])
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
    
        # Record losses at checkpoints
        if (epoch + 1) % checkpoint_interval == 0 or epoch == num_epochs - 1:
            with torch.no_grad():
                test_outputs = model(data['x_test_tensor'])
                test_loss = criterion(test_outputs, data['y_test_tensor'])
            
            train_losses.append(train_loss.item())
            test_losses.append(test_loss.item())
            epochs_recorded.append(epoch + 1)
    
    # Get final predictions for visualization
    with torch.no_grad():
        all_x_tensor = torch.FloatTensor(data['all_x'])
        predictions = model(all_x_tensor).numpy().flatten()
    
    return {
        'model': model,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'epochs': epochs_recorded,
        'final_train_loss': train_losses[-1],
        'final_test_loss': test_losses[-1],
        'predictions': predictions
    }

# ============================================================================
# Experiment Runner
# ============================================================================
def run_experiments(config):
    """Run full experiment sweep"""
    # Setup output directories
    config.output_dir.mkdir(exist_ok=True)
    exp_dir = config.output_dir / config.timestamp
    exp_dir.mkdir(exist_ok=True)
    (exp_dir / "curves").mkdir(exist_ok=True)
    (exp_dir / "models").mkdir(exist_ok=True)
    
    # Save configuration
    config_dict = {k: v for k, v in vars(config).items() 
                   if not k.startswith('_')}
    with open(exp_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    
    # Generate data (fixed across all experiments)
    print("Generating data...")
    data = generate_data(config)
    
    # Save data
    with open(exp_dir / "data.pkl", 'wb') as f:
        pickle.dump(data, f)
    
    # Run experiments
    results = []
    total_experiments = sum(
        len(config.random_seeds) * (len(config.large_model_lrs) 
        if h > config.large_model_threshold else 1)
        for h in config.hidden_units
    )
    
    print(f"\nRunning {total_experiments} experiments...")
    print(f"Results will be saved to: {exp_dir}\n")
    
    with tqdm(total=total_experiments, desc="Overall Progress") as pbar:
        for h in config.hidden_units:
            # Determine learning rates to try
            if h > config.large_model_threshold:
                lrs_to_try = config.large_model_lrs
            else:
                lrs_to_try = [config.base_lr]
            
            for lr in lrs_to_try:
                for seed in config.random_seeds:
                    # Train model
                    result = train_model(
                        data=data,
                        hidden_units=h,
                        num_epochs=config.num_epochs,
                        lr=lr,
                        seed=seed,
                        checkpoints=config.checkpoints
                    )
                    
                    # Store results
                    experiment_id = f"h{h}_lr{lr}_seed{seed}"
                    record = {
                        'experiment_id': experiment_id,
                        'hidden_units': h,
                        'learning_rate': lr,
                        'torch_random_seed': seed,
                        'train_loss': result['final_train_loss'],
                        'test_loss': result['final_test_loss'],
                        'noise_level': config.noise_level,
                        'data_dimension': config.data_dimension,
                        'n_train': data['n_train'],
                        'n_test': data['n_test'],
                        'n_params': h * config.data_dimension + h + h * 1 + 1  # 2-layer network params
                    }
                    results.append(record)
                    
                    # Save individual curve
                    curve_data = {
                        'config': record,
                        'train_losses': result['train_losses'],
                        'test_losses': result['test_losses'],
                        'epochs': result['epochs'],
                        'predictions': result['predictions'],
                        'x_values': data['all_x']
                    }
                    with open(exp_dir / "curves" / f"{experiment_id}.pkl", 'wb') as f:
                        pickle.dump(curve_data, f)
                    
                    # Optionally save model weights
                    torch.save(result['model'].state_dict(), 
                             exp_dir / "models" / f"{experiment_id}.pt")
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'h': h, 
                        'lr': lr,
                        'seed': seed,
                        'test_loss': f"{result['final_test_loss']:.4f}"
                    })
    
    # Create and save dataframe
    df = pd.DataFrame(results)
    df = df.sort_values(['hidden_units', 'learning_rate', 'torch_random_seed'])
    df.to_csv(exp_dir / "results.csv", index=False)
    df.to_pickle(exp_dir / "results.pkl")
    
    print(f"\n✓ Experiments complete!")
    print(f"✓ Results saved to: {exp_dir}")
    print(f"✓ Total experiments: {len(df)}")
    
    return df, data, exp_dir

# ============================================================================
# Visualization
# ============================================================================
def create_visualizations(df, data, exp_dir, config):
    """Create comprehensive visualization of results"""
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Data visualization (dimension-aware)
    ax1 = plt.subplot(3, 3, 1)
    d = data['dimension']
    
    if d == 1:
        ax1.plot(data['all_x'].flatten(), data['all_y'], 'k-', label='True function', linewidth=2)
        ax1.scatter(data['x_train'].flatten(), data['y_train'], c='cyan', s=50, 
                    label=f'Train (n={data["n_train"]})', edgecolors='k')
        ax1.scatter(data['x_test'].flatten(), data['y_test'], c='magenta', s=50, 
                    label=f'Test (n={data["n_test"]})', edgecolors='k')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
    elif d == 2:
        ax1.scatter(data['x_train'][:, 0], data['x_train'][:, 1], c=data['y_train'], 
                   s=100, cmap='viridis', edgecolors='cyan', linewidths=2,
                   label=f'Train (n={data["n_train"]})')
        ax1.scatter(data['x_test'][:, 0], data['x_test'][:, 1], c=data['y_test'], 
                   s=100, cmap='viridis', marker='s', edgecolors='magenta', linewidths=2,
                   label=f'Test (n={data["n_test"]})')
        ax1.set_xlabel('x₁')
        ax1.set_ylabel('x₂')
        plt.colorbar(ax1.collections[0], ax=ax1, label='y')
    else:
        # For higher dimensions, show first 2 dimensions
        ax1.scatter(data['x_train'][:, 0], data['x_train'][:, 1], c=data['y_train'], 
                   s=100, cmap='viridis', edgecolors='cyan', linewidths=2,
                   label=f'Train (n={data["n_train"]})')
        ax1.scatter(data['x_test'][:, 0], data['x_test'][:, 1], c=data['y_test'], 
                   s=100, cmap='viridis', marker='s', edgecolors='magenta', linewidths=2,
                   label=f'Test (n={data["n_test"]})')
        ax1.set_xlabel('x₁')
        ax1.set_ylabel('x₂')
        ax1.set_title(f'Training Data (showing 2/{d} dims)')
        plt.colorbar(ax1.collections[0], ax=ax1, label='y')
    
    if d <= 2:
        ax1.set_title('Training Data')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Double descent curve (mean across seeds)
    ax2 = plt.subplot(3, 3, 2)
    for lr in df['learning_rate'].unique():
        df_lr = df[df['learning_rate'] == lr]
        grouped = df_lr.groupby('hidden_units').agg({
            'test_loss': ['mean', 'std'],
            'n_params': 'first'
        }).reset_index()
        
        mean_loss = grouped['test_loss']['mean']
        std_loss = grouped['test_loss']['std']
        n_params = grouped['n_params']['first']
        
        label = f'LR={lr:.0e}' if lr != config.base_lr else f'LR={lr:.0e} (default)'
        ax2.plot(n_params, mean_loss, 'o-', label=label, linewidth=2, markersize=6)
        ax2.fill_between(n_params, mean_loss - std_loss, mean_loss + std_loss, 
                         alpha=0.2)
    
    ax2.axhline(config.noise_level**2, color='red', linestyle='--', 
                label='Noise floor', linewidth=2)
    ax2.set_xlabel('Number of Parameters')
    ax2.set_ylabel('Test Loss (MSE)')
    ax2.set_title('Double Descent Phenomenon')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    
    # 3. Train vs Test loss
    ax3 = plt.subplot(3, 3, 3)
    for lr in df['learning_rate'].unique():
        df_lr = df[df['learning_rate'] == lr]
        grouped = df_lr.groupby('hidden_units').agg({
            'train_loss': 'mean',
            'test_loss': 'mean',
            'n_params': 'first'
        }).reset_index()
        
        ax3.plot(grouped['n_params'], grouped['train_loss'], 
                'o-', label=f'Train (LR={lr:.0e})', alpha=0.7)
        ax3.plot(grouped['n_params'], grouped['test_loss'], 
                's--', label=f'Test (LR={lr:.0e})', alpha=0.7)
    
    ax3.set_xlabel('Number of Parameters')
    ax3.set_ylabel('Loss (MSE)')
    ax3.set_title('Train vs Test Loss')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, which='both')
    
    # 4-9. Sample fitted curves (dimension-aware)
    selected_widths = [4, 8, 16, 32, 256, 2048]
    for idx, h in enumerate(selected_widths):
        ax = plt.subplot(3, 3, idx + 4)
        
        d = data['dimension']
        
        if d == 1:
            ax.plot(data['all_x'].flatten(), data['all_y'], 'k-', label='True', linewidth=2)
            ax.scatter(data['x_train'].flatten(), data['y_train'], c='cyan', s=30, 
                      alpha=0.5, edgecolors='k', linewidths=0.5)
            
            # Plot predictions from different seeds
            df_h = df[df['hidden_units'] == h]
            for _, row in df_h.iterrows():
                curve_file = exp_dir / "curves" / f"{row['experiment_id']}.pkl"
                if curve_file.exists():
                    with open(curve_file, 'rb') as f:
                        curve_data = pickle.load(f)
                    ax.plot(curve_data['x_values'].flatten(), curve_data['predictions'], 
                           alpha=0.4, linewidth=1.5)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            
        elif d == 2:
            # For 2D, show heat map of predictions
            df_h = df[df['hidden_units'] == h]
            if len(df_h) > 0:
                # Use first seed for visualization
                row = df_h.iloc[0]
                curve_file = exp_dir / "curves" / f"{row['experiment_id']}.pkl"
                if curve_file.exists():
                    with open(curve_file, 'rb') as f:
                        curve_data = pickle.load(f)
                    
                    # Reshape predictions to grid
                    grid_size = int(np.sqrt(len(curve_data['predictions'])))
                    pred_grid = curve_data['predictions'].reshape(grid_size, grid_size)
                    x_vals = curve_data['x_values'][:, 0].reshape(grid_size, grid_size)
                    y_vals = curve_data['x_values'][:, 1].reshape(grid_size, grid_size)
                    
                    im = ax.contourf(x_vals, y_vals, pred_grid, levels=20, cmap='viridis', alpha=0.7)
                    ax.scatter(data['x_train'][:, 0], data['x_train'][:, 1], 
                              c='cyan', s=30, edgecolors='k', linewidths=0.5)
                    plt.colorbar(im, ax=ax)
            ax.set_xlabel('x₁')
            ax.set_ylabel('x₂')
            
        else:
            # For higher dimensions, show projection onto first 2 dims
            df_h = df[df['hidden_units'] == h]
            ax.scatter(data['x_train'][:, 0], data['x_train'][:, 1], c=data['y_train'],
                      s=30, cmap='viridis', alpha=0.5, edgecolors='k', linewidths=0.5)
            ax.set_xlabel('x₁')
            ax.set_ylabel('x₂')
            ax.text(0.5, 0.95, f'{d}D data (projection)', 
                   transform=ax.transAxes, ha='center', va='top', fontsize=8)
        
        mean_test_loss = df_h['test_loss'].mean()
        ax.set_title(f'Width={h}, Params={df_h.iloc[0]["n_params"]}\n'
                    f'Test Loss={mean_test_loss:.4f}', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(exp_dir / 'visualization.png', dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {exp_dir / 'visualization.png'}")
    
    # Create summary statistics
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"Total experiments: {len(df)}")
    print(f"Hidden unit configurations: {df['hidden_units'].nunique()}")
    print(f"Seeds per configuration: {df['torch_random_seed'].nunique()}")
    print(f"\nTest loss statistics:")
    print(df.groupby('hidden_units')['test_loss'].agg(['mean', 'std', 'min', 'max']))
    
    return fig

# ============================================================================
# Main Execution
# ============================================================================
if __name__ == "__main__":
    config = ExperimentConfig()
    
    # Run experiments
    df, data, exp_dir = run_experiments(config)
    
    # Create visualizations
    fig = create_visualizations(df, data, exp_dir, config)
    
    plt.show()
    
    print("\n" + "="*70)
    print("DONE! 🎉")
    print("="*70)