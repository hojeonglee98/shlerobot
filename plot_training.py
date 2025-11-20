#!/usr/bin/env python3
"""
Plot training metrics from SO-100 training logs.

Usage:
    python plot_training.py --log_dir logs/rsl_rl/so100
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_tensorboard_data(log_dir: Path):
    """Load training data from TensorBoard logs."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        print("Error: tensorboard package not found. Install with: pip install tensorboard")
        return None
    
    # Find event files
    event_files = list(log_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        print(f"No TensorBoard event files found in {log_dir}")
        return None
    
    print(f"Found {len(event_files)} event file(s)")
    
    # Load data from all event files
    all_data = {}
    for event_file in event_files:
        print(f"Loading: {event_file.name}")
        ea = event_accumulator.EventAccumulator(str(event_file))
        ea.Reload()
        
        # Get available scalars
        for tag in ea.Tags()['scalars']:
            if tag not in all_data:
                all_data[tag] = {'steps': [], 'values': []}
            
            events = ea.Scalars(tag)
            for event in events:
                all_data[tag]['steps'].append(event.step)
                all_data[tag]['values'].append(event.value)
    
    return all_data


def plot_training_metrics(log_dir: str, save_path: str = None):
    """Plot training metrics."""
    log_dir = Path(log_dir)
    
    if not log_dir.exists():
        print(f"Error: Log directory not found: {log_dir}")
        return
    
    # Load data
    data = load_tensorboard_data(log_dir)
    if data is None:
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'SO-100 Training Metrics\n{log_dir.name}', fontsize=14)
    
    # Plot reward
    if 'Train/mean_reward' in data or 'Rewards/mean_reward' in data:
        key = 'Train/mean_reward' if 'Train/mean_reward' in data else 'Rewards/mean_reward'
        ax = axes[0, 0]
        ax.plot(data[key]['steps'], data[key]['values'], linewidth=1.5)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Mean Reward')
        ax.set_title('Training Reward')
        ax.grid(True, alpha=0.3)
    
    # Plot episode length
    if 'Train/mean_episode_length' in data or 'Episode/mean_length' in data:
        key = 'Train/mean_episode_length' if 'Train/mean_episode_length' in data else 'Episode/mean_length'
        ax = axes[0, 1]
        ax.plot(data[key]['steps'], data[key]['values'], linewidth=1.5, color='orange')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Mean Episode Length')
        ax.set_title('Episode Length')
        ax.grid(True, alpha=0.3)
    
    # Plot learning rate
    if 'Train/learning_rate' in data or 'Policy/learning_rate' in data:
        key = 'Train/learning_rate' if 'Train/learning_rate' in data else 'Policy/learning_rate'
        ax = axes[1, 0]
        ax.plot(data[key]['steps'], data[key]['values'], linewidth=1.5, color='green')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    # Plot policy loss
    if 'Loss/policy_loss' in data or 'Train/policy_loss' in data:
        key = 'Loss/policy_loss' if 'Loss/policy_loss' in data else 'Train/policy_loss'
        ax = axes[1, 1]
        ax.plot(data[key]['steps'], data[key]['values'], linewidth=1.5, color='red')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Policy Loss')
        ax.set_title('Policy Loss')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot SO-100 training metrics")
    parser.add_argument("--log_dir", type=str, default="logs/rsl_rl/so100",
                        help="Path to training log directory")
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save plot (if not provided, will display)")
    args = parser.parse_args()
    
    plot_training_metrics(args.log_dir, args.save)
