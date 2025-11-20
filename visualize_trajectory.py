#!/usr/bin/env python3
"""
Visualize SO-100 target and end-effector trajectories in 3D.

Usage:
    python visualize_trajectory.py --checkpoint /path/to/model.pt --duration 8.0
"""

import argparse
import torch
import numpy as np

# Set matplotlib backend before importing pyplot (for headless operation)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# Initialize Isaac Sim application
from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Visualize SO-100 trajectories in 3D")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
parser.add_argument("--duration", type=float, default=8.0, help="Duration to record (seconds)")
parser.add_argument("--save", type=str, default="trajectory_3d.png",
                    help="Path to save plot (default: trajectory_3d.png)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch Isaac Sim headless
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Import after app launch
from so100_env import So100EnvCfg, So100Env


def load_policy(checkpoint_path: str, obs_dim: int, action_dim: int):
    """Load a trained policy from checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"[INFO] Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cuda:0")
    
    # Extract policy network
    if "model_state_dict" in checkpoint:
        policy_state = checkpoint["model_state_dict"]
    elif "policy" in checkpoint:
        policy_state = checkpoint["policy"]
    elif "actor_critic" in checkpoint:
        policy_state = checkpoint["actor_critic"]
    else:
        policy_state = checkpoint
    
    # Create a simple MLP policy
    class MLPPolicy(torch.nn.Module):
        def __init__(self, obs_dim, action_dim, hidden_dims=[256, 128, 64]):
            super().__init__()
            layers = []
            prev_dim = obs_dim
            for hidden_dim in hidden_dims:
                layers.append(torch.nn.Linear(prev_dim, hidden_dim))
                layers.append(torch.nn.ELU())
                prev_dim = hidden_dim
            layers.append(torch.nn.Linear(prev_dim, action_dim))
            self.net = torch.nn.Sequential(*layers)
        
        def forward(self, obs):
            return self.net(obs)
    
    policy = MLPPolicy(obs_dim, action_dim).to("cuda:0")
    
    # Try to load state dict
    try:
        policy.load_state_dict(policy_state, strict=True)
        print("[INFO] Policy loaded successfully")
    except:
        # Try with prefix stripping
        filtered_state = {}
        for key, value in policy_state.items():
            new_key = key
            for prefix in ["actor.", "policy.", "actor_critic.actor.", "model."]:
                if key.startswith(prefix):
                    new_key = key.replace(prefix, "net.", 1)
                    break
            filtered_state[new_key] = value
        
        policy.load_state_dict(filtered_state, strict=False)
        print("[INFO] Policy loaded successfully (with prefix removal)")
    
    policy.eval()
    return policy


def record_trajectories(checkpoint_path: str, duration: float = 8.0):
    """Record target and end-effector trajectories."""
    
    # Create environment (single instance for clean trajectory)
    env_cfg = So100EnvCfg()
    env_cfg.scene.num_envs = 1
    
    print("[INFO] Creating environment...")
    env = So100Env(cfg=env_cfg, render_mode=None)
    
    # Get observation and action dimensions
    obs_terms = env.observation_manager._group_obs_term_dim["policy"]
    if isinstance(obs_terms, (list, tuple)):
        obs_dim = sum(shape[0] if isinstance(shape, tuple) else shape for shape in obs_terms)
    else:
        obs_dim = obs_terms
    
    action_dim = env.action_manager.total_action_dim
    
    print(f"[INFO] Observation dim: {obs_dim}, Action dim: {action_dim}")
    
    # Load policy
    policy = load_policy(checkpoint_path, obs_dim, action_dim)
    
    # Reset environment
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    
    # Storage for trajectories
    target_positions = []
    ee_positions = []
    timesteps = []
    
    # Calculate number of steps
    num_steps = int(duration / env.step_dt)
    
    print(f"[INFO] Recording {duration}s trajectory ({num_steps} steps)...")
    
    # Run simulation and record
    for step in range(num_steps):
        # Get action from policy
        with torch.no_grad():
            action = policy(obs)
        
        # Step environment
        obs_dict, reward, terminated, truncated, info = env.step(action)
        obs = obs_dict["policy"]
        
        # Record positions
        robot = env.scene["robot"]
        ee_idx = robot.find_bodies("gripper_link")[0]
        if len(ee_idx) == 0:
            ee_idx = [robot.num_bodies - 1]
        
        # Get positions in world frame
        ee_pos_world = robot.data.body_pos_w[0, ee_idx[0], :].cpu().numpy()
        target_pos_world = (env.target_pos[0] + env.scene.env_origins[0]).cpu().numpy()
        
        # Get robot base position (convert to base frame)
        robot_base_pos = env.scene.env_origins[0].cpu().numpy()
        
        # Convert to robot base frame for visualization
        ee_pos = ee_pos_world - robot_base_pos
        target_pos = target_pos_world - robot_base_pos
        
        target_positions.append(target_pos.copy())
        ee_positions.append(ee_pos.copy())
        timesteps.append(step * env.step_dt)
        
        # Print progress
        if (step + 1) % 100 == 0:
            print(f"  Step {step + 1}/{num_steps} ({100 * (step + 1) / num_steps:.1f}%)")
    
    # Convert to numpy arrays
    target_positions = np.array(target_positions)
    ee_positions = np.array(ee_positions)
    timesteps = np.array(timesteps)
    
    # Cleanup
    env.close()
    
    print("[INFO] Recording complete!")
    
    return target_positions, ee_positions, timesteps


def plot_trajectories(target_pos, ee_pos, timesteps, save_path=None):
    """Create 3D visualization of trajectories."""
    
    # Convert from meters to centimeters for better readability
    target_pos_cm = target_pos * 100
    ee_pos_cm = ee_pos * 100
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(16, 7))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Plot target trajectory
    ax1.plot(target_pos_cm[:, 0], target_pos_cm[:, 1], target_pos_cm[:, 2], 
             'r-', linewidth=2, alpha=0.7, label='Target')
    
    # Plot end-effector trajectory
    ax1.plot(ee_pos_cm[:, 0], ee_pos_cm[:, 1], ee_pos_cm[:, 2], 
             'b-', linewidth=1.5, alpha=0.8, label='End-Effector')
    
    # Mark start and end points
    ax1.scatter(*target_pos_cm[0], color='red', s=100, marker='o', label='Start')
    ax1.scatter(*ee_pos_cm[0], color='blue', s=100, marker='o')
    ax1.scatter(*target_pos_cm[-1], color='darkred', s=100, marker='s', label='End')
    ax1.scatter(*ee_pos_cm[-1], color='darkblue', s=100, marker='s')
    
    ax1.set_xlabel('X (cm)')
    ax1.set_ylabel('Y (cm)')
    ax1.set_zlabel('Z (cm)')
    ax1.set_title('3D Trajectory Visualization (Robot Base Frame)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Set equal aspect ratio
    max_range = np.array([
        target_pos_cm[:, 0].max() - target_pos_cm[:, 0].min(),
        target_pos_cm[:, 1].max() - target_pos_cm[:, 1].min(),
        target_pos_cm[:, 2].max() - target_pos_cm[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (target_pos_cm[:, 0].max() + target_pos_cm[:, 0].min()) * 0.5
    mid_y = (target_pos_cm[:, 1].max() + target_pos_cm[:, 1].min()) * 0.5
    mid_z = (target_pos_cm[:, 2].max() + target_pos_cm[:, 2].min()) * 0.5
    
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Tracking error over time
    ax2 = fig.add_subplot(122)
    
    tracking_error = np.linalg.norm(ee_pos - target_pos, axis=1)
    
    ax2.plot(timesteps, tracking_error * 1000, 'g-', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Tracking Error (mm)')
    ax2.set_title('Tracking Error Over Time')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    mean_error = tracking_error.mean() * 1000
    max_error = tracking_error.max() * 1000
    ax2.axhline(mean_error, color='r', linestyle='--', alpha=0.7, 
                label=f'Mean: {mean_error:.2f} mm')
    ax2.legend()
    
    # Add overall statistics as text
    stats_text = f'Mean Error: {mean_error:.2f} mm\n'
    stats_text += f'Max Error: {max_error:.2f} mm\n'
    stats_text += f'Std Error: {tracking_error.std() * 1000:.2f} mm'
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Plot saved to: {save_path}")
    else:
        plt.show()
    
    # Print statistics
    print("\n" + "=" * 60)
    print("TRAJECTORY STATISTICS")
    print("=" * 60)
    print(f"Duration: {timesteps[-1]:.2f} seconds")
    print(f"Number of samples: {len(timesteps)}")
    print(f"Mean tracking error: {mean_error:.3f} mm")
    print(f"Max tracking error: {max_error:.3f} mm")
    print(f"Std tracking error: {tracking_error.std() * 1000:.3f} mm")
    print("=" * 60)


if __name__ == "__main__":
    try:
        # Record trajectories
        target_pos, ee_pos, timesteps = record_trajectories(args.checkpoint, args.duration)
        
        print("[INFO] Creating plot...")
        
        # Plot results (do this BEFORE closing simulation)
        plot_trajectories(target_pos, ee_pos, timesteps, args.save)
        
        print("[INFO] Plot creation complete!")
        
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close simulation at the very end
        try:
            simulation_app.close()
        except:
            pass
