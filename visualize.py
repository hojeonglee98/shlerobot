#!/usr/bin/env python3
"""
Visualize a trained SO-100 policy in Isaac Sim.

Usage:
    python visualize.py --checkpoint /path/to/model.pt --num_envs 16
"""

import argparse
import torch
import numpy as np
from pathlib import Path

# Initialize Isaac Sim application
from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Visualize trained SO-100 policy")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to visualize")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Import after app launch
from so100_env import So100EnvCfg, So100Env

# Try to import debug draw for target visualization
try:
    import omni.isaac.debug_draw._debug_draw as debug_draw
    HAS_DEBUG_DRAW = True
except ImportError:
    HAS_DEBUG_DRAW = False
    print("[Warning] Debug draw not available - target visualization disabled")


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
    
    # Create a simple MLP policy matching RSL-RL default architecture
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
    
    # Try to load state dict with various prefix handling
    try:
        # First try direct load
        policy.load_state_dict(policy_state, strict=True)
        print("[INFO] Policy loaded successfully (direct)")
    except:
        # Try with prefix stripping
        filtered_state = {}
        for key, value in policy_state.items():
            # Remove common prefixes
            new_key = key
            for prefix in ["actor.", "policy.", "actor_critic.actor.", "model."]:
                if key.startswith(prefix):
                    new_key = key.replace(prefix, "net.", 1)
                    break
            filtered_state[new_key] = value
        
        try:
            policy.load_state_dict(filtered_state, strict=False)
            print("[INFO] Policy loaded successfully (with prefix removal)")
        except Exception as e:
            print(f"[WARNING] Could not load policy weights: {e}")
            print("[INFO] Using randomly initialized policy")
    
    policy.eval()
    return policy


def visualize_policy(checkpoint_path: str, num_envs: int = 16):
    """Run visualization loop with trained policy."""
    
    # Create environment configuration
    env_cfg = So100EnvCfg()
    env_cfg.scene.num_envs = num_envs
    
    # Create environment
    print(f"[INFO] Creating environment with {num_envs} instances...")
    env = So100Env(cfg=env_cfg, render_mode="human")
    
    # Get observation and action dimensions
    # obs_shape is a dict with group names, each containing a list of term shapes
    obs_terms = env.observation_manager._group_obs_term_dim["policy"]
    if isinstance(obs_terms, (list, tuple)):
        # Sum up all the dimensions
        obs_dim = sum(shape[0] if isinstance(shape, tuple) else shape for shape in obs_terms)
    else:
        obs_dim = obs_terms
    
    action_dim = env.action_manager.total_action_dim
    
    print(f"[INFO] Observation dim: {obs_dim}, Action dim: {action_dim}")
    
    # Load policy
    policy = load_policy(checkpoint_path, obs_dim, action_dim)
    
    # Initialize debug draw for target visualization
    if HAS_DEBUG_DRAW:
        draw = debug_draw.acquire_debug_draw_interface()
    
    # Reset environment
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    
    # Statistics tracking
    episode_rewards = torch.zeros(num_envs, device=env.device)
    episode_lengths = torch.zeros(num_envs, device=env.device)
    completed_episodes = 0
    total_reward = 0.0
    total_length = 0.0
    
    print("[INFO] Starting visualization (press Ctrl+C to stop)...")
    print("=" * 60)
    
    step_count = 0
    
    # Main loop
    while simulation_app.is_running():
        # Get action from policy
        with torch.no_grad():
            action = policy(obs)
        
        # Step environment
        obs_dict, reward, terminated, truncated, info = env.step(action)
        obs = obs_dict["policy"]
        
        # Update statistics
        episode_rewards += reward
        episode_lengths += 1
        
        # Visualize target positions
        if HAS_DEBUG_DRAW and step_count % 10 == 0:
            draw.clear_lines()
            # Draw target positions for first few environments
            for i in range(min(4, num_envs)):
                target_world = env.target_pos[i] + env.scene.env_origins[i]
                target_np = target_world.cpu().numpy()
                
                # Draw a small sphere at target
                draw.draw_point(target_np.tolist(), (1.0, 0.0, 0.0, 1.0), 10.0)
                
                # Draw cross at target
                size = 0.02
                draw.draw_line(
                    (target_np + np.array([-size, 0, 0])).tolist(),
                    (target_np + np.array([size, 0, 0])).tolist(),
                    (1.0, 0.0, 0.0, 1.0), 2.0
                )
                draw.draw_line(
                    (target_np + np.array([0, -size, 0])).tolist(),
                    (target_np + np.array([0, size, 0])).tolist(),
                    (1.0, 0.0, 0.0, 1.0), 2.0
                )
                draw.draw_line(
                    (target_np + np.array([0, 0, -size])).tolist(),
                    (target_np + np.array([0, 0, size])).tolist(),
                    (1.0, 0.0, 0.0, 1.0), 2.0
                )
        
        # Check for episode terminations
        dones = terminated | truncated
        if dones.any():
            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
            for idx in done_indices:
                completed_episodes += 1
                total_reward += episode_rewards[idx].item()
                total_length += episode_lengths[idx].item()
                
                if completed_episodes % 10 == 0:
                    avg_reward = total_reward / completed_episodes
                    avg_length = total_length / completed_episodes
                    print(f"Episodes: {completed_episodes:4d} | "
                          f"Avg Reward: {avg_reward:8.2f} | "
                          f"Avg Length: {avg_length:6.1f}")
            
            # Reset episode stats for done environments
            episode_rewards[done_indices] = 0
            episode_lengths[done_indices] = 0
        
        step_count += 1
    
    # Cleanup
    env.close()
    print("\n" + "=" * 60)
    print("[INFO] Visualization ended")
    if completed_episodes > 0:
        print(f"[INFO] Total episodes: {completed_episodes}")
        print(f"[INFO] Average reward: {total_reward / completed_episodes:.2f}")
        print(f"[INFO] Average length: {total_length / completed_episodes:.1f}")


if __name__ == "__main__":
    try:
        visualize_policy(args.checkpoint, args.num_envs)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        simulation_app.close()
