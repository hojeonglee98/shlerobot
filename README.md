# SO-100 Robot RL Training with Isaac Lab

This project trains a reinforcement learning policy for the SO-100 manipulator to track a square trajectory using Isaac Lab.

## Project Structure

```
isaac_lab_so100/
├── train.py              # Main training script
├── visualize.py          # Policy visualization script
├── so100_env.py          # Environment configuration
├── so100_ppo.yaml        # PPO training hyperparameters
├── run.sh                # Training wrapper script
├── visualize.sh          # Visualization wrapper script
├── urdf/                 # Robot URDF files
│   ├── so100.urdf
│   └── assets/           # Mesh files
└── logs/                 # Training logs and checkpoints
```

## Setup

1. **Isaac Lab Installation**: Ensure Isaac Lab is installed at `/home/cml/IsaacLab`
2. **Dependencies**: All dependencies are installed via `isaaclab.sh --install`

## Training

### Start Training

To train the SO-100 robot for trajectory tracking:

```bash
./run.sh
```

This will:
- Initialize 512 parallel environments
- Train using PPO algorithm
- Save checkpoints to `logs/so100_tracking/<timestamp>/`
- Log training metrics to TensorBoard

### Monitor Training

View training progress with TensorBoard:

```bash
tensorboard --logdir logs/so100_tracking
```

### Training Configuration

Edit `so100_ppo.yaml` to modify:
- Learning rate
- Network architecture
- PPO hyperparameters
- Number of training iterations

## Visualization

### Visualize Trained Policy

After training completes, visualize the learned policy:

```bash
./visualize.sh logs/so100_tracking/<timestamp>/model_<iteration>.pt
```

Example:
```bash
./visualize.sh logs/so100_tracking/20250119_162000/model_1000.pt
```

### Multiple Environments

Visualize with multiple parallel environments:

```bash
./visualize.sh logs/so100_tracking/20250119_162000/model_1000.pt 4
```

## Environment Details

### Task Description
- **Objective**: Control SO-100 arm to track a square trajectory in 3D space
- **Trajectory**: 0.1m × 0.1m square at position [0.2, 0.0, 0.15]
- **Period**: 4 seconds per cycle

### Observations (15-dim)
- Joint positions (6)
- Joint velocities (6)
- Target position (3)

### Actions (6-dim)
- Arm joint positions (5)
- Gripper position (1)

### Rewards
- `track_target`: Distance to target position (weight: 1.0)
- `action_rate`: Action smoothness penalty (weight: -0.01)

## Key Features

- **Parallel Simulation**: 512 environments for efficient training
- **Physics**: PhysX simulation at 60Hz
- **Control**: 30Hz control frequency (decimation=2)
- **Neural Network**: 
  - Actor: 15 → 256 → 128 → 64 → 6
  - Critic: 15 → 256 → 128 → 64 → 1

## Troubleshooting

### Module Not Found Errors
Always use `./run.sh` or `./visualize.sh` instead of running Python directly. These wrappers ensure the correct Isaac Sim Python environment is used.

### URDF Warnings
Warnings about missing mesh files are non-critical - physics will still work correctly.

### Training Progress
Check `logs/so100_tracking/<timestamp>/` for:
- Model checkpoints (`.pt` files)
- Training metrics
- Configuration files

## Credits

- **Isaac Lab**: NVIDIA Isaac Lab robotics framework
- **RSL-RL**: ETH Zurich's RL library for robotics
- **SO-100**: LeRobot manipulator platform
