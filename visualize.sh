#!/bin/bash

# Visualization wrapper script for SO-100
# Usage: ./visualize.sh [checkpoint_path] [num_envs]

# Find Isaac Lab installation
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check for Isaac Lab
if [ -d "$HOME/IsaacLab" ]; then
    ISAACLAB_PATH="$HOME/IsaacLab"
elif [ -d "/workspace/IsaacLab" ]; then
    ISAACLAB_PATH="/workspace/IsaacLab"
else
    echo "Error: Could not find IsaacLab installation"
    exit 1
fi

# Default values
CHECKPOINT="${1:-logs/rsl_rl/so100/*/model_*.pt}"
NUM_ENVS="${2:-16}"

# Find latest checkpoint if wildcard provided
if [[ "$CHECKPOINT" == *"*"* ]]; then
    LATEST_CHECKPOINT=$(ls -t $CHECKPOINT 2>/dev/null | head -n 1)
    if [ -z "$LATEST_CHECKPOINT" ]; then
        echo "Error: No checkpoint found matching pattern: $CHECKPOINT"
        exit 1
    fi
    CHECKPOINT="$LATEST_CHECKPOINT"
fi

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    echo "Usage: $0 [checkpoint_path] [num_envs]"
    exit 1
fi

echo "========================================"
echo "SO-100 Policy Visualization"
echo "========================================"
echo "IsaacLab: $ISAACLAB_PATH"
echo "Checkpoint: $CHECKPOINT"
echo "Num Envs: $NUM_ENVS"
echo "========================================"

# Fix GLIBCXX compatibility issue
# Remove conda lib paths and use system libraries
CONDA_PREFIX_BACKUP=$CONDA_PREFIX
unset CONDA_PREFIX
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v "conda\|miniconda\|anaconda" | tr '\n' ':')
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Run visualization
"$ISAACLAB_PATH/isaaclab.sh" -p "$SCRIPT_DIR/visualize.py" \
    --checkpoint "$CHECKPOINT" \
    --num_envs "$NUM_ENVS"

# Restore conda prefix
export CONDA_PREFIX=$CONDA_PREFIX_BACKUP
