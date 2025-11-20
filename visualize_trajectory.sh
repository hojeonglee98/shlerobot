#!/bin/bash

# Trajectory visualization wrapper script for SO-100
# Usage: ./visualize_trajectory.sh [checkpoint_path] [duration] [save_path]

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
CHECKPOINT="${1:-logs/so100_tracking/*/model_*.pt}"
DURATION="${2:-8.0}"
SAVE_PATH="${3:-}"

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
    echo "Usage: $0 [checkpoint_path] [duration] [save_path]"
    exit 1
fi

echo "========================================"
echo "SO-100 Trajectory Visualization"
echo "========================================"
echo "IsaacLab: $ISAACLAB_PATH"
echo "Checkpoint: $CHECKPOINT"
echo "Duration: ${DURATION}s"
if [ -n "$SAVE_PATH" ]; then
    echo "Save to: $SAVE_PATH"
fi
echo "========================================"

# Fix GLIBCXX compatibility issue
# Remove conda lib paths and use system libraries
CONDA_PREFIX_BACKUP=$CONDA_PREFIX
unset CONDA_PREFIX
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v "conda\|miniconda\|anaconda" | tr '\n' ':')
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Build command
CMD="$ISAACLAB_PATH/isaaclab.sh -p $SCRIPT_DIR/visualize_trajectory.py --checkpoint $CHECKPOINT --duration $DURATION"

# Add save path if provided
if [ -n "$SAVE_PATH" ]; then
    CMD="$CMD --save $SAVE_PATH"
fi

# Run visualization
eval $CMD

# Restore conda prefix
export CONDA_PREFIX=$CONDA_PREFIX_BACKUP
