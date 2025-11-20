
import argparse
from isaaclab.app import AppLauncher

import sys

# Parse arguments for AppLauncher
parser = argparse.ArgumentParser(description="Train script", add_help=False)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Update sys.argv to ensure Hydra only sees arguments it should handle
sys.argv = [sys.argv[0]] + hydra_args

# Launch the app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import math

from isaaclab.envs import ManagerBasedRLEnv
from so100_env import So100EnvCfg, So100Env

# Import RSL-RL runner
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# So100Env is now imported from so100_env.py

@hydra.main(config_path=".", config_name="so100_ppo", version_base=None)
def main(cfg: DictConfig):
    import os
    from datetime import datetime
    
    # Create environment configuration
    env_cfg = So100EnvCfg()
    
    # Create environment
    env = So100Env(cfg=env_cfg)
    
    # Wrap environment for RSL-RL
    env = RslRlVecEnvWrapper(env)
    
    # Convert config to dict
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Set up log directory with absolute path
    experiment_name = cfg_dict.get("experiment_name", "so100_tracking")
    run_name = cfg_dict.get("run_name", "")
    if not run_name:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use absolute path for log directory
    log_root = os.path.abspath(cfg_dict.get("log_dir", "logs"))
    log_dir = os.path.join(log_root, experiment_name, run_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # Ensure log_dir is set in config
    cfg_dict["log_dir"] = log_dir
    print(f"[INFO] Logging to: {log_dir}")
    
    # Create runner
#    runner = OnPolicyRunner(env, cfg_dict)
    # Create runner
    # log_dir와 device를 명시적으로 전달해야 합니다.
    runner = OnPolicyRunner(env, cfg_dict, log_dir=log_dir, device=env.device)
    
    # Train
    runner.learn(num_learning_iterations=2000, init_at_random_ep_len=True)
    
    # Close
    env.close()

if __name__ == "__main__":
    main()
