
import isaacgym
import isaacgymenvs
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys

from tasks.so100_tracking import So100Tracking
from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

## OmegaConf & Hydra setup
# We need to register our custom task resolver if we were using the full isaacgymenvs registry,
# but here we will just manually load the config and instantiate the runner.

# Helper to resolve config paths
def get_config_path():
    return os.path.join(os.path.dirname(__file__), "cfg")

@hydra.main(config_name="config", config_path="cfg")
def launch_rlg_hydra(cfg: DictConfig):
    # Ensure we can find the task config
    if "task" not in cfg:
        # Load default task config if not present (though hydra should handle this if structure is correct)
        # For simplicity, we assume the user runs with overrides or we load manually if needed.
        pass

    # Set seed
    set_seed(cfg.seed, torch.cuda.is_available())

    # Register the custom task
    from isaacgymenvs.tasks import isaacgym_task_map
    # We can inject our task class here if we were using isaacgymenvs.make
    # But we are using rl_games directly with a custom env creator.
    
    def create_env(**kwargs):
        # Extract env config
        env_cfg = omegaconf_to_dict(cfg.task)
        
        # Create the environment
        env = So100Tracking(
            cfg=env_cfg,
            rl_device=cfg.rl_device,
            sim_device=cfg.sim_device,
            graphics_device_id=cfg.graphics_device_id,
            headless=cfg.headless,
            virtual_screen_capture=False,
            force_render=False
        )
        return env

    # Register the env for rl_games
    vecenv.register('So100Tracking', lambda config_name, num_actors, **kwargs: create_env(**kwargs))
    env_configurations.register('So100Tracking', {
        'vecenv_type': 'So100Tracking',
        'env_creator': lambda **kwargs: create_env(**kwargs),
    })

    # Prepare rl_games config
    rlg_config_dict = omegaconf_to_dict(cfg.train)
    
    # Run
    runner = Runner()
    runner.load(rlg_config_dict)
    runner.reset()
    runner.run({
        'train': not cfg.test,
        'play': cfg.test,
        'checkpoint': cfg.checkpoint,
        'sigma': None
    })

if __name__ == "__main__":
    # Create a default config.yaml for hydra if it doesn't exist, or just pass arguments manually
    # To make it easier, we will create a dummy config.yaml in cfg/
    launch_rlg_hydra()
