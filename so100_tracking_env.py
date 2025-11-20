# so100_tracking_env.py
"""Isaac Lab compatible version of the legacy `so100_tracking` task.

The original task (`isaac_gym_legacy/tasks/so100_tracking.py`) generated a
square trajectory for the robot's end‑effector and used a custom reward based on
the distance to that moving target.  The modern Isaac Lab environment
(`so100_env.py`) already implements the same square trajectory inside the
`So100Env` class.  This file simply provides a thin wrapper that mirrors the
legacy task name while re‑using the existing `So100Env` implementation.

Usage:
```bash
# In your training script (train.py) replace the import:
# from so100_env import So100Env, So100EnvCfg
# with the following:
from so100_tracking_env import So100TrackingEnv as So100Env
from so100_tracking_env import So100TrackingEnvCfg as So100EnvCfg
```
The rest of the training pipeline (`train.py`) stays unchanged.
"""

from isaaclab.utils import configclass
from so100_env import So100Env, So100EnvCfg

@configclass
class So100TrackingEnvCfg(So100EnvCfg):
    """Configuration class that inherits from :class:`So100EnvCfg`.

    It exists solely to give the environment a distinct name that matches the
    legacy task (`so100_tracking`).  All parameters (scene, observations,
    actions, rewards, etc.) are identical to the original ``So100Env``.
    """
    pass

class So100TrackingEnv(So100Env):
    """Compatibility wrapper for the legacy task.

    The implementation is exactly the same as :class:`So100Env`; the class is
    kept separate so that you can refer to the environment by the historic
    ``So100Tracking`` name without touching the rest of the code base.
    """
    pass

# Export symbols for ``from so100_tracking_env import *``
__all__ = ["So100TrackingEnv", "So100TrackingEnvCfg"]
