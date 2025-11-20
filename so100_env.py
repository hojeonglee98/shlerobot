# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp

##
# Scene definition
##

@configclass
class So100SceneCfg(InteractiveSceneCfg):
    """Configuration for the SO-100 scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # robot
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path="assets/so100.urdf",
            fix_base=True,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                target_type="position",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=80.0, damping=40.0)
            ),
            make_instanceable=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "shoulder_pan": 0.0,
                "shoulder_lift": 0.0,
                "elbow_flex": 0.0,
                "wrist_flex": 0.0,
                "wrist_roll": 0.0,
                "gripper": 0.0,
            }
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
                effort_limit=10.0,
                velocity_limit=5.0,
                stiffness=80.0,
                damping=4.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper"],
                effort_limit=10.0,
                velocity_limit=5.0,
                stiffness=80.0,
                damping=4.0,
            ),
        },
    )

##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    pass

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # Joint position control
    arm_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
        scale=1.0,
        use_default_offset=True,
    )
    gripper_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper"],
        scale=1.0,
        use_default_offset=True,
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        
        # Joint positions
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        # Joint velocities
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        
        # Target position relative to robot base (in robot base frame)
        target_pos = ObsTerm(func=lambda env: env.target_pos_relative) 

    policy = PolicyCfg()

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Minimize distance to target
    track_target = RewTerm(
        func=lambda env: torch.exp(-env.dist_to_target / 0.1), 
        weight=1.0,
    )

    # Penalize large actions (smoothness)
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

##
# Environment configuration
##

@configclass
class So100EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the SO-100 RL environment."""
    
    # Scene settings
    scene: So100SceneCfg = So100SceneCfg(num_envs=8192, env_spacing=2.0)
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    decimation: int = 2 # Control at 30Hz

    # Post-initialization settings
    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        self.sim.dt = 0.0166 # 60Hz
        self.sim.render_interval = self.decimation
        self.episode_length_s = 8.0 # 8 seconds episode

##
# Environment Logic
##

class So100Env(ManagerBasedRLEnv):
    """
    SO-100 Environment with Square Trajectory Tracking Logic.
    Target position is provided in robot base frame coordinates.
    """
    def __init__(self, cfg: So100EnvCfg, render_mode: str | None = None, **kwargs):
        # Initialize tensors before super().__init__ so ObservationManager can access them
        temp_num_envs = cfg.scene.num_envs
        
        # Temporary CPU tensors (just for shape during manager initialization)
        self.traj_time = torch.zeros(temp_num_envs)
        self.target_pos = torch.zeros((temp_num_envs, 3))  # Robot base frame
        self.target_pos_relative = torch.zeros((temp_num_envs, 3))  # Robot base frame (for observation)
        self.dist_to_target = torch.zeros(temp_num_envs)
        
        self.traj_period = 4.0  
        self.traj_size = 0.1 
        
        # Initialize the parent class (here Managers are loaded)
        super().__init__(cfg, render_mode, **kwargs)
        
        # After super().__init__, self.device and self.num_envs are available
        # Re-create tensors on the correct device
        self.traj_center = torch.tensor([0.2, 0.0, 0.25], device=self.device)
        
        self.traj_time = torch.zeros(self.num_envs, device=self.device)
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device)  # Robot base frame
        self.target_pos_relative = torch.zeros((self.num_envs, 3), device=self.device)  # For observation
        self.dist_to_target = torch.zeros(self.num_envs, device=self.device)
        
        # Initialize target_pos to center immediately
        self.target_pos[:] = self.traj_center
        self.target_pos_relative[:] = self.traj_center

    def _reset_idx(self, env_ids: torch.Tensor | None):
        # Reset parent (physics, etc.)
        super()._reset_idx(env_ids)
        
        # Reset trajectory time for reset environments
        if env_ids is None:
            env_ids = slice(None)
        self.traj_time[env_ids] = 0.0
        
        # Reset target pos to center for reset environments
        self.target_pos[env_ids] = self.traj_center
        self.target_pos_relative[env_ids] = self.traj_center

    def step(self, action: torch.Tensor):
        # Ensure _pre_physics_step is called
        self._pre_physics_step(action)
        return super().step(action)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        Compute trajectory and update target_pos BEFORE physics step.
        Target is in robot base frame coordinates.
        """
        # 1. Update Time
        self.traj_time += self.step_dt
        
        # 2. Calculate Square Trajectory Logic
        # Period T = 4s. 0-1s: Right, 1-2s: Up, 2-3s: Left, 3-4s: Down
        phase = (self.traj_time % self.traj_period) / self.traj_period * 4.0 # 0 to 4
        
        half_size = self.traj_size / 2.0
        y_local = torch.zeros_like(self.traj_time)
        z_local = torch.zeros_like(self.traj_time)
        
        # Vectorized trajectory generation
        mask0 = (phase < 1.0)
        mask1 = (phase >= 1.0) & (phase < 2.0)
        mask2 = (phase >= 2.0) & (phase < 3.0)
        mask3 = (phase >= 3.0)
        
        # Phase 0: Bottom edge, moving right (y increases)
        y_local[mask0] = -half_size + (phase[mask0] - 0.0) * self.traj_size
        z_local[mask0] = -half_size
        
        # Phase 1: Right edge, moving up (z increases)
        y_local[mask1] = half_size
        z_local[mask1] = -half_size + (phase[mask1] - 1.0) * self.traj_size
        
        # Phase 2: Top edge, moving left (y decreases)
        y_local[mask2] = half_size - (phase[mask2] - 2.0) * self.traj_size
        z_local[mask2] = half_size
        
        # Phase 3: Left edge, moving down (z decreases)
        y_local[mask3] = -half_size
        z_local[mask3] = half_size - (phase[mask3] - 3.0) * self.traj_size
        
        # 3. Update Target Position (in robot base frame)
        self.target_pos[:, 0] = self.traj_center[0]
        self.target_pos[:, 1] = self.traj_center[1] + y_local
        self.target_pos[:, 2] = self.traj_center[2] + z_local
        
        # 4. Calculate Distance (for Reward)
        robot = self.scene["robot"]
        
        # End-effector link (gripper_link or last body)
        ee_idx = robot.find_bodies("gripper_link")[0]
        if len(ee_idx) == 0:
             ee_idx = [robot.num_bodies - 1]
             
        ee_pos_w = robot.data.body_pos_w[:, ee_idx[0], :]
        
        # Convert target from base frame to world frame for distance calculation
        target_pos_w = self.target_pos + self.scene.env_origins
        self.dist_to_target = torch.norm(ee_pos_w - target_pos_w, p=2, dim=-1)
        
        # 5. Update target_pos_relative for observation (already in base frame)
        self.target_pos_relative = self.target_pos.clone()