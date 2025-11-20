
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

import numpy as np
import torch
import os

from tasks.base.vec_task import VecTask

class So100Tracking(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        
        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        super().__init__(config=cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # State tensors
        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(self.root_tensor)
        self.dof_state = gymtorch.wrap_tensor(self.dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.rigid_body_states = gymtorch.wrap_tensor(self.rigid_body_tensor).view(self.num_envs, -1, 13)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.num_bodies = self.gym.get_sim_rigid_body_count(self.sim) // self.num_envs
        
        # Initialize targets
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.initial_root_states = self.root_states.clone()
        self.initial_dof_pos = torch.zeros_like(self.dof_pos, device=self.device)
        self.initial_dof_vel = torch.zeros_like(self.dof_vel, device=self.device)

        # Trajectory parameters
        self.traj_t = torch.zeros(self.num_envs, device=self.device)
        self.traj_period = 4.0 # seconds for one full loop
        self.traj_size = 0.1 # 10cm square
        self.traj_center = torch.tensor([0.2, 0.0, 0.15], device=self.device) # Center of workspace

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "so100.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = False
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        asset_options.override_com = True
        asset_options.override_inertia = True

        so100_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        self.num_dof = self.gym.get_asset_dof_count(so100_asset)
        self.end_effector_index = self.gym.find_asset_rigid_body_index(so100_asset, "gripper_link")
        
        # If gripper_link is not found (collapsed), try another link
        if self.end_effector_index == -1:
             self.end_effector_index = self.gym.find_asset_rigid_body_index(so100_asset, "moving_jaw_so101_v1_link")
        
        if self.end_effector_index == -1:
            print("Warning: Could not find end effector link! Defaulting to last body.")
            self.end_effector_index = self.gym.get_asset_rigid_body_count(so100_asset) - 1

        so100_dof_props = self.gym.get_asset_dof_properties(so100_asset)
        self.so100_dof_lower_limits = torch.tensor(so100_dof_props['lower'], device=self.device)
        self.so100_dof_upper_limits = torch.tensor(so100_dof_props['upper'], device=self.device)
        self.so100_dof_lower_limits = torch.where(self.so100_dof_lower_limits == -float('inf'), torch.tensor(-np.pi, device=self.device), self.so100_dof_lower_limits)
        self.so100_dof_upper_limits = torch.where(self.so100_dof_upper_limits == float('inf'), torch.tensor(np.pi, device=self.device), self.so100_dof_upper_limits)

        so100_dof_props['driveMode'].fill(gymapi.DOF_MODE_POS)
        so100_dof_props['stiffness'].fill(400.0)
        so100_dof_props['damping'].fill(40.0)

        self.so100_handles = []
        self.envs = []

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            so100_handle = self.gym.create_actor(env_ptr, so100_asset, gymapi.Transform(), "so100", i, 1, 0)
            
            self.gym.set_actor_dof_properties(env_ptr, so100_handle, so100_dof_props)
            self.envs.append(env_ptr)
            self.so100_handles.append(so100_handle)

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_so100_reward(
            self.reset_buf, self.progress_buf, self.actions,
            self.end_effector_pos, self.target_pos,
            self.dist_reward_scale, self.action_penalty_scale, self.max_episode_length
        )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.end_effector_pos = self.rigid_body_states[:, self.end_effector_index, 0:3]

        self.obs_buf = torch.cat((
            self.dof_pos,
            self.dof_vel * self.dof_vel_scale,
            self.end_effector_pos,
            self.target_pos
        ), dim=-1)

    def reset_idx(self, env_ids):
        # Reset robot to initial state with some noise
        positions = 0.1 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)
        velocities = 0.1 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)

        self.dof_pos[env_ids, :] = positions
        self.dof_vel[env_ids, :] = velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        self.traj_t[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        # Map actions to target positions
        # Action is -1 to 1, map to joint limits
        targets = self.so100_dof_lower_limits + 0.5 * (actions + 1.0) * (self.so100_dof_upper_limits - self.so100_dof_lower_limits)
        
        targets_tensor = gymtorch.unwrap_tensor(targets)
        self.gym.set_dof_position_target_tensor(self.sim, targets_tensor)

    def post_physics_step(self):
        self.progress_buf += 1
        
        # Update trajectory targets
        dt = self.cfg["sim"]["dt"]
        self.traj_t += dt
        
        # Square trajectory logic in Y-Z plane
        # Period T = 4s. 0-1s: Right, 1-2s: Up, 2-3s: Left, 3-4s: Down
        phase = (self.traj_t % self.traj_period) / self.traj_period * 4.0 # 0 to 4
        
        # Local square coordinates (y, z)
        # 0: (0,0) -> (1,0)
        # 1: (1,0) -> (1,1)
        # 2: (1,1) -> (0,1)
        # 3: (0,1) -> (0,0)
        # Centered at 0: (-0.5, -0.5) to (0.5, 0.5)
        
        half_size = self.traj_size / 2.0
        
        y_local = torch.zeros_like(self.traj_t)
        z_local = torch.zeros_like(self.traj_t)
        
        # Vectorized trajectory generation
        mask0 = (phase < 1.0)
        mask1 = (phase >= 1.0) & (phase < 2.0)
        mask2 = (phase >= 2.0) & (phase < 3.0)
        mask3 = (phase >= 3.0)
        
        # Phase 0: Bottom edge, moving right (y increases)
        # y goes from -half to +half
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
        
        self.target_pos[:, 0] = self.traj_center[0]
        self.target_pos[:, 1] = self.traj_center[1] + y_local
        self.target_pos[:, 2] = self.traj_center[2] + z_local

        self.compute_observations()
        self.compute_reward()
        
        if self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self._draw_debug_viz()

    def _draw_debug_viz(self):
        # Draw target spheres
        for i in range(self.num_envs):
            p = self.target_pos[i]
            sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(0, 1, 0))
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], gymapi.Transform(p=gymapi.Vec3(p[0], p[1], p[2])))

@torch.jit.script
def compute_so100_reward(reset_buf, progress_buf, actions, end_effector_pos, target_pos, dist_reward_scale, action_penalty_scale, max_episode_length):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float) -> Tuple[Tensor, Tensor]
    
    # Distance reward
    d = torch.norm(end_effector_pos - target_pos, p=2, dim=-1)
    dist_reward = 1.0 / (1.0 + d * d)
    dist_reward = dist_reward * dist_reward_scale

    # Action penalty
    action_penalty = torch.sum(actions ** 2, dim=-1) * action_penalty_scale

    rewards = dist_reward - action_penalty
    
    # Reset
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)
    
    return rewards, reset
