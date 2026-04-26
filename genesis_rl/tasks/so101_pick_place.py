"""
SO101 Pick and Place Environment for Genesis
Task: Pick up a cube from a table and place it at a target location
Uses only pure genesis (no genesis-forge dependency)
"""
import numpy as np
import torch
import genesis as gs
from genesis import options
from typing import Dict, Optional, Tuple
from tensordict import TensorDict

class SO101PickPlaceEnv:
    """
    Pick and place environment for SO101 robot arm.

    Observation space (28 dim):
        - joint_pos (6): Current joint positions (6th is gripper)
        - joint_vel (6): Current joint velocities (6th is gripper)
        - ee_pos (3): End-effector position
        - ee_quat (4): End-effector orientation quaternion
        - object_pos (3): Object position
        - object_rel_pos (3): Object position relative to EE
        - target_pos (3): Target position for placing

    Action space (6 dim):
        - joint_pos_target (6): Target joint positions
           [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
           (6th DOF is the gripper joint)
    """

    def __init__(
        self,
        env_cfg: Dict,
        device: str = "cuda",
        show_viewer: bool = False,
    ):
        self.cfg = env_cfg
        self.num_envs = env_cfg["env"].get("num_envs", 1)
        self.max_episode_length = env_cfg["env"].get("episode_length", 3000)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        env_spacing=env_cfg["env"].get("env_spacing", 2.0)

        # Joint names (from MuJoCo XML)
        self.joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",  # 6th DOF is the gripper joint
        ]
        self.num_joints = len(self.joint_names)

        # Task parameters
        self.table_height = 0.5
        self.object_size = 0.05
        self.success_threshold = 0.05
        self.grasp_threshold = 0.03

        # Dimensions
        self.obs_dim = 28
        self.action_dim = 6
        self.num_actions = 6

        # Initialize Genesis
        gs.init(logging_level="error")

        # Create scene
        self.scene = gs.Scene(show_viewer=show_viewer)

        # Add ground
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        # ground = options.morphs.Box(size=(10, 10, 0.1), pos=(0, 0, -0.1))
        # self.scene.add_entity(ground)

        # Add table
        self.table_path = "../src/lerobot_robots_description/urdf/objects/table.urdf"
        table = options.morphs.URDF(file=self.table_path, pos=(0, 0, 0.8), fixed=True)
        # table = options.morphs.Box(size=(1.2, 0.55, 0.05), pos=(0, 0, 0.8), fixed=True)
        self.scene.add_entity(table)

        # Add robot (from URDF)
        # Robot URDF/MJCF path
        self.robot_path = "../src/lerobot_robots_description/urdf/SO101/so101_new_calib.urdf"
        robot = options.morphs.URDF(file=self.robot_path, pos=(-0.4, 0.25, 0.8), fixed=True)
        self.robot_entity = self.scene.add_entity(robot)

        # Add object (pink sponge to be picked up)
        self.object_path = "../src/lerobot_robots_description/urdf/objects/pink_sponge.urdf"
        obj = options.morphs.URDF(file=self.object_path, pos=(0, 0.25, 0.9))
        self.object = self.scene.add_entity(obj)

        # Add target (container to place sponge in)
        self.target_path = "../src/lerobot_robots_description/urdf/objects/container.urdf"
        target = options.morphs.URDF(file=self.target_path, pos=(0, 0.25, 0.9))
        self.target = self.scene.add_entity(target)

        # Build scene
        self.scene.build(n_envs=self.num_envs, env_spacing=(env_spacing, env_spacing))

        # Get joint indices (DOF indices for each joint)
        self.joint_indices = []
        for name in self.joint_names:
            joint = self.robot_entity.get_joint(name)
            # For each joint, get the DOF start index
            self.joint_indices.append(joint.dof_start)

        # Track progress
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.prev_actions = torch.zeros((self.num_envs, self.action_dim), device=self.device)

    def get_observations(self):
        """Get observations for RSL-RL interface."""
        return self._compute_observations()

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Reset environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Reset robot pose
        default_pos = torch.tensor([0.0, -0.5, 1.0, 0.0, 0.0, 0.0], device=self.device)
        self.robot_entity.set_dofs_position(default_pos.repeat(len(env_ids), 1), self.joint_indices, envs_idx=env_ids)

        # Randomize sponge (object) position and orientation (z-axis only)
        obj_pos = torch.zeros(len(env_ids), 3, device=self.device)
        obj_pos[:, 0] = torch.rand(len(env_ids), device=self.device) * 0.2 - 0.5
        obj_pos[:, 1] = torch.rand(len(env_ids), device=self.device) * 0.2 - 0.2
        obj_pos[:, 2] = 0.9
        self.object.set_pos(obj_pos, envs_idx=env_ids)
        
        # Randomize sponge orientation (z-axis rotation only)
        obj_orn = torch.zeros(len(env_ids), 4, device=self.device)
        # Generate random quaternions from uniform distribution
        # Using method: sample from normal distribution and normalize
        obj_orn_raw = torch.randn(len(env_ids), 4, device=self.device)
        obj_orn = obj_orn_raw / torch.norm(obj_orn_raw, dim=1, keepdim=True)

        # x and y remain 0 (no roll/pitch)
        self.object.set_quat(obj_orn, envs_idx=env_ids)

        # Randomize target (container) position and orientation (full 3D)
        tgt_pos = torch.zeros(len(env_ids), 3, device=self.device)
        tgt_pos[:, 0] = torch.rand(len(env_ids), device=self.device) * 0.2 - 0.5
        tgt_pos[:, 1] = torch.rand(len(env_ids), device=self.device) * 0.3 - 0.25
        tgt_pos[:, 2] = 0.83
        self.target.set_pos(tgt_pos, envs_idx=env_ids)
        
        # Randomize target orientation (full 3D rotation)
        tgt_orn = torch.zeros(len(env_ids), 4, device=self.device)
        # Random yaw angle (rotation around z-axis)
        yaw = torch.rand(len(env_ids), device=self.device) * 2 * np.pi
        # Convert to quaternion: [w, x, y, z] where w=cos(yaw/2), z=sin(yaw/2)
        tgt_orn[:, 0] = torch.cos(yaw / 2)  # w
        tgt_orn[:, 3] = torch.sin(yaw / 2)  # z
        self.target.set_quat(tgt_orn, envs_idx=env_ids)

        # Reset buffers
        self.episode_length_buf[env_ids] = 0
        self.prev_actions[env_ids] = 0.0

        # Step to stabilize
        self.scene.step()

        return self._compute_observations()

    def step(self, actions: torch.Tensor) -> Tuple[Dict, torch.Tensor, torch.Tensor, Dict]:
        """Step the environment."""
        # Apply joint targets
        joint_targets = actions[:, :6]
        self.robot_entity.set_dofs_position(joint_targets, self.joint_indices)

        # Step simulation
        self.scene.step()
        self.episode_length_buf += 1

        # Compute
        obs = self._compute_observations()
        rewards = self._compute_rewards(obs, actions)
        dones = self._compute_terminations(obs)

        # Reset done envs
        if dones.any():
            self.reset(torch.where(dones)[0])

        info = {"success": self._check_success(obs)}
        return obs, rewards, dones, info

    def _compute_observations(self) -> torch.Tensor:
        """Compute observations."""
        joint_pos = self.robot_entity.get_dofs_position(self.joint_indices)
        joint_vel = self.robot_entity.get_dofs_velocity(self.joint_indices)

        ee_link = self.robot_entity.get_link("gripper")
        ee_pos = ee_link.get_pos()
        ee_quat = ee_link.get_quat()

        object_pos = self.object.get_pos()
        object_rel_pos = object_pos - ee_pos
        target_pos = self.target.get_pos()

        return TensorDict({
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
            "object_pos": object_pos,
            "object_rel_pos": object_rel_pos,
            "target_pos": target_pos,
        })

    def _compute_rewards(self, obs: Dict, actions: torch.Tensor) -> torch.Tensor:
        """Compute rewards."""
        rewards = torch.zeros(self.num_envs, device=self.device)

        rel_pos = obs["object_rel_pos"]
        rewards += 2.0 * (-torch.norm(rel_pos, dim=-1))

        dist = torch.norm(rel_pos, dim=-1)
        rewards += 5.0 * (dist < self.grasp_threshold).float()

        obj_pos = obs["object_pos"]
        target_pos = obs["target_pos"]
        dist_target = torch.norm(target_pos - obj_pos, dim=-1)
        rewards += 5.0 * (-dist_target)
        rewards += 10.0 * (dist_target < self.success_threshold).float()

        action_diff = torch.norm(actions - self.prev_actions, dim=-1)
        rewards += -0.1 * action_diff
        rewards += 20.0 * (dist_target < self.success_threshold).float()

        self.prev_actions = actions.clone()
        return rewards

    def _compute_terminations(self, obs: Dict) -> torch.Tensor:
        timeout = self.episode_length_buf >= self.max_episode_length
        success = self._check_success(obs)
        return timeout | success

    def _check_success(self, obs: Dict) -> torch.Tensor:
        return torch.norm(obs["target_pos"] - obs["object_pos"], dim=-1) < self.success_threshold

    def close(self):
        pass