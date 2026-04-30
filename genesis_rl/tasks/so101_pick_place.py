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
import IPython

class SO101PickPlaceEnv:
    """
    Pick and place environment for SO101 robot arm.

    Observation space (28 dim):
        - joint_pos (6): Current joint positions (6th is gripper)
        - joint_vel (6): Current joint velocities (6th is gripper)
        - ee_pos (3): End-effector position
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
        self.success_threshold = 0.01
        self.grasp_threshold = 0.03
        self.table_surface_z = 0.835  # Object placed at this height on table

        # Dimensions
        self.obs_dim = 28
        self.action_dim = 6
        self.num_actions = 6

        # Initialize Genesis
        gs.init(
            seed                = None,
            precision           = '64',
            debug               = False,
            eps                 = 1e-12,
            logging_level       = "error",
            theme               = 'dark',
            logger_verbose_time = False
        )

        # Create scene
        self.ctrl_dt = 0.01
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.ctrl_dt,
                substeps=10,
                # requires_grad=True,
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.ctrl_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                constraint_timeconst=0.002,
                iterations=200,
                ls_iterations=200,
                tolerance=1e-7,
                ls_tolerance=1e-3
            ),
            # mpm_options=gs.options.MPMOptions(
            #     lower_bound=(-0.5, -0.5, 0.0),
            #     upper_bound=(0.5, 0.5, 1.0),
            # ),
            # vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(10))),
            # viewer_options=gs.options.ViewerOptions(
            #     max_FPS=int(0.5 / self.ctrl_dt),
            #     camera_pos=(2.0, 0.0, 2.5),
            #     camera_lookat=(0.0, 0.0, 0.5),
            #     camera_fov=40,
            # ),
            # profiling_options=gs.options.ProfilingOptions(show_FPS=False),
            # renderer=gs.options.renderers.BatchRenderer(
            #     use_rasterizer=env_cfg["use_rasterizer"],
            # ),
            show_viewer=show_viewer
        )

        # Add ground
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True, collision=True))
        # ground = options.morphs.Box(size=(10, 10, 0.1), pos=(0, 0, -0.1))
        # self.scene.add_entity(ground)

        # Add table
        self.table_path = "../src/lerobot_robots_description/urdf/objects/table.urdf"
        table = options.morphs.URDF(file=self.table_path, pos=(0, 0, 0.8), fixed=True, collision=True)
        # table = options.morphs.Box(size=(1.2, 0.55, 0.05), pos=(0, 0, 0.8), fixed=True)
        self.scene.add_entity(table)

        # Add robot (from URDF)
        # Robot URDF/MJCF path
        self.robot_path = "../src/lerobot_robots_description/urdf/SO101/so101_new_calib.urdf"
        robot = options.morphs.URDF(file=self.robot_path, pos=(-0.4, 0.25, 0.8), fixed=True, collision=True, links_to_keep=["ee_wrist", "ee_gripper"], convexify=True)
        self.robot_entity = self.scene.add_entity(
            robot,
            material=gs.materials.Rigid(
                friction=2.0,
                sdf_cell_size=1e-3,
            )
        )

        # Add object (pink sponge to be picked up)
        self.object_path = "../src/lerobot_robots_description/urdf/objects/pink_sponge.urdf"
        obj = options.morphs.URDF(file=self.object_path, pos=(0, 0.25, 0.9), collision=True)
        # obj_box = options.morphs.Box(size=(0.045, 0.021, 0.017), pos=(0, 0.25, 0.9), collision=True)
        obj_box = options.morphs.Box(size=(0.03, 0.03, 0.03), pos=(0, 0.25, 0.9), quat=(1.0, 0.0, 0.0, 0.0), collision=True)
        self.object = self.scene.add_entity(
            obj_box,
            material=gs.materials.Rigid(
                friction=5.0,
                sdf_cell_size=1e-3,
            )
        )

        # Add target (container to place sponge in)
        self.target_path = "../src/lerobot_robots_description/urdf/objects/container.urdf"
        target = options.morphs.URDF(file=self.target_path, pos=(0, 0.25, 0.9), collision=True)
        self.target = self.scene.add_entity(target)

        # Build scene
        self.scene.build(n_envs=self.num_envs, env_spacing=(env_spacing, env_spacing))

        self.joint_pos_min, self.joint_pos_max = self.robot_entity.get_dofs_limit()

        # Get joint indices (DOF indices for each joint)
        self.joint_indices = []
        for name in self.joint_names:
            joint = self.robot_entity.get_joint(name)
            # For each joint, get the DOF start index
            self.joint_indices.append(joint.dof_start)

        # Track progress
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.init_object_positions = torch.zeros((self.num_envs, 3), device=self.device)
        self.init_target_positions = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, self.action_dim), device=self.device)

        if 0:  # for debug
            import genesis as gn
            from genesis.utils.geom import trans_quat_to_T
            import numpy as np
            self = env
            marker_frame = self.scene.draw_debug_frame(
                trans_quat_to_T(np.array(self.robot_entity.get_link("gripperframe").get_pos().tolist()[0]), np.array([1.0, 0.0, 0.0, 0.0])),
                axis_length=0.1,   # 各軸の長さ
                origin_size=0.01,  # 原点の球の大きさ
                axis_radius=0.005  # 軸の太さ
            )
            # self.scene.add_entity(options.morphs.Box(size=(0.1, 0.1, 0.1), pos=[-0.3794, -0.0275,  1.0668]))
            # self.scene.add_entity(options.morphs.Box(size=(0.1, 0.1, 0.1), pos=robot.get_link('gripper').get_pos()))


    def get_observations(self):
        """Get observations for RSL-RL interface."""
        return self._compute_observations()

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Reset environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Reset robot pose. All 0 is good pose.
        default_pos = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=self.device)
        self.robot_entity.set_dofs_position(default_pos.repeat(len(env_ids), 1), self.joint_indices, envs_idx=env_ids)

        # Randomize sponge (object) position and orientation (z-axis only)
        # Table surface is at z=0.8 + 0.025 = 0.825m
        # Sponge height=0.017m, so center should be at z=0.825 + 0.0085 = 0.8335m
        obj_pos = torch.zeros(len(env_ids), 3, device=self.device)
        obj_pos[:, 0] = torch.rand(len(env_ids), device=self.device) * 0.2 * 0.0 - 0.4
        obj_pos[:, 1] = torch.rand(len(env_ids), device=self.device) * 0.2 * 0.0 - 0.1
        obj_pos[:, 2] = 0.88  # On table surface
        self.object.set_pos(obj_pos, envs_idx=env_ids, skip_forward=True)
        self.init_object_positions[env_ids] = obj_pos
        
        # Randomize sponge orientation (z-axis rotation only)
        obj_orn = torch.zeros(len(env_ids), 4, device=self.device)
        obj_orn_raw = torch.randn(len(env_ids), 4, device=self.device) * 0.0
        obj_orn = obj_orn_raw / torch.norm(obj_orn_raw, dim=1, keepdim=True)
        obj_orn = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).expand(len(env_ids), -1)
        self.object.set_quat(obj_orn, envs_idx=env_ids, skip_forward=False)

        # Randomize target (container) position and orientation (full 3D)
        tgt_pos = torch.zeros(len(env_ids), 3, device=self.device)
        tgt_pos[:, 0] = torch.rand(len(env_ids), device=self.device) * 0.2 * 0.0 - 0.4
        tgt_pos[:, 1] = torch.rand(len(env_ids), device=self.device) * 0.3 * 0.0 - 0.2
        tgt_pos[:, 2] = 0.835  # On table surface
        self.target.set_pos(tgt_pos, envs_idx=env_ids)
        self.init_target_positions[env_ids] = tgt_pos
        
        # Randomize target orientation (full 3D rotation)
        tgt_orn = torch.zeros(len(env_ids), 4, device=self.device)
        # Random yaw angle (rotation around z-axis)
        yaw = torch.rand(len(env_ids), device=self.device) * 2 * np.pi * 0.0
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
        next_joint_targets = joint_targets * (self.joint_pos_max-self.joint_pos_min) / 2 + (self.joint_pos_max+self.joint_pos_min) / 2
        current_joint_pos = self.robot_entity.get_dofs_position(self.joint_indices)
        clamped_next_joint_targets = torch.clamp(next_joint_targets, current_joint_pos-0.01, current_joint_pos+0.01)
        self.robot_entity.set_dofs_position(clamped_next_joint_targets, self.joint_indices)

        # Step simulation
        self.scene.step()
        self.episode_length_buf += 1

        # Compute
        obs = self._compute_observations()
        rewards = self._compute_rewards(obs, actions)
        dones = self._compute_terminations(obs)

        # Reset done envs
        if dones.any():
            # IPython.embed()
            self.reset(torch.where(dones)[0])

        info = {"success": self._check_success(obs)}
        return obs, rewards, dones, info

    def _compute_observations(self) -> torch.Tensor:
        """Compute observations."""
        # Normalize joint positions to [-1, 1] range
        raw_pos = self.robot_entity.get_dofs_position(self.joint_indices)
        joint_pos = (raw_pos - (self.joint_pos_max+self.joint_pos_min)/2) / ((self.joint_pos_max-self.joint_pos_min)/2)
        
        joint_vel = self.robot_entity.get_dofs_velocity(self.joint_indices)
        
        ee_wrist = self.robot_entity.get_link("ee_wrist")
        ee_wrist_pos = ee_wrist.get_pos()
        ee_wrist_quat = ee_wrist.get_quat()

        ee_gripper = self.robot_entity.get_link("ee_gripper")
        ee_gripper_pos = ee_gripper.get_pos()
        ee_gripper_quat = ee_gripper.get_quat()

        ee_pos = (ee_wrist_pos + ee_gripper_pos) / 2

        object_pos = self.object.get_pos()
        object_quat = self.object.get_quat()
        object_rel_pos = object_pos - ee_pos
        target_pos = self.target.get_pos()
        target_quat = self.target.get_quat()
        
        return TensorDict({
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "object_pos": object_pos,
            "object_quat": object_quat,
            "target_pos": target_pos,
            "target_quat": target_quat,
            "object_rel_pos": object_rel_pos,
            "ee_pos": ee_pos,
        })

    def _close_gripper(self):
        for i in range(1000):
            next_joint_targets = self.robot_entity.get_dofs_position(self.joint_indices)
            next_joint_targets[:, 5] -= 0.01
            clamped_next_joint_targets = torch.clamp(next_joint_targets, self.joint_pos_min, self.joint_pos_max)
            self.robot_entity.set_dofs_position(clamped_next_joint_targets, self.joint_indices)
            self.scene.step()
        IPython.embed()


    def _compute_rewards(self, obs: Dict, actions: torch.Tensor) -> torch.Tensor:
        """Compute rewards for pick (sponge) and place (in container) task."""
        rewards_dict = {}

        # Get states
        rel_pos = obs["object_rel_pos"]  # EE to sponge
        dist_to_obj = torch.norm(rel_pos, dim=-1)
        obj_pos = obs["object_pos"]
        target_pos = obs["target_pos"]
        ee_pos = obs["ee_pos"]  # Already 3D from gripperframe

        # === PHASE 1: APPROACH + GRASP ===
        # Reward for moving toward sponge
        rewards_dict["dist_to_obj_rewards"] = -100.0 * dist_to_obj  # Move closer to sponge

        # grasp rewards
        # Gripper state (action[5]: +1 = open, -1 = closed)
        # Use -0.7 threshold to ensure gripper is sufficiently closed for grasping
        gripper_closed = obs["joint_pos"][:, 5] < -0.3                        # True if sufficiently closed
        gripper_grabbing = gripper_closed & (obs["joint_pos"][:, 5] > actions[:, 5]+0.1)
        near_sponge = dist_to_obj < self.grasp_threshold
        # rewards_dict["close_gripper_near_sponge_rewards"] = 5.0 * (near_sponge).float() * torch.clamp(-obs["joint_pos"][:, 5], min=0.0) # close
        # rewards_dict["press_gripper_near_sponge_rewards"] = 5.0 * (near_sponge).float() * torch.clamp(obs["joint_pos"][:, 5]-actions[:, 5], min=0.0) # press
        # print(f"{near_sponge=}")
        # print(f"{gripper_closed=}")
        # print(f"{gripper_grabbing=}")
        # print(f'{obs["joint_pos"]=}')
        # print(f'{actions=}')

        # CRITICAL: Reward for closing gripper when near sponge
        # This teaches the robot to actually grasp!
        # rewards += 10.0 * near_sponge.float() * gripper_closed.float()

        # Object height above table
        object_height = obj_pos[:, 2] - self.table_surface_z  # Table surface where object is placed
        object_lifted = (0.05 < object_height) & (object_height < 0.15) # 2cm above table = grasped

        # Reward for lifting the sponge (verifies successful grasp)
        rewards_dict["object_lifted_rewards"] = 10.0 * object_height
        
        # === PHASE 2: TRANSPORT + PLACE ===
        # Only reward placing if sponge is lifted (grasped)
        lifted_mask = object_lifted.float()
        
        # Distance from sponge to container (XY only for placement)
        dist_xy = torch.norm(target_pos[:, :2] - obj_pos[:, :2], dim=-1)
        rewards_dict["lift_sponge_to_container_xy_penalty"] = -10.0 * dist_xy * lifted_mask  # Move grasped sponge toward container
        
        # Z distance (sponge should be at container height)
        container_z = target_pos[:, 2]
        dist_z = torch.abs(obj_pos[:, 2] - container_z)
        rewards_dict["lift_sponge_to_container_z_penalty"] = -5.0 * dist_z * lifted_mask
        
        # Success bonus if sponge is inside container
        success = self._check_success(obs)
        rewards_dict["success_rewards"] = 10000.0 * success.float()

        # Penalties
        ## cannot continue 
        uncontinuable = (obs["object_pos"][:, 2] < 0.7) | (obs["target_pos"][:, 2] < 0.7)
        rewards_dict["uncontinuable_penalty"] = -100.0 * uncontinuable

        ## Do not move container too much.
        rewards_dict["container_shift_penalty"] = -10.0 * torch.norm(self.init_target_positions-obs["target_pos"], dim=-1)
        
        # === REGULARIZATION ===
        # Action smoothness
        action_diff = torch.norm(actions - self.prev_actions, dim=-1)
        rewards_dict["action_diff_penalty"] = -0.01 * action_diff

        # Small time penalty
        rewards_dict["lifetime_penalty"] = -0.1

        # Timeout penalty
        rewards_dict["timeout_penalty"] = -5.0 * (self.episode_length_buf >= self.max_episode_length).float()

        # print(f"{rewards_dict=}")

        rewards = torch.zeros(self.num_envs, device=self.device)
        rewards = sum([rewards for reward_key, rewards in rewards_dict.items()])

        self.prev_actions = actions.clone()
        return rewards

    def _compute_terminations(self, obs: Dict) -> torch.Tensor:
        timeout = self.episode_length_buf >= self.max_episode_length
        uncontinuable = (obs["object_pos"][:, 2] < 0.7) | (obs["target_pos"][:, 2] < 0.7)
        success = self._check_success(obs)
        return timeout | uncontinuable | success

    def _check_success(self, obs: Dict) -> torch.Tensor:
        """Check if sponge (object) is inside container (target).
        
        Container is 7.5cm x 7.5cm (0.075m x 0.075m) with 4.9cm (0.049m) height.
        Success if sponge is within container footprint AND at correct height (inside container).
        """
        object_pos = obs["object_pos"]  # (num_envs, 3)
        target_pos = obs["target_pos"]  # (num_envs, 3)
        
        # Container dimensions
        container_half_size = 0.075 / 2  # 3.75cm
        container_height = 0.049
        wall_thickness = 0.002
        
        # Sponge dimensions (triangular prism)
        sponge_height = 0.017
        sponge_half_height = sponge_height / 2
        
        # XY check: sponge must be within inner dimensions of container
        inner_half = container_half_size - wall_thickness
        
        x_inside = torch.abs(object_pos[:, 0] - target_pos[:, 0]) < inner_half
        y_inside = torch.abs(object_pos[:, 1] - target_pos[:, 1]) < inner_half
        
        # Z check: sponge should be inside container (above bottom, below top)
        # Container bottom: target_pos[:, 2] - container_height/2
        # Container top: target_pos[:, 2] + container_height/2
        sponge_bottom = object_pos[:, 2] - sponge_half_height
        sponge_top = object_pos[:, 2] + sponge_half_height
        container_bottom = target_pos[:, 2] - container_height / 2
        container_top = target_pos[:, 2] + container_height / 2
        
        z_inside = (sponge_bottom > container_bottom) & (sponge_top < container_top)
        
        return (x_inside & y_inside & z_inside)

    def close(self):
        pass