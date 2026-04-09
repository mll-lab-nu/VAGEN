from typing import Any, Dict, Union, List
import numpy as np
import torch
import sapien.core as sapien

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import articulations, actors
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.utils.geometry.geometry import transform_points
from collections import defaultdict

@register_env("PutAppleInDrawer", max_episode_steps=3000)
class PutAppleInDrawerEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda", "xmate3_robotiq", "fetch"]
    agent: Union[Panda, Fetch]
    vlm_info_keys=["drawer_open_value"]
    state_keys=["apple_position", "drawer_position"]
    # Asset configuration and constants
    DRAWER_ASSET_ID = "partnet_mobility_cabinet"
    handle_types = ["prismatic"]  # We are interested in prismatic joints (drawers)
    min_open_frac = 0.6  # Fraction of the drawer's range to be open
    skill_config={
        "home_pos": (0.0,0.0,0.65),
    }
    model_id = "ycb:013_apple"
    def __init__(self, stage=0, *args, robot_uids="panda", robot_init_qpos_noise=0.02, drawer_id=1016, **kwargs):
        self.stage = stage
        self.cur_stage = 0
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.workspace_x = [-0.5, -0.1]  # Adjusted workspace X range
        self.workspace_y = [-0.6, 0.2]
        self.workspace_z = [0.01, 0.65]
        self.drawer_id = drawer_id
        self.reward_components = ["success", "afford"]
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def task_skill_indices(self):
        return {
            0: "pick",
            1: "place",
            2: "push",
        }

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[1, 0.0, 0.6], target=[-0.2, 0.0, 0.2])
        return [CameraConfig("base_camera", pose, 300, 300, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1, 0.0, 0.6], [-0.2, 0.0, 0.2])
        return CameraConfig("render_camera", pose, 300,300, 1, 0.01, 100)

    def instruction(self):
        return "Please put the apple in the drawer and close the drawer."
    
    def _load_scene(self, options: dict):
        # Build ground plane
        self.ground = build_ground(self.scene)

        # Load the drawer
        self._load_drawer()

        # Load YCB objects
        builder = actors.get_actor_builder(self.scene, id=self.model_id)
        self.apple = builder.build_dynamic(name="apple")
        # Note: Apple's bounding box will be used for calculations

    def _load_drawer(self):
        cabinet_builder = articulations.get_articulation_builder(
            self.scene, f"partnet-mobility:{self.drawer_id}"
        )
        cos_theta_over_2 = np.cos(-np.pi / 4)
        sin_theta_over_2 = np.sin(-np.pi / 4)

        # Adjusted drawer position and orientation
        cabinet_builder.initial_pose = sapien.Pose(
            p=[-0.5, -1.0, 0.5],  
            q=[cos_theta_over_2, 0, 0, sin_theta_over_2]  
        )
        cabinet = cabinet_builder.build(name=f"{self.drawer_id}-drawer")
        for joint in cabinet.get_joints():
            if joint.type[0] in self.handle_types and joint.active_index is not None:
                self.drawer = cabinet
                self.drawer_joint = joint
                print(f"Loaded drawer model_id: {self.drawer_id}")
                print(f"Found drawer joint: {joint.get_name()}")
                break

    def _after_reconfigure(self, options):
        qlimits = self.drawer.get_qlimits()  
        num_envs = qlimits.shape[0]
        env_idx = torch.arange(num_envs, device=self.device)
        qmin = qlimits[env_idx, self.drawer_joint.active_index, 0]
        qmax = qlimits[env_idx, self.drawer_joint.active_index, 1]
        self.drawer_open_positions = qmin + (qmax - qmin) * self.min_open_frac

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            # Get apple's bounding box bounds correctly
            apple_bounds = self.apple.get_first_collision_mesh().bounding_box.bounds
            apple_height = (apple_bounds[1][2] - apple_bounds[0][2]) / 2  # Calculate height from bounds

            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, 2] = apple_height  # Use apple's actual height
            sampler = randomization.UniformPlacementSampler(bounds=[
                [self.workspace_x[0]+0.35, self.workspace_y[0]+0.7],
                [self.workspace_x[1], self.workspace_y[1]]
            ], batch_size=b)
            apple_xy = sampler.sample(0.04, 100)  # Adjust radius based on apple size if needed
            xyz[:, :2] = apple_xy
            self.apple.set_pose(Pose.create_from_pq(p=xyz.clone()))

            # Reset drawer position
            qpos = self.drawer.get_qpos()
            drawer_active_index=self.drawer_joint.active_index[env_idx]
            qpos[env_idx, drawer_active_index] = self.drawer_open_positions[env_idx]
            self.drawer.set_qpos(qpos[env_idx].clone())

            # Initialize robot's joint positions
            qpos = np.array([0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, np.pi / 4, 0.04, 0.04])
            self.agent.robot.set_pose(sapien.Pose([-0.5, 0, 0]))
            self.agent.robot.set_qpos(qpos)
            self.object_list = {"apple": self.apple}
        
    def compute_normalized_dense_reward(self, obs, action, info):
        # Normalize dense reward
        max_possible_reward = 30.0
        return self.compute_dense_reward(obs, action, info) / max_possible_reward
    
    def _get_obs_info(self):
        info= {}
        for name in self.object_list:
            info[f"is_{name}_grasped"] = self.agent.is_grasping(self.object_list[name])[0]
            info[f"{name}_position"] = self.object_list[name].pose.p[0]
        
        info["stage"] = self.cur_stage
        info["gripper_position"] = self.agent.tcp.pose.p[0]

        drawer_link = self.drawer_joint.get_child_link()
        info["drawer_handle_position"] = drawer_link.pose.p.to(self.device)[0] + np.array([0, 0.37, -0.3])
        info["drawer_position"] = drawer_link.pose.p.to(self.device)[0] + np.array([0, 0.2, -0.3])

        info["drawer_open_value"] = (info["drawer_handle_position"][1] - np.array([0.5, -0.63,  0.2]))[1]
        
     
        return info

    def evaluate(self):
        info = self._get_obs_info()

        def stage0_success(info):
            return info[f"is_apple_grasped"]
        
        def stage1_success(info):
            # Check if apple is in drawer
            abs_diff_xy = torch.abs(info["apple_position"][:2] - info["drawer_position"][:2])

            # Check if x and y differences are within tolerance
            within_xy = (abs_diff_xy <= 0.1).all(dim=-1)

            # Check if z position is within valid range
            within_z = (info["apple_position"][2] >= 0.1) & (info["apple_position"][2] <= 0.4)

            # Combine conditions
            is_apple_in_drawer = within_xy & within_z
            return is_apple_in_drawer & (~info["is_apple_grasped"])

        def stage2_success(info):
            # Check if apple is in drawer
            abs_diff_xy = torch.abs(info["apple_position"][:2] - info["drawer_position"][:2])

            # Check if x and y differences are within tolerance
            within_xy = (abs_diff_xy <= 0.1).all(dim=-1)

            # Check if z position is within valid range
            within_z = (info["apple_position"][2] >= 0.1) & (info["apple_position"][2] <= 0.4)

            # Combine conditions
            is_apple_in_drawer = within_xy & within_z
            
            # Check if drawer is closed
            is_drawer_closed = (info["drawer_open_value"] <= 0.1)
            return is_apple_in_drawer & is_drawer_closed & (~info["is_apple_grasped"])

        info["stage0_success"] = stage0_success(info)
        info["stage1_success"] = stage1_success(info)
        info["success"] = stage2_success(info)

        return info

    def get_obs(self, info: Dict = None):
        if info is None:
            info = self.get_info()
        obs = []
        for name in self.object_list:
            obs += info[f"{name}_position"].flatten().tolist()

        for name in self.object_list:
             obs += info[f"is_{name}_grasped"].flatten().tolist()
        
        obs += [self.cur_stage]
        return torch.tensor([obs], device = self.device, dtype = torch.float32)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return torch.zeros_like(info["success"],dtype=torch.float32,device=self.device)
        

    def task_fail(self, info: Dict):
        return False

    def skill_reward(self, prev_info, cur_info, action, **kwargs):
        return 0.0

    def reset(self, **kwargs):
        # reset reward components to 0
        return super().reset(**kwargs)