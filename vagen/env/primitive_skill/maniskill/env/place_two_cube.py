from typing import Any, Dict, Union

import numpy as np
import torch

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils import common
import time
from transforms3d.euler import euler2quat
from collections import defaultdict
@register_env("PlaceTwoCube", max_episode_steps=2e3)
class PlaceTwoCubeEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["panda", "xmate3_robotiq", "fetch"]
    agent: Union[Panda, Fetch]
    skill_config=None
    vlm_info_keys=[]
    state_keys=["red_cube_position", "green_cube_position","left_target_position","right_target_position"]

    def __init__(self, stage=0,*args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.stage=stage
        self.workspace_x=[-0.10, 0.15]
        self.workspace_y=[-0.2, 0.2]
        self.workspace_z=[0.01, 0.2]
        self.region = np.array([[-0.1, -0.1], [0, 0.1]])
        self.robot_init_qpos_noise = robot_init_qpos_noise
                
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def task_skill_indices(self):
        return {
        0 : "pick",
        1 : "place",
        2 : "push",
    }

    def instruction(self):
        return "Please place red cube at left target and green cube at right target."
        
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[1, 0.0, 0.6], target=[-0.2, 0.0, 0.2])
        return [CameraConfig("base_camera", pose, 300, 300, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1, 0.0, 0.6], [-0.2, 0.0, 0.2])
        return CameraConfig("render_camera", pose, 300,300, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self.cube_half_size = common.to_tensor([0.02] * 3)
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        
        # Create two cubes
        self.red_cube = actors.build_cube(
            self.scene, half_size=0.02, color=[1, 0, 0, 1], name="red_cube"
        )
        self.green_cube = actors.build_cube(
            self.scene, half_size=0.02, color=[0, 1, 0, 1], name="green_cube"
        )
        
        # Create two target areas
        self.goal_radius = 0.05
        self.goal_region_A = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region_A",
            add_collision=False,
            body_type="kinematic",
        )
        self.goal_region_B = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region_B",
            add_collision=False,
            body_type="kinematic",
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            # Place cubes at random positions
            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02
            region =self.region
            sampler = randomization.UniformPlacementSampler(bounds=region, batch_size=b)
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.02

            red_cube_xy = sampler.sample(radius, 100)
            green_cube_xy = sampler.sample(radius, 100, verbose=False)

            # Set initial positions for cubes
            xyz[:, :2] = red_cube_xy
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False)
            self.red_cube.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            xyz[:, :2] = green_cube_xy
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False)
            self.green_cube.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            # Set fixed positions for target areas
            left_target_position = torch.tensor([0.08, -0.1, 0.0])
            right_target_position = torch.tensor([0.08, 0.1, 0.0])
        
            self.goal_region_A.set_pose(Pose.create_from_pq(
                p=left_target_position,
                q=euler2quat(0, np.pi / 2, 0),
            ))
            self.goal_region_B.set_pose(Pose.create_from_pq(
                p=right_target_position,
                q=euler2quat(0, np.pi / 2, 0),
            ))

    def is_cube_in_goal(self, cube_position, goal_position):
        distance = torch.norm(cube_position[..., :2] - goal_position[...,:2], dim=-1)
        return distance <= self.goal_radius

    def _get_obs_extra(self, info: Dict):
        assert "state" in self.obs_mode
        obs = dict(
            red_cube_position=info["red_cube_position"],
            green_cube_position=info["green_cube_position"],
            is_red_cube_grasped=info["is_red_cube_grasped"],
            is_green_cube_grasped=info["is_green_cube_grasped"],
            left_target_position=self.goal_region_A.pose.p,
            right_target_position=self.goal_region_B.pose.p,
        )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return torch.zeros_like(info["success"],dtype=torch.float32,device=self.device)
        

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 80.0
    
    
    def task_fail(self, info: Dict):
        # if cube position is out of workspace return true
        for cube in ["red_cube", "green_cube"]:
            if info[f"{cube}_position"][0] < self.workspace_x[0] or info[f"{cube}_position"][0] > self.workspace_x[1]:
                return True
            if info[f"{cube}_position"][1] < self.workspace_y[0] or info[f"{cube}_position"][1] > self.workspace_y[1]:
                return True
            if info[f"{cube}_position"][2] < 0:
                return True
        return False
    
    def evaluate(self):
        pos_A = self.red_cube.pose.p
        pos_B = self.green_cube.pose.p

        is_red_cube_grasped = self.agent.is_grasping(self.red_cube)
        is_green_cube_grasped = self.agent.is_grasping(self.green_cube)

        is_red_cube_in_goal = self.is_cube_in_goal(pos_A, self.goal_region_A.pose.p)
        is_green_cube_in_goal = self.is_cube_in_goal(pos_B, self.goal_region_B.pose.p)

        stage0_success = is_red_cube_grasped
        stage1_success = is_red_cube_in_goal & (~is_red_cube_grasped)
        stage2_success = is_red_cube_in_goal & is_green_cube_grasped
        stage3_success = is_red_cube_in_goal & is_green_cube_in_goal & (~is_green_cube_grasped) & (~is_red_cube_grasped)

        
        
        success = stage3_success
        


        info = {
            "left_target_position":self.goal_region_A.pose.p,
            "right_target_position":self.goal_region_B.pose.p,
            "is_red_cube_grasped": is_red_cube_grasped,
            "is_green_cube_grasped": is_green_cube_grasped,
            "red_cube_position": pos_A,
            "green_cube_position": pos_B,
            "is_red_cube_in_goal": is_red_cube_in_goal,
            "is_green_cube_in_goal": is_green_cube_in_goal,
            "stage0_success": stage0_success.bool(),
            "stage1_success": stage1_success.bool(),
            "stage2_success": stage2_success.bool(),
            "success": success.bool(),
        }
        return info

    def skill_reward(self, prev_info, cur_info, action, **kwargs):
        return 0.0

            
            

    def reset(self, **kwargs):
        # reset reward components to 0
        return super().reset(**kwargs)
    
    
