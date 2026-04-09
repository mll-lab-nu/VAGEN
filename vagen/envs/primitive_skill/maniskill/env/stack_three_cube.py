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
from collections import defaultdict

@register_env("StackThreeCube", max_episode_steps=3e3)
class StackThreeCubeEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["panda", "xmate3_robotiq", "fetch"]
    agent: Union[Panda,  Fetch]
    skill_config=None
    vlm_info_keys=['cube_size']
    state_keys=["red_cube_position", "green_cube_position", "purple_cube_position"]

    def __init__(self, stage=0,*args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.stage=stage
        self.cur_stage = 0
        self.cube_size = 0.04

        self.workspace_x=[-0.10, 0.15]
        self.workspace_y=[-0.2, 0.2]
        self.workspace_z=[0.01, 0.2]
        
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.reward_components = ["success","afford"]
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def task_skill_indices(self):
        return {
        0 : "pick",
        1 : "place",
        2 : "push",
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
        return "Please stack the red cube on top of the green cube, and then stack purple cube on top of the red cube."
    
    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.red_cube = actors.build_cube(
            self.scene, half_size=self.cube_size/2, color=[1, 0, 0, 1], name="red_cube"
        )
        self.green_cube = actors.build_cube(
            self.scene, half_size=self.cube_size/2, color=[0, 1, 0, 1], name="green_cube"
        )
        self.purple_cube = actors.build_cube(
            self.scene, half_size=self.cube_size/2, color=[1, 0, 1, 1], name="purple_cube"
        )
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02
            region =[[self.workspace_x[0],self.workspace_y[0]],[self.workspace_x[1],self.workspace_y[1]]]
            sampler = randomization.UniformPlacementSampler(bounds=region, batch_size=b)
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.02
            
            red_cube_xy = sampler.sample(radius, 100)
            green_cube_xy = sampler.sample(radius, 100, verbose=False)
            purple_cube_xy = sampler.sample(radius, 100, verbose=False)

            xyz[:, :2] = red_cube_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.red_cube.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            xyz[:, :2] = green_cube_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.green_cube.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            xyz[:, :2] = purple_cube_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.purple_cube.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            self.object_list = {"red_cube": self.red_cube, 
                                "green_cube": self.green_cube,
                                "purple_cube": self.purple_cube}
    
    def _get_obs_info(self):
        info = {}
        for name in self.object_list:
            info[f"is_{name}_grasped"] = self.agent.is_grasping(self.object_list[name])[0]
            info[f"{name}_position"] = self.object_list[name].pose.p[0]
        info["cube_size"]=torch.ones_like(info["red_cube_position"])*0.04
        info["gripper_position"] = self.agent.tcp.pose.p[0]
        return info

    def evaluate(self):
        info= self._get_obs_info()
        
        def stage0_success(info):
            return info["is_red_cube_grasped"]
        
        def stage1_success(info):
            red_not_grasped = ~info["is_red_cube_grasped"]
            red_on_green = (torch.linalg.norm(info["red_cube_position"][:2] - info["green_cube_position"][:2]) < self.cube_size/2) and (info["red_cube_position"][2] > (info["green_cube_position"][2] + self.cube_size/2))
            return (red_on_green and red_not_grasped)
        
        def stage2_success(info):
            return info["is_purple_cube_grasped"]
        
        def stage3_success(info):
            purple_not_grasped = ~info["is_purple_cube_grasped"]
            purple_on_red = (torch.linalg.norm(info["purple_cube_position"][:2] - info["red_cube_position"][:2]) < self.cube_size/2) and (info["purple_cube_position"][2] > (info["red_cube_position"][2] + self.cube_size/2))
            
            return purple_on_red and purple_not_grasped and stage1_success(info)
        
        info["stage0_success"] = stage0_success(info)
        info["stage1_success"] = stage1_success(info)
        info["stage2_success"] = stage2_success(info)
        info["success"] = stage3_success(info)
        
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

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 80.0
    
    def skill_reward(self, prev_info, cur_info, action,**kwargs):
        return 0.0



    def reset(self, **kwargs):
        self.cur_stage = 0
        return super().reset(**kwargs)