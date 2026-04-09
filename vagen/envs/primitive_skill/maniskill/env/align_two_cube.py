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
@register_env("AlignTwoCube", max_episode_steps=2e3)
class AlignTwoCubeEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["panda", "xmate3_robotiq", "fetch"]
    agent: Union[Panda,  Fetch]
    skill_config=None
    vlm_info_keys=[]
    state_keys=["red_cube_position", "green_cube_position"]

    def __init__(self, stage=0,*args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.stage=stage
        self.workspace_x=[-0.10, 0.15]
        self.workspace_y=[-0.2, 0.2]
        self.workspace_z=[0.01, 0.2]
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.region_1 = np.array([[0.1, -0.2], [0.15, 0.1]])
        self.region_2 = np.array([[0.1, 0.1], [0.15, 0.2]])
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def task_skill_indices(self):
        return {
        0 : "pick",
        1 : "place",
        2: "push",
    }
    
    def instruction(self):
        return "Please align the cubes in the y-axis, which means the x-coordinates of both cubes should be 0 (+-10mm)"
    
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
        self.red_cube = actors.build_cube(
            self.scene, half_size=0.02, color=[1, 0, 0, 1], name="red_cube"
        )
        self.green_cube = actors.build_cube(
            self.scene, half_size=0.02, color=[0, 1, 0, 1], name="green_cube"
        )

    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02
            sampler_1 = randomization.UniformPlacementSampler(bounds=self.region_1, batch_size=b)
            sampler_2 = randomization.UniformPlacementSampler(bounds=self.region_2, batch_size=b)
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.02
            
            red_cube_xy = sampler_1.sample(radius, 100)
            green_cube_xy = sampler_2.sample(radius, 100, verbose=False)
            
            
            

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

            
    
    def evaluate(self):
        pos_A = self.red_cube.pose.p
        pos_B = self.green_cube.pose.p
       

        is_red_cube_grasped = self.agent.is_grasping(self.red_cube)
        is_green_cube_grasped = self.agent.is_grasping(self.green_cube)
        
        x_tolerance = 0.01

        is_red_cube_at_x0 = (torch.abs(pos_A[..., 0]) <= x_tolerance) & (torch.abs(pos_A[..., 2]) <= 0.05)
        is_green_cube_at_x0 = (torch.abs(pos_B[..., 0]) <= x_tolerance) & (torch.abs(pos_B[..., 2]) <= 0.05)

       
        success = is_red_cube_at_x0 & is_green_cube_at_x0

        info = {
            "is_red_cube_grasped": is_red_cube_grasped,
            "is_red_cube_at_x0": is_red_cube_at_x0,
            "is_green_cube_grasped": is_green_cube_grasped,
            "is_green_cube_at_x0": is_green_cube_at_x0,
            "red_cube_position": pos_A,
            "green_cube_position": pos_B,
            "stage_0_success": is_red_cube_at_x0,
            "stage_1_success": is_green_cube_at_x0,
            "success": success.bool(),
        }
        return info


    def _get_obs_extra(self, info: Dict):
        assert "state" in self.obs_mode
        obs = dict(
            red_cube_position=info["red_cube_position"],
            green_cube_position=info["green_cube_position"],
            is_red_cube_at_x0=info["is_red_cube_at_x0"],
            is_green_cube_at_x0=info["is_green_cube_at_x0"],
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
        for cube in ["red_cube", "green_cube", "cubeC"]:
            if info[f"{cube}_pos"][0] < self.workspace_x[0] or info[f"{cube}_pos"][0] > self.workspace_x[1]:
                return True
            if info[f"{cube}_pos"][1] < self.workspace_y[0] or info[f"{cube}_pos"][1] > self.workspace_y[1]:
                return True
            if info[f"{cube}_pos"][2] < 0:
                return True
        return False
        
    def skill_reward(self, prev_info, cur_info, action, **kwargs):
        """
        prev_info: dict containing previous evaluation info
        cur_info: dict containing current evaluation info
        action: np.array (with action indices and target positions)
        """
        return 0.0
            
            

    def reset(self, **kwargs):
        # reset reward components to 0
        return super().reset(**kwargs)
    
    

