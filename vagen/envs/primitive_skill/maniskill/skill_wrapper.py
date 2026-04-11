import gymnasium as gym
from gymnasium.spaces import Box
import sapien.physx as physx
#from rosetta.maniskill.utils.primitive_skills_pose import PrimitiveSkillDelta
from .primitive_skills_cpu import PrimitiveSkillDelta,PrimitiveSkillAbs
import numpy as np
from mani_skill.utils import common
import sys
from mani_skill.utils.visualization.misc import (
    images_to_video,
    put_info_on_image,
    tile_images,
)
from mani_skill.utils import common, gym_utils
from copy import deepcopy
import os
import json



class SkillGymWrapper(gym.Wrapper):
    """This wrapper wraps any maniskill CPUEnv, takes the input of primitve skill and its parameters, 
    execute multiple steps in the original environment and returns the output as a single step in the new environment."""

    """currently only support panda robot and pd_ee_delta_pose controller"""
    """
    skill_indices={
                0 : "pick",
                1 : "place",
            }
            
    # to use skillwrapper, please define skill_reward function for original maniskill envrionment
    """
    def __init__(self, env: gym.Env,
                 skill_indices,
                 record_dir=None,
                 max_episode_steps=None,
                 max_steps_per_video=None,
                 video_fps: int = 30,
                 control_mode="pd_ee_delta_pose",
                 **kwargs):
        super().__init__(env)
        if env.unwrapped.skill_config is None:
            skill_config={}
        else:
            skill_config=env.unwrapped.skill_config
            
        # add skill_config to **kwargs
        kwargs.update(skill_config)
        # currently support both abs ee pose and delta ee pose
        assert control_mode in ["pd_ee_delta_pose", "pd_ee_pose"], "Only support pd_ee_delta_pose and pd_ee_pose"
        self.use_abs_skill = control_mode == "pd_ee_pose"
        if self.use_abs_skill:
            self.primitive_skill = PrimitiveSkillAbs(skill_indices,**kwargs)
        else:
            self.primitive_skill = PrimitiveSkillDelta(skill_indices,**kwargs)
        self.cur_info = dict()
        self.prev_info = dict()
        low = -np.ones(self.primitive_skill.n_skills+self.primitive_skill.max_num_params)
        high = np.ones(self.primitive_skill.n_skills+self.primitive_skill.max_num_params)
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.record_dir = record_dir

        self.max_episode_steps = max_episode_steps
        self.elapsed_step = 0
        self.max_steps_per_video=max_steps_per_video
        self.video_fps=video_fps
        self.render_images=[]
        self.num_skills=self.primitive_skill.n_skills
        self._video_id=0
        self._video_steps=0
        self.episode_id=0
        self.is_params_scaled = True
        self.traj = []
        
        # obs_shape = self.env.unwrapped.observation_space.shape
        # obs_shape = (obs_shape[1]+1,)
        # self.observation_space = Box(float('-inf'), float('inf'), shape=obs_shape)


        if self.record_dir is not None:
            os.makedirs(self.record_dir, exist_ok=True)
            os.makedirs(os.path.join(self.record_dir, f"episode_{self.episode_id}"), exist_ok=True)

    def scale_params(self, params):
        """
        Scales normalized parameter ([-1, 1]) to appropriate raw values
        """
        scaled_params = np.copy(params)
        scaled_params[0] = ( ((params[0] + 1) / 2 ) * (self.env.unwrapped.workspace_x[1] - self.env.unwrapped.workspace_x[0]) ) + self.env.unwrapped.workspace_x[0]
        scaled_params[1] = ( ((params[1] + 1) / 2 ) * (self.env.unwrapped.workspace_y[1] - self.env.unwrapped.workspace_y[0]) ) + self.env.unwrapped.workspace_y[0]
        scaled_params[2] = ( ((params[2] + 1) / 2 ) * (self.env.unwrapped.workspace_z[1] - self.env.unwrapped.workspace_z[0]) ) + self.env.unwrapped.workspace_z[0]
        return scaled_params
    
    def norm_params(self, params):
        """
        Normalized parameter to ([-1, 1]
        """
        norm_params = np.copy(params)
        norm_params[0] = 2*(params[0] - self.env.unwrapped.workspace_x[0])/(self.env.unwrapped.workspace_x[1] - self.env.unwrapped.workspace_x[0]) - 1
        norm_params[1] = 2*(params[1] - self.env.unwrapped.workspace_y[0])/(self.env.unwrapped.workspace_y[1] - self.env.unwrapped.workspace_y[0]) - 1
        norm_params[2] = 2*(params[2] - self.env.unwrapped.workspace_z[0])/(self.env.unwrapped.workspace_z[1] - self.env.unwrapped.workspace_z[0]) - 1
        return norm_params
    
    def step(self, action,**kwargs):
        self.elapsed_step+=1
        if self.is_params_scaled:
            action[self.num_skills:self.num_skills+3] = self.scale_params(action[self.num_skills:self.num_skills+3])
            action[self.num_skills+3:self.num_skills+6] = self.scale_params(action[self.num_skills+3:self.num_skills+6])

        done, skill_done, skill_success = False, False, False
        self.prev_info = deepcopy(self.cur_info)
        step=0
        reward=0
        # import pdb; pdb.set_trace()
        # while not done and not skill_done:

        while not skill_done:
            eef_state={
                "robot0_eef_pos":common.unbatch(common.to_numpy(self.env.unwrapped.agent.tcp.pose.p)),
                "robot0_eef_quat":common.unbatch(common.to_numpy(self.env.unwrapped.agent.tcp.pose.q)),
            }
            action_ll, skill_done, skill_success = self.primitive_skill.get_action(action, eef_state)
            if self.use_abs_skill:
                action_ll[:3]-=self.env.unwrapped.agent.robot.pose.p[0].cpu().numpy()
            obs, reward, terminated, truncated, info = super().step(action_ll)
            done = terminated or truncated
            if self.record_dir is not None:
                self.record_image()
            step+=1

        if self.record_dir is not None:
            self._video_steps+=1
            if (
                self.max_steps_per_video is not None
                and self._video_steps>= self.max_steps_per_video
            ):
                self.flush_video()
                
        # print("skill steps: ", step)
        info['num_timesteps']=step
        info['skill_success']=skill_success
        info['is_success']=info["success"]
        self.cur_info = info
        reward=self.env.unwrapped.skill_reward(prev_info=self.prev_info, cur_info=self.cur_info, action=action,reward=reward,**kwargs)
        if self.max_episode_steps is not None and self.elapsed_step>=self.max_episode_steps:
            truncated = True
        info["reward_components"]={}
        rews=0.0
        if isinstance(reward, dict):
            for k,v in reward.items():
                info["reward_components"][k]=v
                rews+=v
            info["reward_components"]['reward']=rews
        else:
            rews+=reward
            info["reward_components"]['reward']=reward
        
        if self.record_dir is not None:
            action = {
                "action": self.primitive_skill.skill_indices[action[:self.primitive_skill.n_skills].argmax()],
                "params": action[self.primitive_skill.n_skills:].tolist(),
            }
            self.traj.append(action)
            self.traj.append(info)
        # obs = np.append(obs, self.elapsed_step)
        
        new_obs = np.array(self.env.unwrapped.get_obs())[0]

        return new_obs, rews, terminated, truncated, info

    def reset(self,seed=None, options=None):
        self.primitive_skill.reset()
        self.env.unwrapped.cur_stage = 0
        obs, info = super().reset(seed=seed, options=options)
        action_ll=np.zeros(7)
        obs, reward, terminated, truncated, info = super().step(action_ll)
        
        # obs = np.append(obs, self.elapsed_step)
        info['num_timesteps']=0
        info['skill_success']=False
        info['is_success']=info["success"]
        if info["success"]:
            print("happen to be successful in the beginning")
        self.elapsed_step=0
        self.cur_info=info
        self.prev_info=info
        # TODO HACK

        # record the trajectory
        if self.record_dir is not None:
            if len(self.traj) > 0:
                self.traj = convert_arrays_to_lists(self.traj)
                try:
                    with open(os.path.join(self.record_dir, f"episode_{self.episode_id}", "trajectory.json"), "w") as f:
                        json.dump(self.traj, f, indent=4) # HACK to handle np.float32 objects
                except Exception as e:
                    print(f"Failed to save trajectory: {e}")
                    print(self.traj)
                self.episode_id += 1
                os.makedirs(os.path.join(self.record_dir, f"episode_{self.episode_id}"), exist_ok=True)
            self.traj = [info]
        return obs, info
    
    def record_image(self):
        img=super().render()
        self.render_images.append(img)
            
    def flush_video(
        self,
        name=None,
        suffix="",
        verbose=False,
        ignore_empty_transition=True,
        save: bool = True,
    ):
        """
        Flush a video of the recorded episode(s) anb by default saves it to disk

        Arguments:
            name (str): name of the video file. If None, it will be named with the episode id.
            suffix (str): suffix to add to the video file name
            verbose (bool): whether to print out information about the flushed video
            ignore_empty_transition (bool): whether to ignore trajectories that did not have any actions
            save (bool): whether to save the video to disk
        """
        if len(self.render_images) == 0:
            return
        if ignore_empty_transition and len(self.render_images) == 1:
            return
        if save:
            self._video_id += 1
            if name is None:
                video_name = "{}".format(self._video_id)
                if suffix:
                    video_name += "_" + suffix
            else:
                video_name = name
            images_to_video(
                self.render_images,
                str(os.path.join(self.record_dir, f"episode_{self.episode_id}", "video")),
                video_name=video_name,
                fps=self.video_fps,
                verbose=verbose,
            )
        self._video_steps = 0
        self.render_images = []


def convert_arrays_to_lists(data):
    """
    Recursively traverse a nested dictionary (or list) and convert numpy arrays to lists.
    
    Args:
        data (dict, list, any): The input nested dictionary, list, or value.
        
    Returns:
        The transformed structure with numpy arrays converted to lists.
    """
    if isinstance(data, dict):
        return {key: convert_arrays_to_lists(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_arrays_to_lists(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.float32):
        return float(data)
    else:
        return data