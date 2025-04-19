import gymnasium as gym
import torch
import mani_skill.envs
from tqdm.notebook import tqdm
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
from vagen.env.manipulation.maniskill.skill_wrapper import SkillGymWrapper
import numpy as np
import os



def build_env(env_id, control_mode="pd_ee_pose", stage=0, record_dir='./test'):
    env_kwargs = dict(obs_mode="state", control_mode=control_mode, render_mode="rgb_array", sim_backend="cpu")
    env = gym.make(env_id, num_envs=1, enable_shadow=True, stage=stage, **env_kwargs)
    env = CPUGymWrapper(env)
    env = SkillGymWrapper(env,
                          skill_indices=env.task_skill_indices,
                          record_dir=os.path.join(record_dir, env_id),
                          record_video=True,
                          max_episode_steps=3,
                          max_steps_per_video=1,
                          controll_mode=control_mode,
                          )
    env.is_params_scaled = False
    return env


def handel_info(info):
    obj_positions={}
    other_info={}
    info.pop('is_success')
    info.pop('num_timesteps')
    info.pop('elapsed_steps')
    for k,v in info.items():
        if k.endswith('_pos'):
            # convert to cm round to 2 decimal places
            obj_positions[k] = np.round(v*1000, 0).astype(int)
        else:
            other_info[k] = v
    return {
        'obj_positions': obj_positions,
        'other_info': other_info
    }