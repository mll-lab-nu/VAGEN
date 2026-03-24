import gymnasium as gym
import torch
import mani_skill.envs
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
from .skill_wrapper import SkillGymWrapper
import numpy as np
import os

# Register custom ManiSkill environments
import vagen.envs.maniskill.maniskill.env  # noqa: F401


def build_env(env_id, control_mode="pd_ee_delta_pose", stage=0, record_dir=None):
    env_kwargs = dict(
        obs_mode="state",
        control_mode=control_mode,
        render_mode="rgb_array",
        sim_backend="cpu",
        render_backend="gpu",
    )
    env = gym.make(env_id, num_envs=1, enable_shadow=True, stage=stage, **env_kwargs)
    env = CPUGymWrapper(env)
    env = SkillGymWrapper(
        env,
        skill_indices=env.task_skill_indices,
        record_dir=os.path.join(record_dir, env_id) if record_dir is not None else None,
        record_video=record_dir is not None,
        max_episode_steps=10,
        max_steps_per_video=1,
        controll_mode=control_mode,
    )
    env.is_params_scaled = False
    return env


def handle_info(info, state_keys, mask_success=False, env=None):
    obj_positions = {}
    other_info = {}

    pop_list = ['is_success', 'num_timesteps', 'elapsed_steps', 'skill_success', 'reward_components']
    for k in pop_list:
        info.pop(k, None)

    for k, v in info.items():
        if k.endswith('_position'):
            if k in state_keys:
                obj_positions[k] = tuple(np.round(v * 1000, 0).astype(int).tolist())
        elif k.endswith('_value'):
            other_info[k] = np.round(v * 1000, 0).astype(int).item()
        elif k.endswith('_size'):
            other_info[k] = tuple(np.round(v * 1000, 0).astype(int).tolist())
        else:
            if isinstance(v, np.ndarray):
                if v.ndim == 0:
                    other_info[k] = v.item()
                else:
                    other_info[k] = tuple(v.flatten().tolist())
            else:
                other_info[k] = v

    if mask_success and env is not None:
        final_info = {}
        for k in env.vlm_info_keys:
            if k in other_info:
                final_info[k] = other_info[k]
        if not final_info:
            final_info = "No other information needed"
    else:
        final_info = dict(other_info)

    return {
        'obj_positions': obj_positions,
        'other_info': final_info,
    }


def get_workspace_limits(env):
    x_workspace = tuple(np.round(np.array(env.workspace_x) * 1000, 0).astype(int))
    y_workspace = tuple(np.round(np.array(env.workspace_y) * 1000, 0).astype(int))
    z_workspace = tuple(np.round(np.array(env.workspace_z) * 1000, 0).astype(int))
    return x_workspace, y_workspace, z_workspace
