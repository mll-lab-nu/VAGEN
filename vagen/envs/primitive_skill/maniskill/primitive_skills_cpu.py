"""
Basic, hardcoded parameterized primitive motions

Suggested wrist angle range [-0.5pi, 0.5pi]

NOTE Currently, only move_to_pos works with option return_all_states = True
"""
import numpy as np
import math
import numba
ENABLE_NUMBA = True
CACHE_NUMBA = True

PI = np.pi
EPS = np.finfo(float).eps * 4.0

import numpy as np
import math
import numba
# Constants
EPS = np.finfo(float).eps

# Axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# Map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    "sxyz": (0, 0, 0, 0),
    "sxyx": (0, 0, 1, 0),
    "sxzy": (0, 1, 0, 0),
    "sxzx": (0, 1, 1, 0),
    "syzx": (1, 0, 0, 0),
    "syzy": (1, 0, 1, 0),
    "syxz": (1, 1, 0, 0),
    "syxy": (1, 1, 1, 0),
    "szxy": (2, 0, 0, 0),
    "szxz": (2, 0, 1, 0),
    "szyx": (2, 1, 0, 0),
    "szyz": (2, 1, 1, 0),
    "rzyx": (0, 0, 0, 1),
    "rxyx": (0, 0, 1, 1),
    "ryzx": (0, 1, 0, 1),
    "rxzx": (0, 1, 1, 1),
    "rxzy": (1, 0, 0, 1),
    "ryzy": (1, 0, 1, 1),
    "rzxy": (1, 1, 0, 1),
    "ryxy": (1, 1, 1, 1),
    "ryxz": (2, 0, 0, 1),
    "rzxz": (2, 0, 1, 1),
    "rxyz": (2, 1, 0, 1),
    "rzyz": (2, 1, 1, 1),
}

# Function to enable Numba just-in-time compilation
def jit_decorator(func):
    return numba.jit(nopython=ENABLE_NUMBA, cache=CACHE_NUMBA)(func) if True else func

def vec(values):
    """
    Converts value tuple into a numpy vector.

    Args:
        values (n-array): a tuple of numbers

    Returns:
        np.array: vector of given values
    """
    return np.array(values, dtype=np.float32)

def _wrap_to_pi(angles):
    """
    Normalize angle in radians to range [-pi, pi]
    """
    pi2 = 2 * np.pi
    result = np.fmod(np.fmod(angles, pi2) + pi2, pi2)
    if result > np.pi:
        result = result - pi2
    if result < -np.pi:
        result = result + pi2
    return result

def _wrap_to_2pi(angle):
    """
    Normalize angle in radians to range [0, 2pi]
    """
    return angle % (2 * np.pi)

def quat2yaw(quat):
    """
    Given quaternion "xyzw", returns yaw [rad]
    """
    quat = np.array(quat)
    quat = quat / (np.linalg.norm(quat)+1e-6)
    x, y, z, w = quat
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

def mat2euler(rmat, axes="sxyz"):
    """
    Converts given rotation matrix to Euler angles in radians.

    Args:
        rmat (np.array): 3x3 rotation matrix
        axes (str): One of 24 axis sequences as string or encoded tuple (see top of this module)

    Returns:
        np.array: (r,p,y) converted Euler angles in radians vec3 float
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    M = np.array(rmat, dtype=np.float32, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        if sy > EPS:
            ax = math.atan2(M[i, j], M[i, k])
            ay = math.atan2(sy, M[i, i])
            az = math.atan2(M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(sy, M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        if cy > EPS:
            ax = math.atan2(M[k, j], M[k, k])
            ay = math.atan2(-M[k, i], cy)
            az = math.atan2(M[j, i], M[i, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(-M[k, i], cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return vec((ax, ay, az))

@jit_decorator
def quat2mat(quaternion):
    """
    Converts given quaternion to matrix.

    Args:
        quaternion (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: 3x3 rotation matrix
    """
    # Awkward semantics for use with numba
    inds = np.array([3, 0, 1, 2])
    q = np.asarray(quaternion).copy().astype(np.float32)[inds]

    n = np.dot(q, q)
    if n < EPS:
        return np.identity(3)
    q *= np.sqrt(2.0 / (n+1e-6))
    q2 = np.outer(q, q)
    return np.array(
        [
            [1.0 - q2[2, 2] - q2[3, 3], q2[1, 2] - q2[3, 0], q2[1, 3] + q2[2, 0]],
            [q2[1, 2] + q2[3, 0], 1.0 - q2[1, 1] - q2[3, 3], q2[2, 3] - q2[1, 0]],
            [q2[1, 3] - q2[2, 0], q2[2, 3] + q2[1, 0], 1.0 - q2[1, 1] - q2[2, 2]],
        ]
    )

# Modified _quat2euler to ensure proper handling and added debug statements
def _quat2euler(quat):
    quat = np.array(quat)
    if quat.size != 4:
        raise ValueError("Quaternion must be of size 4")
    ##print("Quaternion:", quat)  # Debugging statement
    rpy = mat2euler(quat2mat(quat))
    ##print("RPY (roll, pitch, yaw):", rpy)  # Debugging statement
    return np.array([_wrap_to_2pi(rpy[0]), _wrap_to_pi(rpy[1]), _wrap_to_pi(rpy[2])])

def _roll_pitch_correction(quat):
    """
    Given current (roll, pitch) and desired (roll, pitch), returns roll and pitch actions to maintain desired angles
    """
    rp_des = np.array([np.pi, 0])  # Desired roll and pitch to maintain
    # put the quat first dimension to last
    # maniskill quaternions format is differnt from robosuite
    quat = np.roll(quat, -1)
    rp = _quat2euler(quat)[:-1]
    ##print("current pos:",rp)
    ##print("desired pos:",rp_des)
    ##print("Current roll-pitch:", rp)  # Debugging statement
    # give a larger speed to correction, and a negative sign
    # simply works but IDK why
    action = -5*(rp_des - rp)
    # set a threshold for roll and pitch (< threshold -> no action)
    action = np.where(np.abs(action) < 0.05, 0, action)
    ##print("Action:", action)  # Debugging statement
    return action


import pdb
class PrimitiveSkillDelta():
    def __init__(
        self,
        skill_indices=None,
        home_pos=(0, 0, 0.3),
        home_wrist_ori=0.0,
        waypoint_height=0.05,
        aff_pos_thresh=None,
        use_yaw=False,
        **kwargs
    ):

        """
        Args:
            skill_indices (dict): assigns one-hot vector indices to primitive skill names.
                skill names can be selected from TODO            
            home_pos (3-tuple):
                position in 3d space for end effector to return to after each primitive action (excluding move_to)
                if not specified, position of the end effector when the PrimitiveSkill class is initialized is used
            home_wrist_ori (float): wrist orientation at home position in radians
            waypoint_height (float): height of waypoint used in skills such as pick, place
        """
   
        self.home_pos = home_pos
        self.home_wrist_ori = _wrap_to_pi(home_wrist_ori)
        self.waypoint_height = waypoint_height
        self.use_yaw = use_yaw

        self.skill_names = [
            "move_to",
            "pick",
            "place",
            "push",
            "gripper_release",
            "gripper_close",
            "atomic",
            "move_to_w_gripper_closed",
            "rotate",
        ]

        self.name_to_skill = {
            "move_to" : self._move_to,
            "gripper_release" : self._gripper_release,
            "gripper_close" : self._gripper_close,
            "pick" : self._pick,
            "place" : self._place,
            "push" : self._push,
            "atomic": self._atomic,
            "move_to_w_gripper_closed" : self._move_to_w_gripper_closed,
        }
        
        self.max_steps = {
            "move_to" : 150,
            "gripper_release" : 10,
            "gripper_close" : 10,
            "pick" : 150,
            "place" : 150,
            "push" : 200,
            "move_to_w_gripper_closed" : 150,
            "move_home" : 150,
        }
        
        self.name_to_num_params = {
            "move_to" : 5 if self.use_yaw else 4,
            "gripper_release" : 0,
            "gripper_close" : 0,
            "pick" : 4 if self.use_yaw else 3,
            "place" : 4 if self.use_yaw else 3,
            "push" : 8 if self.use_yaw else 7,
            "atomic": 7,
            "move_to_w_gripper_closed" : 4 if self.use_yaw else 3,
        }

        self.skill_indices = skill_indices
        if not skill_indices:
            self.skill_indices = {
                0 : "move_to",
                1 : "pick",
                2 : "place",
                3 : "push",
                4 : "gripper_release",
                5 : "gripper_close",
                6 : "atomic",
                7 : "move_to_w_gripper_closed",
            }

        for key in self.skill_indices.keys():
            assert self.skill_indices[key] in self.skill_names, f"skill {self.skill_indices[key]} is undefined. skill name must be one of {self.skill_names}"

        self.n_skills = len(self.skill_indices)
        self.max_num_params = max([self.name_to_num_params[skill_name] for skill_name in self.skill_indices.values()])

        self.reset() # init grip_steps,steps,phase,prev_success,skill_failed

    def get_action(self, action, obs):
        """
        Args:
            action (tuple): one-hot vector for skill selection concatenated with skill parameters
                one-hot vector dimension must be same as self.n_skills. skill parameter can have variable dimension

        Returns:
            action (7-tuple): action commands for simulation environment - (position commands, orientation commands, gripper command)    
            skill_done (bool): True if goal skill completed successfully or if max allowed steps is reached
        """
        # choose right skill
        skill_idx = np.argmax(action[:self.n_skills])
        skill = self.name_to_skill[self.skill_indices[skill_idx]]
        
        # extract params
        params = action[self.n_skills:]
        return skill(obs, params)
    
    def _atomic(self, obs, params, robot_id=0):
        action = np.array(params)
        skill_done=True
        skill_success=True
        return action, skill_done, skill_success
        
    def _move_to_w_gripper_closed(self, obs, params, robot_id=0, speed=0.5, thresh=0.005, yaw_thresh=0.1, slow_speed=0.15, count_steps=True,max_steps=None):
        """
        Moves end effector to goal position and orientation with gripper closed
        Args:
            obs: observation dict from environment
            params (tuple of floats): [goal_pos, goal_yaw, gripper_command]
                goal_pos (3-tuple): goal end effector position (x, y, z) in world
                gripper_command (float): gripper is closed if > 0, opened if <= 0 
                goal_yaw (tuple): goal yaw angle for end effector
            robot_id (int): specifies which robot's observations are used (if more than one robots exist in environment)
            speed (float): controls magnitude of position commands. Values over ~0.75 is not recommended
            thresh (float): how close end effector position must be to the goal for skill to be complete
            yaw_thresh (float): how close end effector yaw angle must be to the goal value for skill to be complete
            normalized input (bool): set to True if input parameters are normalized to [-1, 1]. set to False to use raw params
        Returns:
            action: 7d action commands for simulation environment - (position commands, orientation commands, gripper command)
            skill_done: True if goal skill completed successfully or if max allowed steps is reached
        """
        goal_pos = params[:3]
        # goal_yaw = params[3]
        # gripper_action = params[4]
        if self.use_yaw:
            goal_yaw = params[3]
            # gripper_action = 1 if params[4] > 0 else -1

        else:
            goal_yaw = 0.0
            # gripper_action = 1 if params[3] > 0 else -1

        gripper_action = 1
        if max_steps is None:
            max_steps = self.max_steps["move_to"]

        ori_speed = 1
        slow_dist = 0.05 # slow down when the end effector is slow_dist away from goal

        skill_done = False
        success = False

        eef_pos = obs[f"robot{robot_id}_eef_pos"]
        eef_quat = obs[f"robot{robot_id}_eef_quat"]
        pos_error = goal_pos - eef_pos

        if goal_yaw:
            goal_ori = _wrap_to_pi(goal_yaw)
        else:
            goal_ori = self.home_wrist_ori

        cur_ori = _quat2euler(obs[f"robot{robot_id}_eef_quat"])
        cur_yaw = cur_ori[-1]
        yaw_error = goal_ori - cur_yaw
        pos_reached = np.all(np.abs(pos_error) < thresh)
        yaw_reached = np.abs(yaw_error) < yaw_thresh
        
        # set goal reached condition depending on use_yaw parameter
        if self.use_yaw:
            goal_reached = pos_reached and yaw_reached
        else:
            goal_reached = pos_reached

        # if close to goal, reduce speed
        if np.abs(np.linalg.norm(pos_error)) < slow_dist:
            speed = slow_speed
        if abs(yaw_error) < 0.75:
            ori_speed = 0.05
        pos_action = speed * (pos_error / (np.linalg.norm(pos_error)+1e-6)) # unit vector in direction of goal * speed
        ori_action = np.append(_roll_pitch_correction(eef_quat), np.sign(yaw_error) * ori_speed)
        action = np.concatenate([pos_action, ori_action, np.array([gripper_action])])

        # max steps reached - skill done with fail
        if count_steps and (self.steps > max_steps):
            print("Max steps for primitive reached: ", self.steps)
            print(f"Goal was {params}\nReached {eef_pos}, {cur_yaw}")
            self.steps = 0
            success = False
            skill_done = True

        # goal is reached - skill done with success
        if goal_reached:
            success = True
            skill_done = True
            if count_steps:
                self.steps = 0
        else:
            if count_steps:
                self.steps += 1
        return action, skill_done, success

    def _move_to(self, obs, params, robot_id=0, speed=0.5, thresh=0.005, yaw_thresh=0.1, slow_speed=0.15, count_steps=True,max_steps=None):
        
        """
        Moves end effector to goal position and orientation.
        Args:
            obs: observation dict from environment
            params (tuple of floats): [goal_pos, goal_yaw, gripper_command]
                goal_pos (3-tuple): goal end effector position (x, y, z) in world
                gripper_command (float): gripper is closed if > 0, opened if <= 0 
                goal_yaw (tuple): goal yaw angle for end effector
            robot_id (int): specifies which robot's observations are used (if more than one robots exist in environment)
            speed (float): controls magnitude of position commands. Values over ~0.75 is not recommended
            thresh (float): how close end effector position must be to the goal for skill to be complete
            yaw_thresh (float): how close end effector yaw angle must be to the goal value for skill to be complete
            normalized input (bool): set to True if input parameters are normalized to [-1, 1]. set to False to use raw params
        Returns:
            action: 7d action commands for simulation environment - (position commands, orientation commands, gripper command)
            skill_done: True if goal skill completed successfully or if max allowed steps is reached
        """
        goal_pos = params[:3]
        # goal_yaw = params[3]
        # gripper_action = params[4]
        if self.use_yaw:
            goal_yaw = params[3]
            gripper_action = 1 if params[4] > 0 else -1

        else:
            goal_yaw = 0.0
            gripper_action = 1 if params[3] > 0 else -1

        if max_steps is None:
            max_steps = self.max_steps["move_to"]

        ori_speed = 5
        slow_dist = 0.02 # slow down when the end effector is slow_dist away from goal

        skill_done = False
        success = False

        eef_pos = obs[f"robot{robot_id}_eef_pos"]
        eef_quat = obs[f"robot{robot_id}_eef_quat"]
        pos_error =goal_pos-eef_pos

        if goal_yaw:
            goal_ori = _wrap_to_pi(goal_yaw)
        else:
            goal_ori = self.home_wrist_ori

        cur_ori = _quat2euler(obs[f"robot{robot_id}_eef_quat"])
        cur_yaw = cur_ori[-1]
        yaw_error = goal_ori - cur_yaw
        pos_reached = np.all(np.abs(pos_error) < thresh)
        yaw_reached = np.abs(yaw_error) < yaw_thresh
        
        # set goal reached condition depending on use_yaw parameter
        if self.use_yaw:
            goal_reached = pos_reached and yaw_reached
        else:
            goal_reached = pos_reached

        # if close to goal, reduce speed
        if np.abs(np.linalg.norm(pos_error)) < slow_dist:
            speed = slow_speed
        if abs(yaw_error) < 0.5:
            ori_speed = 1
        if abs(yaw_error) < 0.05:
            ori_speed = 0
        
        # if pos_error[k]<threshold then set to 0
        pos_error = np.where(np.abs(pos_error) < thresh, 0, pos_error)
        pos_action = speed * (pos_error / (np.linalg.norm(pos_error)+1e-6)) # unit vector in direction of goal * speed
        ori_action = np.append(_roll_pitch_correction(eef_quat), np.sign(yaw_error) * ori_speed)
        action = np.concatenate([pos_action, ori_action, np.array([gripper_action])])

        # max steps reached - skill done with fail
        if count_steps and (self.steps > max_steps):
            print("Max steps for primitive reached: ", self.steps)
            print(f"Goal was {params}\nReached {eef_pos}, {cur_yaw}")
            self.steps = 0
            success = False
            skill_done = True

        # goal is reached - skill done with success
        if goal_reached:
            success = True
            skill_done = True
            if count_steps:
                self.steps = 0
        else:
            if count_steps:
                self.steps += 1

        return action, skill_done, success

    def _gripper_release(self, obs={}, params=(), robot_id=0,max_steps=None):
        """
        Opens gripper

        Args:
            obs: observation dict from environment - not used
            params (tuple of floats): not used
        
        Returns:
            action: 7d action commands for simulation environment - (position commands, orientation commands, gripper command)
            skill_done: True if goal skill completed successfully or if max allowed steps is reached
        """
        if max_steps is None:
            max_steps = self.max_steps["gripper_release"]
        action = np.array([0, 0, 0, 0, 0, 0, 1.0])

        if self.grip_steps < max_steps:
            self.grip_steps += 1
            return action, False, False
        
        self.grip_steps = 0
        return action, True, True

    def _gripper_close(self, obs={}, params=(), robot_id=0,max_steps=None):
        """
        Closes gripper

        Args:
            obs: observation dict from environment - not used
            params (tuple of floats): not used
        
        Returns:
            action: 7d action commands for simulation environment - (position commands, orientation commands, gripper command)
            skill_done: True if goal skill completed successfully or if max allowed steps is reached
        """
        if max_steps is None:
            max_steps = self.max_steps["gripper_close"]
        action = np.array([0, 0, 0, 0, 0, 0, -1.0])

        if self.grip_steps < max_steps:
            self.grip_steps += 1
            return action, False, False
        
        self.grip_steps = 0
        return action, True, True

    def _pick(self, obs, params, robot_id=0, speed=0.5, thresh=0.005, yaw_thresh=0.005,max_steps=None):       
        """
        Picks up an object at a target position and returns to home position.
        Args:
            obs: observation dict from environment
            params (tuple of floats): [goal_pos, goal_yaw]
                goal_pos (3-tuple): goal end effector position (x, y, z) in world
                goal_yaw (tuple): goal yaw angle for end effector
            robot_id (int): specifies which robot's observations are used (if more than one robots exist in environment)
            speed (float): controls magnitude of position commands. Values over ~0.75 is not recommended
            thresh (float): how close end effector position must be to the goal for skill to be complete
            yaw_thresh (float): how close end effector yaw angle must be to the goal value for skill to be complete
            normalized input (bool): set to True if input parameters are normalized to [-1, 1]. set to False to use raw params
        Returns:
            action: 7d action commands for simulation environment - (position commands, orientation commands, gripper command)
            skill_done: True if goal skill completed successfully or if max allowed steps is reached
        """
        goal_pos = params[:3]
        if self.use_yaw:
            goal_yaw = params[3]
        else:
            goal_yaw = 0.0

        if max_steps is None:
            max_steps = self.max_steps["pick"]

        above_pos = (goal_pos[0], goal_pos[1], self.waypoint_height+params[2])


        if self.prev_success:
            self.phase += 1
            self.prev_success = False

        # if max steps reached go to rehoming phase
        if not self.phase == 4 and self.steps > max_steps:
            self.phase = 4
            self.skill_failed = True
            print("Max steps for pick reached:", self.steps)

        # phase 0: move to above grip site
        if self.phase == 0:
            if self.use_yaw:
                params = np.concatenate([above_pos, np.array([goal_yaw, 1])])
            else:
                ##print("phase 0","above_pos",above_pos)
                params = np.concatenate([above_pos, [1]])
            action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, speed=speed, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)

        # phase 1: move down to grip site
        if self.phase == 1:
            if self.use_yaw:
                params = np.concatenate([goal_pos, np.array([goal_yaw, 1])])
            else:
                ##print("phase 1","goal_pos",goal_pos)
                params = np.concatenate([goal_pos, [1]])
            action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, speed=speed, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)

        # phase 2: grip
        if self.phase == 2:
            ##print("phase 2")
            action, skill_done, self.prev_success = self._gripper_close()

        # phase 3: lift
        if self.phase == 3:
            if self.use_yaw:
                params = np.concatenate([above_pos, np.array([goal_yaw, -1])])
            else:
                params = np.concatenate([above_pos, [-1]])
            ##print("phase 3","above_pos",above_pos)
            action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, speed=speed, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)

        # phase 4: move to home
        if self.phase == 4:
            self.grip_steps = 0
            if self.use_yaw:
                params = np.concatenate([self.home_pos, np.array([0, -1])])
            else:
                params = np.concatenate([self.home_pos, [-1]])

            if self.skill_failed:
                params[-1] = 1
            ##print("phase 4","home_pos",self.home_pos)
            action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, speed=speed, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)
            if self.prev_success:
                skill_success=(not self.skill_failed)
                skill_done=True
                self.reset()
                return action, skill_done, skill_success
            self.home_steps+=1
            if self.home_steps>self.max_steps["move_home"]:
                skill_done=True
                skill_success=False
                print('max move home steps achieved', self.home_steps)
                self.reset()
                return action, skill_done, skill_success

        
        skill_done=False
        skill_success=False
        self.steps += 1

        return action, skill_done, skill_success

    def _place(self, obs, params, robot_id=0, speed=0.5, thresh=0.005, yaw_thresh=0.005):       
        """
        Places an object at a target position and returns to home position.
        Args:
            obs: observation dict from environment
            params (tuple of floats): [goal_pos, goal_yaw]
                goal_pos (3-tuple): goal end effector position (x, y, z) in world
                goal_yaw (tuple): goal yaw angle for end effector
            robot_id (int): specifies which robot's observations are used (if more than one robots exist in environment)
            speed (float): controls magnitude of position commands. Values over ~0.75 is not recommended
            thresh (float): how close end effector position must be to the goal for skill to be complete
            yaw_thresh (float): how close end effector yaw angle must be to the goal value for skill to be complete
            normalized input (bool): set to True if input parameters are normalized to [-1, 1]. set to False to use raw params
        Returns:
            action: 7d action commands for simulation environment - (position commands, orientation commands, gripper command)
            skill_done: True if goal skill completed successfully or if max allowed steps is reached
        """
        goal_pos = params[:3]        
        if self.use_yaw:
            goal_yaw = params[3]
        else:
            goal_yaw = 0.0
            
        max_steps = self.max_steps["place"]

        above_pos = (goal_pos[0], goal_pos[1], self.waypoint_height+params[2])

        skill_success = False

        if self.prev_success:
            self.phase += 1
            self.prev_success = False
        
        if not self.phase == 4 and self.steps > max_steps:
            self.phase = 4
            self.skill_failed=True
            #print("max steps for place reached:", self.steps)

        # phase 0: move to above place site
        if self.phase == 0:
            #print("phase 0")
            if self.use_yaw:
                params = np.concatenate([above_pos, np.array([goal_yaw, -1.])])
            else:
                params = np.concatenate([above_pos, [-1.]])
            ##print("phase 0","above_pos",above_pos)
            action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, speed=speed, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)

        # phase 1: move down to drop site
        if self.phase == 1:
            #print("phase 1")
            if self.use_yaw:
                params = np.concatenate([goal_pos, np.array([goal_yaw, -1.])])
            else:
                params = np.concatenate([goal_pos, [-1.]])
            ##print("phase 1","goal_pos",goal_pos)
            action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, speed=speed, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)

        # phase 2: release
        if self.phase == 2:
            #print("phase 2")
            ##print("phase 2")
            action, skill_done, self.prev_success = self._gripper_release()

        # phase 3: lift
        if self.phase == 3:
            #print("phase 3")
            if self.use_yaw:
                params = np.concatenate([above_pos, np.array([goal_yaw, 1.])])
            else:
                params = np.concatenate([above_pos, [1.]])
            ##print("phase 3","above_pos",above_pos)
            action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, speed=speed, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)

        # phase 4: move to home
        if self.phase == 4:
            ##print("phase 4")
            self.grip_steps = 0
            self.steps = 0
            if self.use_yaw:
                params = np.concatenate([self.home_pos, np.array([0, 1.])])
            else:
                params = np.concatenate([self.home_pos, [1.]])
            ##print("phase 4","home_pos",self.home_pos)
            action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, speed=speed, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)
            
            if self.prev_success:
                skill_done=True
                skill_success= (not self.skill_failed)
                self.reset()
                return action, skill_done, skill_success
            
            self.home_steps+=1
            if self.home_steps>self.max_steps["move_home"]:
                skill_done=True
                skill_success=False
                #print('max move home steps achieved', self.home_steps)
                self.reset()
                return action, skill_done, skill_success

        
        self.steps += 1
        skill_done=False
        skill_success=False
        
        return action, skill_done, skill_success

    def _push(self, obs, params, robot_id=0, speed=0.5, thresh=0.005, yaw_thresh=0.005):
        """
        Moves end effector to above push starting position, moves down to start position, moves to goal position, up, then back to the home position,
        Positions are defined in world coordinates

        Args:
            obs: current observation
            params:
                start_pos (3-tuple or array of floats): world coordinate location to start push
                end_pos (3-tuple or array of floats): world coordinate location to end push
                wrist_yaw (float): wrist joint angle to keep while pushing
                gripper_closed (bool): if True, keeps gripper closed during pushing 
            robot_id (int): id of robot to be controlled
            speed (float): how fast the end effector will move (0,1]
        
        Returns:
            obs, reward, done, info from environment's step function (or lists of these if self.return_all_states is True)
        """
        start_pos = params[:3]
        end_pos = params[3:6]
        if self.use_yaw:
            goal_yaw = params[6]
            gripper_action = 1 if params[7] > 0 else -1
        else:
            goal_yaw = 0.0
            gripper_action = -1 if len(params)>=7 and params[6] < 0 else 1

        max_steps = self.max_steps["push"]

        above_start_pos = (start_pos[0], start_pos[1], self.waypoint_height+params[2])
        above_end_pos = (end_pos[0], end_pos[1], self.waypoint_height+params[2])

        
        if self.prev_success:
            self.phase += 1
            self.prev_success = False

        if not self.phase == 4 and self.steps > max_steps:
            self.phase = 4
            self.skill_failed = True
            #print("max steps for push reached:", self.steps)

        # phase 0: move to above start pos
        if self.phase == 0:
            #print("phase 0")
            params = np.concatenate([above_start_pos, np.array([goal_yaw, gripper_action])])
            action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, speed=speed, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)
        
        # phase 1: move down to start pos
        if self.phase == 1:
            #print("phase 1")
            params = np.concatenate([start_pos, np.array([goal_yaw, gripper_action])])
            action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, speed=speed, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)

        # phase 2: move to goal pos
        if self.phase == 2:
            #print("phase 2")
            params = np.concatenate([end_pos, np.array([goal_yaw, gripper_action])])
            action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, speed=speed, thresh=0.03, yaw_thresh=yaw_thresh, count_steps=False, slow_speed=0.5)

        # phase 3: move to above end pos
        if self.phase == 3:
            #print("phase 3")
            params = np.concatenate([above_end_pos, np.array([goal_yaw, gripper_action])])
            action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, speed=speed, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)

        # phase 4: move to home
        if self.phase == 4:
            self.steps = 0
            params = np.concatenate([self.home_pos, np.array([0, -1])])
            action, skill_done, self.prev_success = self._move_to(obs=obs, params=params, robot_id=robot_id, speed=speed, thresh=thresh, yaw_thresh=yaw_thresh, count_steps=False)
            self.home_steps+=1
            if self.prev_success:
                skill_success=(not self.skill_failed)
                skill_done=True
                self.reset()
                return action, skill_done, skill_success
            
            self.home_steps+=1
            if self.home_steps>self.max_steps['move_home']:
                #print('max move home steps achieved', self.home_steps)
                skill_done=True
                skill_success=False
                self.reset()
                return action, skill_done, skill_success
            

        self.steps += 1
        
        skill_done=False
        skill_success=False
        
        return action, skill_done, skill_success
    
    
    def reset(self):
        self.home_steps=0
        self.grip_steps=0
        self.steps = 0
        self.phase = 0
        self.prev_success = False
        self.skill_failed = False
        
        
import numpy as np

import numpy as np

class PrimitiveSkillAbs():
    def __init__(
        self,
        skill_indices=None,
        home_pos=(0, 0, 0.3),
        waypoint_height=0.08,
        **kwargs
    ):
        """
        Args:
            skill_indices (dict): assigns one-hot vector indices to primitive skill names
            home_pos (3-tuple): position in 3d space for end effector to return to after each primitive action
            waypoint_height (float): height of waypoint used in skills such as pick, place
            robot: robot instance for getting current pose
        """
   
        self.home_pos = home_pos
        self.waypoint_height = waypoint_height
        
        # Fixed orientation [3.14, 0, 0]
        self.fixed_ori = np.array([3.14, 0, 0])

        self.skill_names = [
            "move_to",
            "pick",
            "place",
            "push",
            "gripper_release",
            "gripper_close",
        ]

        self.name_to_skill = {
            "move_to": self._move_to,
            "gripper_release": self._gripper_release,
            "gripper_close": self._gripper_close,
            "pick": self._pick,
            "place": self._place,
            "push": self._push,
        }
        
        self.max_steps = {
            "move_to": 150,
            "gripper_release": 10,
            "gripper_close": 10,
            "pick": 150,
            "place": 150,
            "push": 200,
            "move_home": 150,
        }
        
        self.name_to_num_params = {
            "move_to": 4,
            "gripper_release": 0,
            "gripper_close": 0,
            "pick": 3,
            "place": 3,
            "push": 6,
        }

        self.skill_indices = skill_indices or {
            0: "move_to",
            1: "pick",
            2: "place",
            3: "push",
            4: "gripper_release",
            5: "gripper_close",
        }

        for key in self.skill_indices.keys():
            assert self.skill_indices[key] in self.skill_names, f"skill {self.skill_indices[key]} is undefined. skill name must be one of {self.skill_names}"

        self.n_skills = len(self.skill_indices)
        self.max_num_params = max([self.name_to_num_params[skill_name] for skill_name in self.skill_indices.values()])

        self.reset()

    def get_action(self, action, obs):
        """
        Args:
            action (tuple): one-hot vector for skill selection concatenated with skill parameters
            obs: observation dict from environment

        Returns:
            action: 7-tuple for simulation environment - (position commands, orientation commands, gripper command)    
            skill_done (bool): True if completed successfully or max steps reached
        """
        skill_idx = np.argmax(action[:self.n_skills])
        skill = self.name_to_skill[self.skill_indices[skill_idx]]
        params = action[self.n_skills:]
        return skill(obs, params)

    def _move_to(self, obs, params, robot_id=0, speed=0.5, thresh=0.005, count_steps=True, max_steps=None):
        """
        Moves end effector to absolute goal position with fixed orientation.
        First calculates error using current end effector position from obs,
        then returns action as target_pos - robot.pose.p for the controller.
        """
        target_pos = params[:3]
        gripper_action = 1 if params[3] > 0 else -1

        if max_steps is None:
            max_steps = self.max_steps["move_to"]

        skill_done = False
        success = False
        
        # Get current end effector position from obs
        eef_pos = obs[f"robot{robot_id}_eef_pos"]
        
        # Calculate position error using current end effector position
        pos_error = target_pos - eef_pos
        pos_reached = np.all(np.abs(pos_error) < thresh)
        
        # Combine target position with fixed orientation and gripper
        action = np.concatenate([target_pos, self.fixed_ori, np.array([gripper_action])])

        if count_steps and (self.steps > max_steps):
            print(f"Max steps reached: {self.steps}")
            print(f"Goal was {target_pos}\nReached {eef_pos}")
            self.steps = 0
            success = False
            skill_done = True

        if pos_reached:
            success = True
            skill_done = True
            if count_steps:
                self.steps = 0
        else:
            if count_steps:
                self.steps += 1

        return action, skill_done, success

    def _gripper_release(self, obs={}, params=(), robot_id=0, max_steps=None):
        """Opens gripper"""
        if max_steps is None:
            max_steps = self.max_steps["gripper_release"]
            
        # Zero position action, fixed orientation, open gripper
        action = np.concatenate([np.zeros(3), self.fixed_ori, np.array([1.0])])

        if self.grip_steps < max_steps:
            self.grip_steps += 1
            return action, False, False
        
        self.grip_steps = 0
        return action, True, True

    def _gripper_close(self, obs={}, params=(), robot_id=0, max_steps=None):
        """Closes gripper"""
        if max_steps is None:
            max_steps = self.max_steps["gripper_close"]
            
        # Zero position action, fixed orientation, close gripper
        action = np.concatenate([np.zeros(3), self.fixed_ori, np.array([-1.0])])

        if self.grip_steps < max_steps:
            self.grip_steps += 1
            return action, False, False
        
        self.grip_steps = 0
        return action, True, True

    def _pick(self, obs, params, robot_id=0, speed=0.5, thresh=0.005):
        """Picks up an object at absolute target position"""
        target_pos = params[:3]
        above_pos = np.array([target_pos[0], target_pos[1], self.waypoint_height + target_pos[2]])
        
        max_steps = self.max_steps["pick"]

        if self.prev_success:
            self.phase += 1
            self.prev_success = False

        if not self.phase == 4 and self.steps > max_steps:
            self.phase = 4
            self.skill_failed = True

        # Move above -> Move down -> Grip -> Lift -> Home
        if self.phase == 0:
            params = np.concatenate([above_pos, [1]])
            action, skill_done, self.prev_success = self._move_to(obs, params, robot_id, speed, thresh, count_steps=False)
        elif self.phase == 1:
            params = np.concatenate([target_pos, [1]])
            action, skill_done, self.prev_success = self._move_to(obs, params, robot_id, speed, thresh, count_steps=False)
        elif self.phase == 2:
            action, skill_done, self.prev_success = self._gripper_close()
        elif self.phase == 3:
            params = np.concatenate([above_pos, [-1]])
            action, skill_done, self.prev_success = self._move_to(obs, params, robot_id, speed, thresh, count_steps=False)
        elif self.phase == 4:
            params = np.concatenate([self.home_pos, [-1]])
            action, skill_done, self.prev_success = self._move_to(obs, params, robot_id, speed, thresh, count_steps=False)
            
            if self.prev_success:
                self.reset()
                return action, True, not self.skill_failed
                
            self.home_steps += 1
            if self.home_steps > self.max_steps["move_home"]:
                self.reset()
                return action, True, False

        self.steps += 1
        return action, False, False

    def _place(self, obs, params, robot_id=0, speed=0.5, thresh=0.005):
        """Places object at absolute target position"""
        target_pos = params[:3]
        above_pos = np.array([target_pos[0], target_pos[1], self.waypoint_height + target_pos[2]])
        
        max_steps = self.max_steps["place"]

        if self.prev_success:
            self.phase += 1
            self.prev_success = False

        if not self.phase == 4 and self.steps > max_steps:
            self.phase = 4
            self.skill_failed = True

        # Move above -> Move down -> Release -> Lift -> Home
        if self.phase == 0:
            params = np.concatenate([above_pos, [-1]])
            action, skill_done, self.prev_success = self._move_to(obs, params, robot_id, speed, thresh, count_steps=False)
        elif self.phase == 1:
            params = np.concatenate([target_pos, [-1]])
            action, skill_done, self.prev_success = self._move_to(obs, params, robot_id, speed, thresh, count_steps=False)
        elif self.phase == 2:
            action, skill_done, self.prev_success = self._gripper_release()
        elif self.phase == 3:
            params = np.concatenate([above_pos, [1]])
            action, skill_done, self.prev_success = self._move_to(obs, params, robot_id, speed, thresh, count_steps=False)
        elif self.phase == 4:
            params = np.concatenate([self.home_pos, [1]])
            action, skill_done, self.prev_success = self._move_to(obs, params, robot_id, speed, thresh, count_steps=False)
            
            if self.prev_success:
                self.reset()
                return action, True, not self.skill_failed
                
            self.home_steps += 1
            if self.home_steps > self.max_steps["move_home"]:
                self.reset()
                return action, True, False

        self.steps += 1
        return action, False, False

    def _push(self, obs, params, robot_id=0, speed=0.5, thresh=0.005):
        """Pushes object between absolute start and end positions"""
        start_pos = params[:3]
        end_pos = params[3:6]
        
        above_start = np.array([start_pos[0], start_pos[1], self.waypoint_height + start_pos[2]])
        above_end = np.array([end_pos[0], end_pos[1], self.waypoint_height + end_pos[2]])
        
        max_steps = self.max_steps["push"]

        if self.prev_success:
            self.phase += 1
            self.prev_success = False

        if not self.phase == 4 and self.steps > max_steps:
            self.phase = 4
            self.skill_failed = True

        # Above start -> Down to start -> Push to end -> Lift -> Home
        if self.phase == 0:
            params = np.concatenate([above_start, [1]])
            action, skill_done, self.prev_success = self._move_to(obs, params, robot_id, speed, thresh, count_steps=False)
        elif self.phase == 1:
            params = np.concatenate([start_pos, [1]])
            action, skill_done, self.prev_success = self._move_to(obs, params, robot_id, speed, thresh, count_steps=False)
        elif self.phase == 2:
            params = np.concatenate([end_pos, [1]])
            action, skill_done, self.prev_success = self._move_to(obs, params, robot_id, speed, thresh=0.03, count_steps=False)
        elif self.phase == 3:
            params = np.concatenate([above_end, [1]])
            action, skill_done, self.prev_success = self._move_to(obs, params, robot_id, speed, thresh, count_steps=False)
        elif self.phase == 4:
            params = np.concatenate([self.home_pos, [-1]])
            action, skill_done, self.prev_success = self._move_to(obs, params, robot_id, speed, thresh, count_steps=False)
            
            if self.prev_success:
                self.reset()
                return action, True, not self.skill_failed
                
            self.home_steps += 1
            if self.home_steps > self.max_steps["move_home"]:
                self.reset()
                return action, True, False

        self.steps += 1
        return action, False, False

    def reset(self):
        """Reset all internal state variables"""
        self.home_steps = 0
        self.grip_steps = 0
        self.steps = 0
        self.phase = 0
        self.prev_success = False
        self.skill_failed = False