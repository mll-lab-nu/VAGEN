import numpy as np
from typing import List

from ...core.room import BaseRoom
from ...core.object import Object, Agent


def rotation_matrix_from_ori(ori: np.ndarray) -> np.ndarray:
    ori_to_R = {
        (0, 1): np.array([[1, 0], [0, 1]]),
        (1, 0): np.array([[0, -1], [1, 0]]),
        (0, -1): np.array([[-1, 0], [0, -1]]),
        (-1, 0): np.array([[0, 1], [-1, 0]]),
    }
    key = tuple(int(x) for x in (ori.tolist() if hasattr(ori, "tolist") else ori))
    return ori_to_R.get(key, ori_to_R[(0, 1)])


def transform_point(pos_world: np.ndarray, anchor_pos: np.ndarray, anchor_ori: np.ndarray) -> np.ndarray:
    R = rotation_matrix_from_ori(anchor_ori)
    return (R @ (pos_world.astype(float) - anchor_pos.astype(float))).astype(float)


def transform_ori(ori_world: np.ndarray, anchor_ori: np.ndarray) -> np.ndarray:
    R = rotation_matrix_from_ori(anchor_ori)
    v = (R @ ori_world.astype(float)).astype(int)
    return np.array([int(np.sign(v[0])), int(np.sign(v[1]))], dtype=int)


def transform_baseroom(room: BaseRoom, anchor_pos: np.ndarray, anchor_ori: np.ndarray) -> BaseRoom:
    # NOTE: Only transform positions. Facing/orientation is evaluated in absolute frame
    # (and local/rooms predicted facings are already normalized upstream).
    for obj in room.objects:
        p = transform_point(obj.pos, anchor_pos, anchor_ori)
        obj.pos = p
    return room


def inv_transform_point(pos_local: np.ndarray, anchor_pos: np.ndarray, anchor_ori: np.ndarray) -> np.ndarray:
    R = rotation_matrix_from_ori(anchor_ori)
    return (R.T @ pos_local.astype(float)) + anchor_pos.astype(float)


def inv_transform_ori(ori_local: np.ndarray, anchor_ori: np.ndarray) -> np.ndarray:
    R = rotation_matrix_from_ori(anchor_ori)
    v = (R.T @ ori_local.astype(float))
    return np.array([int(np.sign(v[0])), int(np.sign(v[1]))], dtype=int)


def br_from_anchor_to_initial(br_anchor: BaseRoom, anchor_pos: np.ndarray, anchor_ori: np.ndarray, gt_agent: Agent) -> BaseRoom:
    objs_world = []
    for o in br_anchor.objects:
        p_w = inv_transform_point(o.pos, anchor_pos, anchor_ori)
        if o.has_orientation:
            ori_w = inv_transform_ori(o.ori, anchor_ori)
        else:
            ori_w = o.ori
        objs_world.append(Object(name=o.name, pos=p_w, ori=ori_w, has_orientation=o.has_orientation))
    br_world = BaseRoom(objects=objs_world, name=br_anchor.name)
    return transform_baseroom(
        br_world,
        anchor_pos=np.array(gt_agent.init_pos, dtype=float),
        anchor_ori=np.array(gt_agent.init_ori, dtype=int),
    )


__all__ = [
    "rotation_matrix_from_ori",
    "transform_point",
    "transform_ori",
    "inv_transform_point",
    "inv_transform_ori",
    "transform_baseroom",
    "br_from_anchor_to_initial",
]


