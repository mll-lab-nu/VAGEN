import numpy as np
from typing import Dict

from ...core.room import BaseRoom, Object, Gate
from ...core.relationship import (
    PairwiseRelationshipDiscrete,
    CardinalBinsAllo,
)
from .types import MapCogMetrics


def compute_dir_sim(pred_room: BaseRoom, gt_room: BaseRoom) -> float:
    pred = {o.name: o for o in pred_room.objects} | {'initial': Object(name='initial', pos=np.array([0.0, 0.0]), ori=np.array([0.0, 1.0]))}
    gt = {o.name: o for o in gt_room.objects} | {'initial': Object(name='initial', pos=np.array([0.0, 0.0]), ori=np.array([0.0, 1.0]))}
    names = sorted(gt.keys())
    if len(names) < 2:
        return 1.0
    bin_system = CardinalBinsAllo()
    tot = cor = 0.0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = gt[names[i]], gt[names[j]]
            gt_rel = PairwiseRelationshipDiscrete.relationship(a.pos, b.pos, None, bin_system)
            p1, p2 = pred.get(names[i]), pred.get(names[j])
            if p1 is not None and p2 is not None:
                pr = PairwiseRelationshipDiscrete.relationship(p1.pos, p2.pos, None, bin_system)
                if pr.direction.bin_id == gt_rel.direction.bin_id:
                    cor += 1.0
            tot += 1.0
    return cor / tot if tot else 0.0


def compute_facing_sim(pred_room: BaseRoom, gt_room: BaseRoom) -> float:
    pred = {o.name: o for o in pred_room.objects}
    gt = {o.name: o for o in gt_room.objects}
    names = sorted(gt.keys())
    tot = cor = 0.0
    for name in names:
        g = gt[name]
        name_l = str(g.name).lower()
        if (not g.has_orientation) or isinstance(g, Gate) or ("door" in name_l):
            continue
        p = pred.get(name)
        tot += 1.0
        if p is not None and np.array_equal(p.ori, g.ori):
            cor += 1.0
    return cor / tot if tot else 1.0


def compute_pos_sim(pred_room: BaseRoom, gt_room: BaseRoom, allow_scale: bool, pos_norm_L: float | None) -> float:
    pred = {o.name: o for o in pred_room.objects}
    gt = {o.name: o for o in gt_room.objects}
    gt_names = sorted(gt.keys())
    matched = [n for n in gt_names if n in pred]
    if len(gt_names) == 0:
        return 1.0
    elif len(matched) == 0:
        return 0.0
    P1 = np.array([pred[n].pos for n in matched], dtype=float)
    P2 = np.array([gt[n].pos for n in matched], dtype=float)
    if allow_scale:
        den = float((P1 * P1).sum())
        if den == 0.0:
            return 0.0
        scale = float((P2 * P1).sum()) / den
    else:
        scale = 1.0
    rmse = np.sqrt(((P1 * scale - P2) ** 2).sum(axis=1).mean())
    L = float(pos_norm_L or float(np.sqrt((P2 ** 2).sum(axis=1).mean())))
    base = float(np.exp(-rmse / L)) if L > 0 else 0.0
    coverage = float(len(matched)) / float(len(gt_names))
    return base * coverage


def compute_overall(dir_sim: float, facing_sim: float, pos_sim: float) -> float:
    return (dir_sim + facing_sim + pos_sim) / 3.0


def compute_map_metrics(pred_room: BaseRoom, gt_room: BaseRoom, allow_scale: bool, pos_norm_L: float | None) -> MapCogMetrics:
    d = compute_dir_sim(pred_room, gt_room)
    f = compute_facing_sim(pred_room, gt_room)
    p = compute_pos_sim(pred_room, gt_room, allow_scale=allow_scale, pos_norm_L=pos_norm_L)
    return MapCogMetrics(dir=float(d), facing=float(f), pos=float(p), overall=float(compute_overall(d, f, p)))


__all__ = [
    "compute_dir_sim",
    "compute_facing_sim",
    "compute_pos_sim",
    "compute_overall",
    "compute_map_metrics",
]


