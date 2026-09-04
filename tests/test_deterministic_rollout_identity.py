from vagen.training.agent_loop.gym_loop import _stable_rollout_id


def test_stable_rollout_id_repeats_for_the_same_trajectory():
    kwargs = {
        "env_name": "Sokoban",
        "seed": 10001,
        "__vagen_rollout_index__": 7,
        "traj_idx": 0,
    }

    assert _stable_rollout_id(kwargs) == _stable_rollout_id(dict(kwargs))


def test_stable_rollout_id_separates_repeated_trajectories():
    base = {
        "env_name": "Sokoban",
        "seed": 10001,
        "__vagen_rollout_index__": 7,
        "traj_idx": 0,
    }
    identities = set()
    for key, value in (
        ("env_name", "FrozenLake"),
        ("seed", 10002),
        ("__vagen_rollout_index__", 8),
        ("traj_idx", 1),
    ):
        changed = dict(base)
        changed[key] = value
        identities.add(_stable_rollout_id(changed))

    assert len(identities | {_stable_rollout_id(base)}) == 5
