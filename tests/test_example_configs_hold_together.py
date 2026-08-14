"""Invariants across every shipped example, checked from the files themselves.

Each of these corresponds to something that shipped broken and that the suite did not
notice, because nothing compared a script against the yaml it loads.
"""

from __future__ import annotations

import glob
import os
import re

import pytest
from omegaconf import OmegaConf

#: Anchored to the repo, not the working directory. Read at module scope from a relative
#: path, a pytest run from anywhere but the root failed at COLLECTION and took the whole
#: suite with it -- not just this file.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = sorted(glob.glob(os.path.join(_ROOT, "examples/train/*/*.sh")))
assert SCRIPTS, "no training scripts found; the glob is wrong, not the repo empty"
BASE_FLAGS = open(os.path.join(_ROOT, "vagen/configs/baseline_vllm.flags")).read()

#: Kept out of the "every yaml is reachable" check, with the reason.
UNREFERENCED_BY_DESIGN = {
    # A frozen copy of the pre-2026-08-10 environment for reproducing the `*_old` wandb
    # runs: it differs from the live yaml by exactly `strict_format: false` and
    # `format_reward: 0.1`, and the rationale for both is in
    # vagen/envs/sokoban/sokoban_env.py's `strict_format` comment. Unrunnable without
    # hand-editing a script, which is the hazard this test exists to catch -- exempted
    # deliberately, because reproducing an old run is what it is for.
    "examples/train/sokoban/train_sokoban_vision_oldenv.yaml",
}


def _flag(text, key, default=None):
    m = re.search(rf"{re.escape(key)}=(\S+)", text)
    return int(m.group(1).rstrip(" \\")) if m else default


def _script_yamls(text, script):
    out = []
    for m in re.finditer(r'data\.(?:train|val)_files="\$SCRIPTDIR/([^"]+)"', text):
        out.append(os.path.join(os.path.dirname(script), m.group(1)))
    return out


@pytest.mark.parametrize("script", SCRIPTS)
def test_a_concat_episode_fits_the_response_region(script):
    """★ T*g <= max_response_length. spatial_gym shipped six scripts where it did not:
    11 turns of 1024 against a 2000-token region, so `exhausted()` ended every episode
    around turn 3 of 11 -- and that env pays its reward only on turn 11, so the runs could
    not score at all while every other metric looked ordinary."""
    text = open(script).read()
    harness = (re.search(r"trainer\.harness=(\w+)", text) or [None, "concat"])[1]
    if harness != "concat":
        pytest.skip("only concat holds the whole episode in one row")
    n_r = _flag(text, "data.max_response_length") or _flag(BASE_FLAGS, "data.max_response_length")
    if n_r is None:
        pytest.fail(f"{script} sets no data.max_response_length and neither does "
                    f"baseline_vllm.flags; verl's default is 512, which none of these fit")
    for path in _script_yamls(text, script):
        for spec in OmegaConf.to_container(OmegaConf.load(path)).get("envs", []):
            g = spec.get("response_length_per_turn")
            if not g:
                continue
            need = int(spec.get("max_turns", 1)) * int(g)
            assert need <= n_r, (
                f"{os.path.basename(script)} + {os.path.basename(path)}: "
                f"{spec.get('max_turns')} turns x {g} = {need} > max_response_length {n_r}")


@pytest.mark.parametrize("script", SCRIPTS)
def test_the_context_window_covers_both_regions(script):
    text = open(script).read()
    ctx = _flag(text, "actor_rollout_ref.rollout.max_model_len")
    if ctx is None:
        pytest.skip("inherits the engine default")
    n_p = _flag(text, "data.max_prompt_length") or _flag(BASE_FLAGS, "data.max_prompt_length", 1000)
    n_r = _flag(text, "data.max_response_length") or _flag(BASE_FLAGS, "data.max_response_length")
    assert n_r is not None, f"{script}: no data.max_response_length anywhere"
    assert n_p + n_r <= ctx, f"{script}: prompt {n_p} + response {n_r} > max_model_len {ctx}"


def test_every_example_yaml_is_reachable_from_a_script():
    """★ Seven sokoban yamls were orphans, including the only pair configured for the
    thinking arm -- and they pointed the reader at a training script for a flag that
    existed in no file. A config nobody can run is a config nobody maintains."""
    referenced = set()
    for script in SCRIPTS:
        referenced.update(os.path.normpath(p) for p in _script_yamls(open(script).read(), script))
    exempt = {os.path.normpath(os.path.join(_ROOT, p)) for p in UNREFERENCED_BY_DESIGN}
    orphans = sorted(
        p for p in glob.glob(os.path.join(_ROOT, "examples/train/*/*.yaml"))
        if os.path.normpath(p) not in referenced and os.path.normpath(p) not in exempt
    )
    assert not orphans, f"no .sh loads these: {orphans}"


def test_experiment_name_defaults_are_unique():
    """★ 15 scripts across five directories shared a default with a sibling, so
    EXPERIMENT_DIR collided: running two arms back to back overwrote the first one's
    run.log and checkpoints, and both reported under one wandb name."""
    seen = {}
    for script in SCRIPTS:
        m = re.search(r"^EXPERIMENT_NAME=\$\{EXPERIMENT_NAME:-([^}]+)\}", open(script).read(), re.M)
        if m:
            seen.setdefault(m.group(1), []).append(script)
    clashes = {k: v for k, v in seen.items() if len(v) > 1}
    assert not clashes, f"shared EXPERIMENT_NAME defaults: {clashes}"


def test_the_two_seed_derivations_agree():
    """★ Evaluation hashed `split` into the per-spec RNG seed and training did not, so the
    same directive produced different seeds on the two sides -- which makes "evaluate on
    the val seeds" unreachable through the directive, and weakens any check that compares
    declared ranges rather than realised sets."""
    from types import SimpleNamespace

    from vagen.evaluate.utils.seeding_utils import generate_seeds_for_spec as ev
    from vagen.gym_agent_dataset import _generate_seeds_for_spec as tr

    for directive in ([100, 200], [1, 50, 1], [7]):
        a = SimpleNamespace(name="Sokoban", n_envs=8, seed=directive, seed_list=None, split="test")
        b = SimpleNamespace(name="Sokoban", n_envs=8, seed=directive, seed_list=None)
        assert ev(a, 0, 0) == tr(b, 0, 0), f"diverged on seed: {directive}"


def test_a_seed_list_of_exactly_n_envs_is_accepted_on_both_sides():
    """Training demanded *more* than n_envs and then sliced to n_envs, so naming exactly
    the seeds you want was rejected. Evaluation always accepted it."""
    from types import SimpleNamespace

    from vagen.evaluate.utils.seeding_utils import generate_seeds_for_spec as ev
    from vagen.gym_agent_dataset import _generate_seeds_for_spec as tr

    a = SimpleNamespace(name="S", n_envs=3, seed=[0], seed_list=[1, 2, 3], split="test")
    b = SimpleNamespace(name="S", n_envs=3, seed=[0], seed_list=[1, 2, 3])
    assert ev(a, 0, 0) == [1, 2, 3]
    assert tr(b, 0, 0) == [1, 2, 3]


@pytest.mark.parametrize("script", SCRIPTS)
def test_the_run_is_stopped_by_its_step_count_and_not_by_an_epoch_count(script):
    """★ verl's fit loop is `for epoch in range(trainer.total_epochs)`, and
    total_training_steps only ends it early. So the real bound is

        min(total_training_steps, total_epochs * floor(n_envs / train_batch_size))

    and with verl's default total_epochs=30 the right-hand term binds whenever the seed
    list is small. spatial_gym (n_envs 50, train_batch_size 32, drop_last=True) got one
    step an epoch and stopped at 30 of its 401 -- never reaching save_freq=100, so a run
    that looked configured for 401 steps produced 30 and no checkpoint. navigation
    stopped at 270. Nothing said so: the run simply ended and reported success.
    """
    text = open(script).read()
    steps = _flag(text, "trainer.total_training_steps")
    batch = _flag(text, "data.train_batch_size")
    if steps is None or batch is None:
        pytest.skip("script does not pin both a step count and a batch size")

    yamls = _script_yamls(text, script)
    train_yaml = next((y for y in yamls if "/train_" in y), None)
    if train_yaml is None:
        pytest.skip("script names no train yaml")
    cfg = OmegaConf.load(train_yaml)
    n_envs = sum(int(e["n_envs"]) for e in cfg["envs"])

    epochs = _flag(text, "trainer.total_epochs") or int(
        OmegaConf.load(os.path.join(_ROOT, "vagen/configs/vagen_multiturn.yaml"))
        .trainer.total_epochs)

    per_epoch = n_envs // batch          # drop_last=True
    assert per_epoch >= 1, (
        f"train_batch_size={batch} exceeds n_envs={n_envs}, so drop_last discards every "
        f"batch and the run does zero steps")
    assert epochs * per_epoch >= steps, (
        f"{os.path.basename(script)} asks for {steps} steps but can only reach "
        f"{epochs * per_epoch} ({per_epoch} steps/epoch x {epochs} epochs): "
        f"n_envs={n_envs}, train_batch_size={batch}")
