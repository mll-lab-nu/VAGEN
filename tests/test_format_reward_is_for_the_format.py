"""The format bonus must be paid for the format, not for a salvageable action.

``parse_*`` in every env is deliberately two-tier: ``format_correct`` is a strict regex
over the full tag sequence, while extraction is lenient so that a malformed turn still
yields its action and the episode keeps its data (measured: half of all base-model
episodes reached zero usable actions when extraction was strict).

That split only works if the *reward* reads the strict tier. Sokoban read the lenient one
-- it paid `format_reward` whenever an action could be salvaged -- so under
``prompt_format=wm`` a policy that emitted ``<answer>Left, Left</answer>`` and nothing
else collected exactly what a policy writing all four sections collected, while spending
~100 fewer tokens. Sokoban rollouts collapsed to precisely that string.

These tests are parametrised over the envs rather than written once for sokoban, because
the bug was an inconsistency between four implementations of the same idea.
"""

from __future__ import annotations

import pytest

WM_FULL = (
    "<perception>The box is below and right of the player</perception>"
    "<reasoning>I should move right</reasoning>"
    "<prediction>The box will be below and same column of the player</prediction>"
    "<answer>Right</answer>"
)
#: What the collapsed sokoban policy actually emitted, `<th` and all.
ANSWER_ONLY = "<answer>Right</answer><th"


# ------------------------------------------------------------------- the parsers agree


@pytest.mark.parametrize(
    "module,fmt",
    [
        ("vagen.envs.sokoban.utils.utils", "wm"),
        ("vagen.envs.frozenlake.utils.utils", "wm"),
    ],
)
def test_answer_only_is_not_format_correct(module, fmt):
    """★ The strict tier must reject it. If this ever passes, every reward gate built on
    `format_correct` silently stops gating."""
    import importlib

    parse = importlib.import_module(module).parse_response
    assert parse(WM_FULL, prompt_format=fmt)["format_correct"] is True
    assert parse(ANSWER_ONLY, prompt_format=fmt)["format_correct"] is False


def test_all_environment_parsers_share_the_same_salvage_policy():
    from vagen.envs.frozenlake.utils.utils import parse_response as fl_parse
    from vagen.envs.sokoban.utils.utils import parse_response as sk_parse

    assert [a.lower() for a in sk_parse(ANSWER_ONLY, prompt_format="wm")["actions"]] == ["right"]
    assert fl_parse(ANSWER_ONLY, prompt_format="wm")["actions"] == ["right"]
    assert fl_parse(ANSWER_ONLY, prompt_format="wm")["format_correct"] is False


@pytest.mark.parametrize(
    "module,separator",
    [
        ("vagen.envs.sokoban.utils.utils", ","),
        ("vagen.envs.frozenlake.utils.utils", ","),
        ("vagen.envs.primitive_skill.utils.parse", "|"),
        ("vagen.envs.navigation.utils.parse", "|"),
    ],
)
def test_every_wm_environment_accepts_the_same_canonical_order(module, separator):
    import importlib

    parse = importlib.import_module(module).parse_response
    response = WM_FULL.replace("Right", f"Right{separator}Left")
    parsed = parse(response, prompt_format="wm", action_sep=separator)
    assert parsed["format_correct"] is True
    assert [action.lower() for action in parsed["actions"]] == ["right", "left"]


def test_spatial_gym_uses_the_shared_free_think_protocol():
    from vagen.envs.spatial_gym.utils.utils import parse_llm_response

    reasoning, answer, correct = parse_llm_response(
        "<think>Map the room.</think><answer>Actions: [Rotate(90), Observe()]</answer>"
    )
    assert (reasoning, answer, correct) == (
        "Map the room.",
        "Actions: [Rotate(90), Observe()]",
        True,
    )
    assert parse_llm_response("THINK: map\nFINAL ANSWER: Actions: [Observe()]")[2] is False


def test_every_wm_prompt_prints_the_same_order():
    from vagen.envs.frozenlake.utils.prompt import format_prompt as frozenlake_prompt
    from vagen.envs.navigation.utils.prompt import get_format_instruction as navigation_prompt
    from vagen.envs.primitive_skill.utils.prompt import get_format_instruction as primitive_prompt
    from vagen.envs.sokoban.utils.prompt import format_prompt as sokoban_prompt

    prompts = [
        sokoban_prompt(3, ",", add_example=False, prompt_format="wm"),
        frozenlake_prompt(3, ",", add_example=False, prompt_format="wm"),
        primitive_prompt("wm", 2, "|"),
        navigation_prompt("wm", 5, "|"),
    ]
    for prompt in prompts:
        positions = [prompt.index(f"<{tag}>") for tag in ("perception", "reasoning", "prediction", "answer")]
        assert positions == sorted(positions)


def test_sokoban_strict_format_discards_the_salvaged_action():
    """The env-level gate: with `strict_format` the salvaged action must not be executed,
    so a turn that skipped <perception>/<reasoning>/<prediction> does nothing at all."""
    import inspect

    from vagen.envs.sokoban.sokoban_env import Sokoban, SokobanEnvConfig

    assert SokobanEnvConfig().strict_format is True, "the default must enforce the format"
    src = inspect.getsource(Sokoban.step)
    assert "strict_format" in src and "action_list = []" in src, (
        "the strict_format gate is gone from Sokoban.step; a malformed turn will act again"
    )


# --------------------------------------------------------------- the reward gate itself


def _gate_source(env_module: str, fn: str = "step") -> str:
    import importlib
    import inspect

    mod = importlib.import_module(env_module)
    env_cls = next(
        v for v in vars(mod).values()
        if isinstance(v, type) and v.__module__ == env_module and hasattr(v, fn)
        and not v.__name__.endswith("Config")
    )
    return inspect.getsource(getattr(env_cls, fn))


@pytest.mark.parametrize(
    "env_module",
    [
        "vagen.envs.sokoban.sokoban_env",
        "vagen.envs.frozenlake.frozenlake_env",
        "vagen.envs.spatial_gym.spatial_gym_env",
    ],
)
def test_the_format_reward_is_gated_on_format_correct(env_module):
    """★ The structural guard, over every env that pays a per-turn format reward.

    Asserted on the source rather than by running the env, because constructing a real gym
    env needs rendering and the property is about which flag the branch reads. The failure
    it catches -- `if self.valid_actions:` without the conjunct -- is one line and reads as
    correct.

    Parsed with `ast` rather than matched line by line: the sokoban guard became a
    multi-line condition when `strict_format` was folded into it, and a line-based version
    of this test silently started reading only `if self.valid_actions and (`.
    """
    import ast

    import textwrap

    tree = ast.parse(textwrap.dedent(_gate_source(env_module)))
    guards = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            not isinstance(statement, ast.If)
            and "format_reward" in ast.unparse(statement)
            for statement in node.body
        )
    ]
    assert guards, f"{env_module} no longer pays a format reward; update this test"
    for node in guards:
        cond = ast.unparse(node.test)
        assert "format_correct" in cond, (
            f"{env_module} pays format_reward under `{cond}`, which does not consult "
            "format_correct -- any response with a salvageable <answer> collects it"
        )


def test_primitive_skill_format_reward_requires_canonical_format():
    from vagen.envs.primitive_skill.utils.parse import compute_reward

    good = {"format_correct": True, "actions": ["MoveForward"]}
    bad = {"format_correct": False, "actions": ["MoveForward"]}
    assert compute_reward(good, ["MoveForward"], False, 0.0, format_reward=0.2) == 0.2
    assert compute_reward(bad, ["MoveForward"], False, 0.0, format_reward=0.2) == 0.0


def test_navigation_format_rewards_require_canonical_format():
    from vagen.envs.navigation.utils.parse import compute_reward

    good = {"format_correct": True}
    bad = {"format_correct": False}
    assert compute_reward(good, ["MoveAhead"], True, 0.2, 0.1, 1.0, True) == 1.3
    assert compute_reward(bad, ["MoveAhead"], True, 0.2, 0.1, 1.0, True) == 1.0


def test_strict_format_never_turns_off_the_format_reward_gate():
    import ast
    import inspect

    from vagen.envs.sokoban.sokoban_env import Sokoban

    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(Sokoban.step)))

    pay = next(n for n in ast.walk(tree)
               if isinstance(n, ast.If) and "format_reward" in ast.unparse(n.body))
    assert "format_correct" in ast.unparse(pay.test)
    assert "strict_format" not in ast.unparse(pay.test)
    drop = next(n for n in ast.walk(tree)
                if isinstance(n, ast.If) and "action_list = []" in ast.unparse(n.body))
    assert "strict_format" in ast.unparse(drop.test)
