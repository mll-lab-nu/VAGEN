"""The surface a new advantage estimator is written against.

Covers the plumbing rather than any particular algorithm: what arrives in
``AdvantageInputs``, what ``AdvantageOutputs`` does with the masks, and the two places
verl has to cooperate for either to reach a loss.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from verl.trainer.ppo.core_algos import get_adv_estimator_fn

from vagen.algorithms import (
    AdvantageInputs,
    AdvantageOutputs,
    advantage_estimator,
    spans_rows,
)


@pytest.fixture(autouse=True)
def _restore_the_registries():
    """Registration mutates three module-level containers. These tests register probe
    estimators, so without this the very next test to assert on the registry contents
    fails -- and only when the files run in that order, which is the worst kind."""
    from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY

    from vagen.algorithms import (
        SENTINEL_RETURN_ESTIMATORS,
        TRAJECTORY_ESTIMATORS,
    )

    snapshots = [
        (ADV_ESTIMATOR_REGISTRY, dict(ADV_ESTIMATOR_REGISTRY)),
        (TRAJECTORY_ESTIMATORS, set(TRAJECTORY_ESTIMATORS)),
        (SENTINEL_RETURN_ESTIMATORS, set(SENTINEL_RETURN_ESTIMATORS)),
    ]
    yield
    for live, saved in snapshots:
        live.clear()
        live.update(saved)


class _Cfg(dict):
    gamma, lam = 1.0, 1.0

    def get(self, key, default=None):
        return getattr(self, key, default)


def _batch(**extra):
    n, width = 2, 3
    base = {
        "token_level_scores": torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]),
        "response_mask": torch.ones(n, width, dtype=torch.long),
    }
    base.update(extra)
    return TensorDict(base, batch_size=[n])


def _nt(n=2):
    return {
        "group_idx": np.array(["g"] * n, dtype=object),
        "traj_idx": np.array(list(range(n))),
        "turn_idx": np.zeros(n, dtype=np.int64),
        "episode_id": np.array([f"ep{i}" for i in range(n)], dtype=object),
        "conversation_id": np.zeros(n, dtype=np.int64),
    }


def _inputs(batch=None, non_tensor=None, config=None, name="probe"):
    # `batch or _batch()` would call __bool__ on the TensorDict, which raises. Every
    # container here is checked against None explicitly for the same reason.
    return AdvantageInputs(
        _batch() if batch is None else batch,
        _nt() if non_tensor is None else non_tensor,
        _Cfg() if config is None else config,
        name,
    )


# ------------------------------------------------------------------------- the inputs


def test_rewards_prefers_the_kl_penalised_tensor():
    """★ verl writes the KL-penalised reward into `token_level_rewards` and leaves
    `token_level_scores` untouched. Reading the scores makes use_kl_in_reward a silent
    no-op: the penalty is computed, stored, and never read."""
    inputs = _inputs(_batch(token_level_rewards=torch.full((2, 3), -1.0)))
    assert torch.all(inputs.rewards == -1.0)
    assert torch.any(inputs.scores != -1.0), "scores must stay the raw, un-penalised ones"


def test_rewards_falls_back_to_scores_when_there_is_no_penalty():
    assert torch.equal(_inputs().rewards, _inputs().scores)


def test_values_are_zeros_without_a_critic():
    """So a critic-free estimator can read `values` without guarding."""
    values = _inputs().values
    assert values.shape == (2, 3) and torch.all(values == 0)


def test_optional_tensors_are_none_when_absent():
    inputs = _inputs()
    assert inputs.old_log_probs is None
    assert inputs.ref_log_probs is None
    assert inputs.rollout_log_probs is None
    assert inputs.kl() is None, "no reference policy means no KL to report"


def test_ref_log_probs_reads_verls_singular_key():
    """★ verl spells it `ref_log_prob` while the actor's is `old_log_probs`. Reading the
    plural for both silently returns None and the KL term vanishes."""
    inputs = _inputs(_batch(ref_log_prob=torch.full((2, 3), -0.5)))
    assert inputs.ref_log_probs is not None
    assert torch.all(inputs.ref_log_probs == -0.5)


def test_kl_is_computed_per_token_and_masked_to_responses():
    """verl stores only `token_level_rewards` and a scalar metric, so an estimator that
    wants per-token KL has to recompute it -- this is that."""
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long)
    inputs = _inputs(
        _batch(
            response_mask=mask,
            old_log_probs=torch.full((2, 3), -1.0),
            ref_log_prob=torch.full((2, 3), -2.0),
        )
    )
    kl = inputs.kl()
    assert kl.shape == (2, 3)
    assert float(kl[0, 2]) == 0.0, "a non-response position must not contribute KL"
    assert float(kl[0, 0]) == pytest.approx(1.0), "kl of a 1.0 log-prob gap"


def test_identity_columns_are_reachable_by_name():
    inputs = _inputs()
    assert list(inputs.group_idx) == ["g", "g"]
    assert list(inputs.traj_idx) == [0, 1]
    assert list(inputs.episode_id) == ["ep0", "ep1"]
    assert list(inputs.conversation_id) == [0, 0]
    assert inputs.uid is None, "absent columns read as None rather than raising"


def test_conversation_id_falls_back_to_turn_idx():
    """Concat emits one and not always the other; an estimator should not have to care."""
    non_tensor = _nt()
    del non_tensor["conversation_id"]
    assert list(_inputs(non_tensor=non_tensor).conversation_id) == [0, 0]


def test_view_is_built_once():
    inputs = _inputs()
    assert inputs.view is inputs.view


# -------------------------------------------------------------------- hyperparameters


def test_required_param_names_the_estimator_and_the_plus_syntax():
    """★ verl's AlgoConfig has a fixed field list, so a new knob needs hydra's `+`.
    The error has to say so, or the next step is a confusing override failure."""
    with pytest.raises(ValueError) as exc:
        _inputs(name="my_algo").required_param("beta", "It sets the thing.")
    message = str(exc.value)
    assert "my_algo" in message and "algorithm.beta" in message
    assert "+algorithm.beta=" in message, "the message must show hydra's append syntax"
    assert "It sets the thing." in message


def test_required_param_rejects_an_explicit_none():
    """`+algorithm.beta=null` is a missing value, not a value."""
    cfg = _Cfg()
    cfg.beta = None
    with pytest.raises(ValueError, match="algorithm.beta"):
        _inputs(config=cfg).required_param("beta", "why")


def test_optional_param_uses_the_default():
    assert _inputs().param("beta", 0.25) == 0.25


# ------------------------------------------------------------------------ the outputs


def test_a_bare_tuple_is_accepted():
    """The verl-native return. An estimator that needs no masks writes no boilerplate."""

    @advantage_estimator("probe_tuple")
    def _probe(inputs):
        return inputs.zeros(), inputs.zeros() + 1.0

    adv, ret = get_adv_estimator_fn("probe_tuple")(
        batch=_batch(), non_tensor_batch=_nt(), config=_Cfg()
    )
    assert torch.all(adv == 0) and torch.all(ret == 1.0)


def test_masks_are_written_into_the_batch():
    """★ verl's return contract is exactly two tensors, so a mask can only travel as a
    side channel in the batch. `batch` is the same object verl holds."""

    @advantage_estimator("probe_masks")
    def _probe(inputs):
        return AdvantageOutputs(
            advantages=inputs.zeros(),
            returns=inputs.zeros(),
            value_mask=torch.tensor([[1, 0, 0], [1, 0, 0]]),
            extra={"probe_diag": torch.ones(2, 3)},
        )

    batch = _batch()
    get_adv_estimator_fn("probe_masks")(batch=batch, non_tensor_batch=_nt(), config=_Cfg())

    assert batch["value_mask"].tolist() == [[1, 0, 0], [1, 0, 0]]
    assert torch.all(batch["probe_diag"] == 1.0)


def test_no_mask_means_no_key():
    """Absent must mean "every response token", not "a mask of ones" -- an unconditional
    write would make the optional key mandatory for every downstream reader."""

    @advantage_estimator("probe_nomask")
    def _probe(inputs):
        return AdvantageOutputs(advantages=inputs.zeros(), returns=inputs.zeros())

    batch = _batch()
    get_adv_estimator_fn("probe_nomask")(batch=batch, non_tensor_batch=_nt(), config=_Cfg())
    assert "value_mask" not in batch.keys()


def test_registering_declares_the_estimator_spans_rows():
    """Which is what lets it run under no_concat and compact."""

    @advantage_estimator("probe_spans")
    def _probe(inputs):
        return inputs.zeros(), inputs.zeros()

    assert spans_rows("probe_spans") is True


def test_sentinel_returns_is_declared_at_registration():
    from vagen.algorithms import needs_value_mask

    @advantage_estimator("probe_sentinel", sentinel_returns=True)
    def _probe(inputs):
        return inputs.zeros(), inputs.zeros()

    assert needs_value_mask("probe_sentinel") is True
    assert spans_rows("probe_sentinel") is True, "sentinel estimators span rows too"


def test_the_adapter_does_not_hide_its_signature_from_verl():
    """★ The trap. `functools.wraps` sets `__wrapped__`; `inspect.signature` follows it
    to the inner function, which takes `inputs` and names neither container -- and verl
    decides what to pass by reading exactly that signature. The estimator would then be
    handed no batch at all."""

    @advantage_estimator("probe_signature")
    def _probe(inputs):
        return inputs.zeros(), inputs.zeros()

    fn = get_adv_estimator_fn("probe_signature")
    params = inspect.signature(fn).parameters
    assert "batch" in params and "non_tensor_batch" in params
    assert not hasattr(fn, "__wrapped__"), "__wrapped__ would redirect inspect.signature"


# -------------------------------------------------------- verl has to cooperate twice


def test_the_critic_loss_honours_value_mask():
    """One of VAGEN's verl patches. Without it a sentinel return is a regression target."""
    from verl.workers.utils import losses

    src = inspect.getsource(losses)
    assert '"value_mask" in data.keys()' in src, "the critic no longer looks for it"
    assert 'response_mask & data["value_mask"].to(bool)' in src


def test_there_is_no_actor_side_mask_and_that_is_deliberate():
    """★ A `policy_mask` was written and then removed, because it could not do what it
    claimed. `compute_policy_loss_vanilla` divides by `batch_num_tokens`, which the engine
    computes as `data["loss_mask"].sum()` all-reduced across DP ranks *before* the loss
    runs -- so a mask applied inside the loss shrinks only the numerator and, under the
    default `token-mean`, is arithmetically identical to zeroing the advantage.

    Pinned so nobody re-adds it on the same false premise. The critic side is genuinely
    different: `compute_value_loss` does not forward `global_batch_info`, so its
    denominator *is* recomputed from the narrowed mask.
    """
    from verl.trainer.ppo import core_algos
    from verl.workers.utils import losses

    assert "policy_mask" not in inspect.getsource(losses)
    assert not hasattr(AdvantageOutputs(advantages=None, returns=None), "policy_mask")

    # The asymmetry that makes value_mask work and a policy_mask not.
    assert "global_batch_info" in inspect.getsource(core_algos.compute_policy_loss_vanilla)
    assert "global_batch_info" not in inspect.getsource(core_algos.compute_value_loss)


def test_a_zeroed_advantage_does_not_survive_whitening():
    """★ Why "just zero the advantage" is not a way to exclude a token.

    Every estimator finishes with `masked_whiten`, which is affine: an advantage set to
    exactly 0.0 at a mask-1 position comes out as `-mean/std`, an ordinary non-zero
    coefficient. A mask is a 0/1 declaration and survives; a float magnitude does not.
    """
    import verl.utils.torch_functional as verl_F

    adv = torch.tensor([[3.0, 3.0, 0.0, 0.0]])
    mask = torch.ones(1, 4)
    whitened = verl_F.masked_whiten(adv, mask) * mask

    assert float(whitened[0, 2]) != pytest.approx(0.0, abs=1e-6), (
        "the deliberate zero was renormalised into a real gradient coefficient"
    )
    # Taking it out of the mask is what actually excludes it.
    mask_out = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    assert float((verl_F.masked_whiten(adv, mask_out) * mask_out)[0, 2]) == 0.0


def test_entropy_and_kl_ignore_the_advantage_entirely():
    """The other half: even an advantage that did stay zero would not stop the token being
    learned. Both extra terms are aggregated over `response_mask` alone."""
    from verl.workers.utils import losses

    src = inspect.getsource(losses.ppo_loss)
    assert "loss_mat=entropy, loss_mask=response_mask" in src
    assert "loss_mat=kld, loss_mask=response_mask" in src


def test_response_mask_is_the_lever_that_reaches_the_denominator():
    """`loss_mask` is `response_mask`, and `batch_num_tokens` is its sum -- so dropping a
    token there removes it from numerator and denominator alike."""
    from verl.experimental.separation import ray_trainer
    from verl.workers.engine.fsdp import transformer_impl

    assert "loss_mask=response_masks" in inspect.getsource(ray_trainer)
    assert 'batch_num_tokens = data["loss_mask"].sum()' in inspect.getsource(transformer_impl)
