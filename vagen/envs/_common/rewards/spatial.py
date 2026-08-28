"""Score predicted spatial relations against environment ground truth.

The agent describes where things are in natural language; a judge turns that into a list
of ``{"object_id", "vertical_relation", "horizontal_relation"}`` items, and this scores
that list against what the environment actually contained.

Items are matched optimally rather than positionally: two boxes described in the other
order are the same description, and a greedy left-to-right pairing would penalise it.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

Item = dict


def exact_relation_match(pred: Item, gold: Item) -> float:
    """1.0 when both relations agree, 0.5 when one does, 0.0 otherwise.

    Partial credit is deliberate: getting "the box is below me" right while missing the
    column is more than nothing, and an all-or-nothing score would make the reward
    almost binary at the item level.
    """
    got = 0
    for axis in ("vertical_relation", "horizontal_relation"):
        p, g = pred.get(axis), gold.get(axis)
        got += 1 if p is not None and p == g else 0
    return got / 2.0


def f1(
    predicted: Iterable[Item],
    gold: Iterable[Item],
    similarity: Callable[[Item, Item], float] = exact_relation_match,
) -> float:
    """F1 over an optimal one-to-one matching of predicted items to gold items.

    Both empty counts as a perfect description -- there was nothing to say and nothing
    was said -- rather than as a failure.
    """
    predicted, gold = list(predicted), list(gold)
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0

    matched = _max_matching_score(predicted, gold, similarity)
    precision = matched / len(predicted)
    recall = matched / len(gold)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _max_matching_score(predicted, gold, similarity) -> float:
    """Total similarity of the best one-to-one pairing."""
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        scores = np.array([[similarity(p, g) for g in gold] for p in predicted])
        rows, cols = linear_sum_assignment(-scores)
        return float(scores[rows, cols].sum())
    except ImportError:  # pragma: no cover - scipy is a hard dependency of verl
        # Greedy fallback: worse matchings, never a crash mid-rollout.
        remaining, total = list(range(len(gold))), 0.0
        for p in predicted:
            if not remaining:
                break
            best = max(remaining, key=lambda j: similarity(p, gold[j]))
            total += similarity(p, gold[best])
            remaining.remove(best)
        return total


def grouped_f1(predicted: Iterable[Item], gold: Iterable[Item], weights: dict[str, float]) -> float:
    """Weighted mean of the per-object-type F1 scores.

    Scored per type rather than over one pooled list so that describing five boxes
    perfectly cannot compensate for missing the single target -- the two carry different
    amounts of information about the task.

    ★ A type absent from both the description and the scene is left out of the mean
    rather than scored 1.0. Counting it would pay the agent for staying silent about
    things that were never there, which is free credit available in any scene missing an
    object type. Hallucinating one still costs, since the type is then present in the
    description.
    """
    predicted, gold = list(predicted), list(gold)

    scored, total = 0.0, 0.0
    for object_id, weight in weights.items():
        p = [i for i in predicted if i.get("object_id") == object_id]
        g = [i for i in gold if i.get("object_id") == object_id]
        if not p and not g:
            continue
        scored += weight * f1(p, g)
        total += weight

    # Nothing to describe and nothing described: a correct, if empty, account.
    return scored / total if total else 1.0
