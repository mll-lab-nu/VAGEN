"""One episode, rendered as one thing you can read.

Three levels, and the words for them:

* **episode** -- one whole agent/environment interaction, reset to terminal or turn cap.
  Identified by ``(group_idx, traj_idx)``.
* **conversation** -- one continuous exchange with the model.
* **turn** -- one model call within a conversation.

The context policy is exactly what shape an episode takes:

    concat      1 conversation per episode,  many turns in it
    no_concat   many conversations,          1 turn each
    compact     several conversations,       many turns each

"trajectory" is deliberately not one of these words: ``traj_idx`` upstream already means
*which sampled rollout of a prompt*, which is the axis that identifies the episode -- so
reusing it for the segment between compactions would collide with the thing doing the
identifying.

The validation table logs a row per turn, which is the wrong unit for looking at an
agent under any of the three policies. This reassembles them: turns in order, frames
inline where the agent saw them, and a visible seam wherever the conversation restarted.
"""

from __future__ import annotations

import base64
import html
import io
import re
from collections import defaultdict
from typing import Any

# Enough to see the board, small enough that a table of them still loads.
_MAX_W = 320
# A distinct value no conversation id can equal. Not `object()` inline: that builds a new
# one on every comparison, so "have we seen a conversation yet" is always false and the
# first turn of every episode is reported as a compaction.
_UNSET = object()
_STYLE = (
    "font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;"
    "line-height:1.45;white-space:pre-wrap;word-break:break-word"
)


def _img_tag(image: Any) -> str:
    """A PIL image as an inline <img>, or nothing if it cannot be encoded.

    Downscaled before encoding, not just styled down. The width below used to be a CSS
    attribute only, so the payload was whatever the environment rendered -- invisible at
    sokoban's 192px and a per-frame cost proportional to resolution for anything larger.
    """
    try:
        frame = image.convert("RGB")
        if frame.width > _MAX_W:
            height = max(1, round(frame.height * _MAX_W / frame.width))
            frame = frame.resize((_MAX_W, height))
        buf = io.BytesIO()
        frame.save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode()
    except Exception:  # noqa: BLE001 - a frame that will not encode must not lose the text
        return ""
    return f'<img src="data:image/png;base64,{b64}" style="max-width:{_MAX_W}px;display:block;margin:6px 0">'


def group_turns(rows: list[dict]) -> dict[tuple, list[dict]]:
    """Group rows into episodes and order each one by turn.

    Keyed on ``episode_id``, which the agent loop mints once per episode and stamps on
    every row alongside ``conversation_id`` and ``turn_idx``. The three travel together
    so they cannot disagree about what they identify.

    ``(group_idx, traj_idx)`` is the fallback, and used to be the primary key -- but that
    is the dataset's axis rather than the loop's, and measured at validation it was
    unique per row, so every row grouped as its own one-turn episode. Rows with neither
    fall back to their own position, which logs one entry per row rather than silently
    collapsing everything into a single bogus episode.
    """
    episodes: dict[tuple, list[dict]] = defaultdict(list)
    for i, r in enumerate(rows):
        ep = r.get("episode_id")
        if ep is not None:
            key = ("ep", ep)
        else:
            g, t = r.get("group_idx"), r.get("traj_idx")
            key = (g, t) if g is not None and t is not None else ("_row", i)
        episodes[key].append(r)
    for turns in episodes.values():
        turns.sort(key=lambda r: (r.get("turn_idx") if r.get("turn_idx") is not None else 0))
    return dict(episodes)


def episode_html(turns: list[dict]) -> str:
    """The episode as it was spoken, with one thing added: where a conversation starts.

    Nothing else is labelled. The decoded text already carries the template's own
    ``system`` / ``user`` / ``assistant`` markers, so adding our own put two of each on
    the screen -- and ours were a guess at what the block contained, while the
    template's are what the model actually read. Turn boundaries are visible in the text
    for the same reason.

    A conversation boundary is the exception: it is the one thing the text cannot show,
    because from inside the transcript a compaction looks like a model that forgot.

    Padding never appears: the merge decodes each span from the real tokens.
    """
    parts = [f'<div style="{_STYLE}">']
    first = turns[0]
    conversations = first.get("conversations")

    if not conversations:
        # Pre-merge callers: one row per turn, no conversation structure to recover.
        conversations = [{
            "conversation_id": 0,
            "prompt": first.get("input") or "",
            "prompt_image": None,
            "turns": [
                {"turn_id": i, "response": t.get("output") or "", "observation": "",
                 "observation_image": None}
                for i, t in enumerate(turns)
            ],
        }]

    for n, conversation in enumerate(conversations):
        if n:
            parts.append('<hr style="border:0;border-top:2px solid #c00;margin:18px 0 6px">')
        parts.append(
            f'<div style="color:#c00;font-size:13px"><b>conversation '
            f'{conversation.get("conversation_id", n)}</b></div>'
        )
        _text(parts, conversation.get("prompt"), conversation.get("prompt_image"))

        for turn in conversation.get("turns", []):
            _text(parts, turn.get("response"))
            _text(parts, turn.get("observation"), turn.get("observation_image"))

    parts.append("</div>")
    return "".join(parts)


#: The template's cue for the decoder to start writing, at the end of a rendered turn.
#: Real tokens in the sequence, carried at mask 0 -- so they are shown, not hidden. What
#: they are useful for here is placement: a frame belongs before the cue, since it is
#: part of what the model was shown, and the cue is where its own writing begins.
_GENERATION_CUE = re.compile(r"((?:\n)?(?:assistant|model)\n?)\s*$", re.IGNORECASE)


def _text(parts: list[str], text, frame=None) -> None:
    """One block of the transcript, with its frame placed inside it.

    Nothing is removed. The block is only split where the decoder cue begins, so the
    frame lands at the end of what the model was shown rather than after the marker that
    starts its reply -- which read as though the picture were the assistant's own.
    """
    if not text:
        if frame is not None:
            parts.append(_img_tag(frame))
        return
    body = str(text)
    cue = ""
    match = _GENERATION_CUE.search(body)
    if match and frame is not None:
        body, cue = body[: match.start()], match.group(1)
    if body:
        parts.append(f'<div>{html.escape(body)}</div>')
    if frame is not None:
        parts.append(_img_tag(frame))
    if cue:
        parts.append(f'<div>{html.escape(cue)}</div>')


def episode_rows(rows: list[dict]) -> list[dict]:
    """One record per episode: its html, and the numbers worth sorting a table by."""
    out = []
    for key, turns in group_turns(rows).items():
        scores = [t["score"] for t in turns if t.get("score") is not None]
        successes = [t["traj_success"] for t in turns if t.get("traj_success") is not None]
        sources = [t.get("data_source") for t in turns if t.get("data_source")]
        out.append(
            {
                "episode": str(key[1]) if key[0] == "ep" else f"{key[0]}/{key[1]}",
                "data_source": sources[0] if sources else None,
                # Turns the episode ran, as the loop counted them. len(turns) is the
                # number of rows, which in concat mode is 1 for any episode however long.
                "turns": next((t["episode_turns"] for t in turns if t.get("episode_turns")), len(turns)),
                "reward": round(sum(scores), 4) if scores else None,
                # An episode spanning more than one conversation was compacted. After
                # the validation merge a row is a whole episode, so the per-turn
                # conversation ids are gone and the count comes across pre-computed.
                "conversations": next(
                    (t["n_conversations"] for t in turns if t.get("n_conversations")),
                    len({t.get("conversation_id") for t in turns}),
                ),
                "score": round(sum(scores), 4) if scores else None,
                "success": max(successes) if successes else None,
                "html": episode_html(turns),
            }
        )
    out.sort(key=lambda r: r["episode"])
    return out


#: How to choose which episodes get logged. A name from here goes in the config; the
#: default shows successes and failures side by side, because a log of failures alone
#: has nothing to compare against and one of successes alone hides what is going wrong.
#: Each takes the episodes of one validation round and returns them in preference order;
#: `select_episodes` then takes as many as asked for.
def _by_reward(episodes, *, worst_first):
    """Ranked by reward, with the reward-less always last.

    ``reverse=True`` would flip the "has no reward" component of the key too, so under
    "best" the episodes with no score at all sorted first -- the opposite of the ask.
    """
    sign = 1.0 if worst_first else -1.0
    return sorted(episodes, key=lambda e: (e.get("reward") is None, sign * (e.get("reward") or 0.0)))


SELECTORS = {
    # Alternating succeeded / failed, spread across data sources. See select_episodes.
    "balanced": None,                                       # handled inline
    "first": lambda eps: list(eps),
    "failures": lambda eps: [e for e in eps if not e.get("success")],
    "successes": lambda eps: [e for e in eps if e.get("success")],
    "worst": lambda eps: _by_reward(eps, worst_first=True),
    "best": lambda eps: _by_reward(eps, worst_first=False),
}


def select_episodes(
    episodes: list[dict],
    n: int,
    strategy: str = "balanced",
    success_ratio: float = 0.5,
) -> list[dict]:
    """Pick ``n`` episodes to log, by the named strategy.

    ``balanced`` (the default) alternates succeeded and failed in the proportion
    ``success_ratio``, spread across data sources. Taking whatever sorted first is not
    a sample: at a 12% success rate it is n failures, and a log with nothing to compare
    against teaches nothing.

    A class that runs out simply stops contributing and the others fill the gap, so a
    run with no successes yet still logs ``n`` episodes rather than a half-empty table.

    Other strategies are for looking at something specific -- ``failures`` while chasing
    a format problem, ``worst`` while chasing a reward bug -- and are selected by name
    from ``SELECTORS`` rather than by editing this.
    """
    if n <= 0 or not episodes:
        return []
    if strategy != "balanced":
        try:
            chosen = SELECTORS[strategy](episodes)
        except KeyError:
            raise ValueError(
                f"unknown val log strategy {strategy!r}; choose from {sorted(SELECTORS)}"
            ) from None
        return chosen[:n]

    want_success = max(0, min(n, round(n * success_ratio)))
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for e in episodes:
        buckets[(e.get("data_source"), bool(e.get("success")))].append(e)

    # Deterministic order so the table reads the same way step to step.
    order = sorted(buckets, key=lambda k: (str(k[0]), not k[1]))
    quota = {True: want_success, False: n - want_success}
    picked: list[dict] = []
    for pass_respects_quota in (True, False):
        while len(picked) < n:
            took = False
            for key in order:
                if not buckets[key]:
                    continue
                if pass_respects_quota and quota[key[1]] <= 0:
                    continue
                picked.append(buckets[key].pop(0))
                quota[key[1]] -= 1
                took = True
                if len(picked) == n:
                    break
            if not took:
                break   # this pass can give no more; the second ignores the quota
    picked.sort(key=lambda r: (str(r.get("data_source")), not bool(r.get("success")), r["episode"]))
    return picked


def describe_columns(extras: dict, n_rows: int) -> str:
    """What actually arrived. The grouping depends on these and fails silently without
    them: every row becomes its own episode, so a five-turn episode reads as five
    one-turn ones -- or as one, once only n are shown."""
    bits = []
    for key in ("episode_id", "group_idx", "turn_idx", "conversation_id", "episode_turns",
                "n_conversations", "conversations"):
        vals = extras.get(key) or []
        present = sum(1 for v in vals if v is not None)
        bits.append(f"{key}={present}/{n_rows}")
    return " ".join(bits)


def rows_from_validation(inputs, outputs, scores, images, extras) -> list[dict]:
    """One record per turn, from the parallel columns validation produces.

    Here rather than in the trainer: the trainer's part is deciding to log, not knowing
    the shape of a validation batch.
    """
    n = len(outputs)

    def col(name):
        v = list(extras.get(name) or [])
        return v + [None] * (n - len(v))

    return [
        {
            "input": inp,
            "output": out,
            "score": sc,
            "images": im,
            "episode_id": ep,
            "group_idx": g,
            "traj_idx": t,
            "turn_idx": ti,
            "conversation_id": c,
            # The environment's own verdict, forwarded among the reward extras. Not
            # reading it here left the column present and always empty, which reads as
            # "no episode succeeded" rather than "this was never wired up".
            "traj_success": su,
            "data_source": ds,
            # Turns the episode really ran. The per-row num_turns is 1 by construction,
            # and in concat mode an episode is one row, so without this every episode
            # looks like a single turn no matter how long it was.
            "episode_turns": et,
            "n_conversations": nc,
            "conversations": tr,
        }
        for inp, out, sc, im, ep, g, t, ti, c, su, ds, et, nc, tr in zip(
            inputs, outputs, scores, images or [None] * n,
            col("episode_id"), col("group_idx"), col("traj_idx"), col("turn_idx"),
            col("conversation_id"), col("traj_success"), col("data_source"), col("episode_turns"),
            col("n_conversations"), col("conversations"),
            strict=True,
        )
    ]
