"""One episode, rendered as one thing you can read.

The validation table logs a row per model call. That is the wrong unit for looking at an
agent: an episode is several calls, and when the context is compacted it is several
*conversations*, so the same trajectory arrives as unrelated rows with no indication that
they belong together or in what order.

This reassembles them -- turns in order, frames inline where the agent saw them, and a
visible seam wherever the conversation restarted.
"""

from __future__ import annotations

import base64
import html
import io
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

    Keyed by (group_idx, traj_idx). Rows missing those fall back to their own position,
    so a loop that publishes no episode ids still logs one row per call rather than
    silently collapsing everything into a single bogus episode.
    """
    episodes: dict[tuple, list[dict]] = defaultdict(list)
    for i, r in enumerate(rows):
        g, t = r.get("group_idx"), r.get("traj_idx")
        key = (g, t) if g is not None and t is not None else ("_row", i)
        episodes[key].append(r)
    for turns in episodes.values():
        turns.sort(key=lambda r: (r.get("turn_idx") if r.get("turn_idx") is not None else 0))
    return dict(episodes)


def episode_html(turns: list[dict]) -> str:
    """Render one episode: prompt, then each turn's frames and text, in order."""
    parts = [f'<div style="{_STYLE}">']
    prev_conversation = _UNSET

    first = turns[0]
    if first.get("input"):
        parts.append('<div style="color:#666"><b>prompt</b></div>')
        parts.append(f'<div style="color:#444">{html.escape(str(first["input"]))}</div>')

    for position, turn in enumerate(turns):
        conversation = turn.get("conversation_id")
        if conversation != prev_conversation:
            if prev_conversation is not _UNSET:
                # The context was compacted: what follows is a fresh conversation that
                # continues the same episode. Without the seam the transcript reads as a
                # model that inexplicably forgot everything.
                parts.append(
                    '<hr style="border:0;border-top:2px dashed #c00;margin:12px 0">'
                    '<div style="color:#c00"><b>— context compacted, conversation restarted —</b></div>'
                )
            prev_conversation = conversation

        # Position within the episode when the loop did not label the turn. The label is
        # for reading; ordering still depends on turn_idx, and a batch that lost it is a
        # separate problem that this must not paper over -- so an unlabelled turn is
        # marked, rather than silently numbered as though it were known.
        n = turn.get("turn_idx")
        head = f"turn {n}" if n is not None else f"turn {position} (unlabelled)"
        bits = []
        for key, label in (("score", "score"), ("traj_success", "success")):
            if turn.get(key) is not None:
                bits.append(f"{label}={turn[key]}")
        parts.append(f'<div style="color:#06c;margin-top:10px"><b>{head}</b> {" ".join(bits)}</div>')

        for image in turn.get("images") or []:
            tag = _img_tag(image)
            if tag:
                parts.append(tag)
        parts.append(f'<div>{html.escape(str(turn.get("output") or ""))}</div>')

    parts.append("</div>")
    return "".join(parts)


def episode_rows(rows: list[dict]) -> list[dict]:
    """One record per episode: its html, and the numbers worth sorting a table by."""
    out = []
    for key, turns in group_turns(rows).items():
        scores = [t["score"] for t in turns if t.get("score") is not None]
        successes = [t["traj_success"] for t in turns if t.get("traj_success") is not None]
        sources = [t.get("data_source") for t in turns if t.get("data_source")]
        out.append(
            {
                "episode": f"{key[0]}/{key[1]}",
                "data_source": sources[0] if sources else None,
                "turns": len(turns),
                # An episode spanning more than one conversation was compacted.
                "conversations": len({t.get("conversation_id") for t in turns}),
                "score": round(sum(scores), 4) if scores else None,
                "success": max(successes) if successes else None,
                "html": episode_html(turns),
            }
        )
    out.sort(key=lambda r: r["episode"])
    return out


def select_episodes(episodes: list[dict], n: int) -> list[dict]:
    """Pick ``n`` to log: successes and failures in balance, spread across sources.

    Taking the first ``n`` shows whatever sorted first, and at a 12% success rate that
    is eight failures -- a log you cannot learn anything from, because there is nothing
    to compare a failure against. Half and half is the useful sample.

    Round-robin over (source, succeeded) buckets. A bucket that runs out simply drops
    out and the others fill the gap, so a run with no successes yet still logs ``n``
    episodes rather than half a table.
    """
    if n <= 0 or not episodes:
        return []

    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for e in episodes:
        buckets[(e.get("data_source"), bool(e.get("success")))].append(e)

    # Deterministic order so the table reads the same way step to step: succeeded first
    # within a source, sources in name order.
    order = sorted(buckets, key=lambda k: (str(k[0]), not k[1]))
    picked: list[dict] = []
    while len(picked) < n:
        took = False
        for key in order:
            if not buckets[key]:
                continue
            picked.append(buckets[key].pop(0))
            took = True
            if len(picked) == n:
                break
        if not took:  # every bucket exhausted
            break
    picked.sort(key=lambda r: (str(r.get("data_source")), not bool(r.get("success")), r["episode"]))
    return picked
