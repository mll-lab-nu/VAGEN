"""Is the judge turning descriptions into the right relations?

    python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-4B-Instruct-2507 \\
        --port 8123 --gpu-memory-utilization 0.35 --max-model-len 4096 &
    JUDGE_URL=http://127.0.0.1:8123/v1 python tools/judge_eval.py

Measured on Qwen3-4B-Instruct-2507, 2026-08-11: **13/14 exact, 6/7 on the "same" cases**.
The single miss is genuinely ambiguous -- "level with me" states the row and not the
column, and the judge filled the column in as "same" where the prompt asks for null when
a relation is not stated. The judge is not the weak link.

The model says "same" in 1.4% of relation words; the ground truth needs it 39.4% of the
time. One explanation is that the judge mis-parses the "same row"/"same column" phrasing
the prompt's own examples use -- the model would then be punished for saying "same" and
would learn not to. This measures that directly.
"""
import asyncio, json, os, sys
from vagen.rewards.judge import StructuredJudge
from vagen.envs.sokoban.state_reward_spec import JUDGE_PROMPT

B = lambda v, h: {"object_id": "box", "vertical_relation": v, "horizontal_relation": h}
T = lambda v, h: {"object_id": "target", "vertical_relation": v, "horizontal_relation": h}

CASES = [
  # --- the dominant template (82.8% of real observations)
  ("The box is below and right of the player, and the target is above and left of the player", [B("below","right"), T("above","left")]),
  ("The box is above and left of the player, and the target is below and right of the player", [B("above","left"), T("below","right")]),
  # --- ★ the "same" phrasings, which are the whole question
  ("The box is below and same column of the player, and the target is above and same column of the player", [B("below","same"), T("above","same")]),
  ("The box is same row and left of the player, and the target is same row and right of the player", [B("same","left"), T("same","right")]),
  ("The box is same row and same column of the player", [B("same","same")]),
  ("The box is below and same column of the player", [B("below","same")]),
  ("The target is same row and left of the player", [T("same","left")]),
  # --- the phrasings the system prompt's own EXAMPLES teach
  ("A box is directly below me in my column, and the target is also below me, in that same column.", [B("below","same"), T("below","same")]),
  ("The box will still be below me in my column, and I will have moved onto the target's row, so the target will be level with me.", [B("below","same"), T("same",None)]),
  # --- tail variants seen in real rollouts
  ("The box is below and left of the player, and the target is below of the player", [B("below","left"), T("below",None)]),
  ("The box is above and right of the player", [B("above","right")]),
  # --- adversarial
  ("I am not sure where anything is.", []),
  ("The player is below and left of the box", [B("above","right")]),   # inverted framing; hard
  ("There are two boxes: one above and left, one below and right of the player", [B("above","left"), B("below","right")]),
]

def norm(items):
    if items is None: return None
    out = []
    for i in items:
        if not isinstance(i, dict): continue
        out.append((str(i.get("object_id","")).lower(),
                    i.get("vertical_relation"), i.get("horizontal_relation")))
    return sorted(out, key=str)

async def main():
    j = StructuredJudge(base_url=os.environ.get("JUDGE_URL","http://127.0.0.1:8123/v1"),
                        model=os.environ.get("JUDGE_MODEL","Qwen/Qwen3-4B-Instruct-2507"))
    got = await j.parse_batch([JUDGE_PROMPT.format(content=c[0]) for c in CASES])
    exact = same_ok = same_tot = 0
    print("=" * 100)
    for (text, gold), g in zip(CASES, got):
        ok = norm(g) == norm(gold)
        exact += ok
        has_same = any("same" in str(v) for it in gold for v in it.values())
        if has_same:
            same_tot += 1; same_ok += ok
        print(f"{'PASS' if ok else 'FAIL'} | {text[:76]}")
        if not ok:
            print(f"       gold: {norm(gold)}")
            print(f"       got : {norm(g)}")
    print("=" * 100)
    print(f"exact match          : {exact}/{len(CASES)}")
    print(f"cases involving same : {same_ok}/{same_tot}   <-- the one that matters")

asyncio.run(main())
