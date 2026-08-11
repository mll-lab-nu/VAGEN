# VAGEN — orientation for an agent picking this up

Read this first; it is meant to take five minutes. `VAGEN_ARCH.md` has the design
rationale, and it is long. Anything here that disagrees with it, believe this.

Training backend: `verl@release/v0.8.0`, a **sibling checkout** at `../verl` — verl is
**not pip-installed**, so it has to be on `PYTHONPATH`.

```bash
PYTHONPATH=$(pwd)/../verl:$(pwd) python -m pytest -q     # 534 passed, 6 skipped
```

---

## 1. What this is

A multi-turn agentic RL layer over verl. An episode is an agent talking to an
environment; the question this codebase is organised around is **how that conversation is
laid out for training**, because that choice is what varies between experiments.

Three layouts, one axis — *when does a new conversation start?*

| mode | shape | one episode becomes |
|---|---|---|
| `concat` | one conversation, many turns | 1 training row |
| `no_concat` | many conversations, one turn each | N rows |
| `compact` | several conversations, several turns each | a few rows |

Set with `trainer.harness=concat|no_concat|compact`.

## 2. The layering, and the one rule that keeps it honest

```
run_episode          core/runner.py     one loop, for training and evaluation
  └ harness          core/harness.py    when to start a new conversation. No tokenizer,
                     harness/*.py       no env, no client -- text and messages only
  └ client           core/client.py     the ONLY layer that knows a token
                     agent_loop/verl_client.py
  └ env              core/env.py        observations in, actions out
```

**A conversation id is the whole protocol.** Passing one continues that conversation;
passing `None` starts a new one. That is why the same harness drives a verl rollout and a
closed chat API, and why the three modes are three points on one axis rather than three
mechanisms.

`Conversation` (`core/tape.py`) holds one conversation's tokens and which of them the
model produced. **One conversation is one training row.**

## 3. Invariants you can break silently

These have all been broken at least once, and none of them raised when they were.

1. **`multi_modal_data={"images": ...}` — plural.** Upstream renamed the key; the singular
   is silently ignored, the model gets image-pad tokens with no features, and Qwen skips
   the `masked_scatter` rather than complaining. The rollout still sees the pictures; only
   the model being optimised is blind.
2. **Placeholder blocks ↔ frames, strictly 1:1, and the block is
   `vision_start .. vision_end`, not the run.** `get_rope_index` counts images by reading
   the token *after* every `vision_start`, so a run that lost its sentinel is laid out as
   text and every later position shifts — while the counts still agree.
   `multi_modal_inputs` is built from the frames list **alone**, so nothing downstream
   cross-checks it.
3. **The critic's value at position `p` is conditioned on tokens ≤ `p`**, so `V(s_t)` for
   response token `i` is the output at `i-1`. verl left-shifts in `padding.py`. Exactly
   once — not zero times, not twice.
4. **`response_mask` / `logprobs` / `scores` / `per_token_reward` / `response_spans` all
   index `response_ids`.** Cut one, cut all of them.
5. **An observation in the response region is mask 0 and carries no reward.** Rewards land
   at `scores[end-1]` of a *model* span. This is what makes truncation safe.
6. **A conversation's ordinal is decided when it is opened**, never by position in
   `rows()` — a conversation the model never spoke in is dropped there, and numbering the
   survivors renumbers everything after the gap with no hole to notice.

## 4. Token budgets

Full treatment in `../logs/三个mode-token限制与结束逻辑.md`. The short version:

Every turn, in every mode, asks the same question:

```python
left = max_response_length - response_region_spent - measured_pending_observation - reserve
max_new_tokens = max(floor, left)
exhausted      = left < floor        # compact overrides: it can make room
```

The observation is **measured, not estimated** — which is why `render` is separable from
`encode` (measuring used to record the frames a second time). `reserve` is 0 except under
compaction, where it is `summary_budget + summary_request_len`; the request is a user
message into the *same* conversation, so leaving it out overflows deterministically.

Four defences, outermost first: static checks (`harness/budget.py`) → per-call ceilings
(`client._check_context`) → the per-turn room check → image-aware truncation at the batch
boundary. Only two static checks are fatal; the rest warn, because refusing on a worst
case rules out long episodes that a real rollout handles fine.

**The prompt region raises, the response region truncates.** Deliberate: the prompt is the
opening call and has nothing old to drop, so cutting it takes the system prompt.

## 4b. Advantage estimators

The context policy decides **how an episode is laid out in rows**; the estimator decides
**how it is scored**. They are independent, and every estimator here works under all
three policies — which is the property `TrajectoryView` exists to provide.

| `algorithm.adv_estimator` | one MDP step is | notes |
|---|---|---|
| **`vanilla_gae`** | one model-emitted token | **the baseline.** Ordinary single-turn GAE: the episode's whole reward is summed onto its *last* model-output token before the recursion runs, so at `lam=1` every token is handed the same return and only the critic apportions it. Identical to `token_level_gae` in every other respect, which is what makes the comparison mean something. Under `concat` it is verl's `gae` exactly; under the other two it is not, because verl would credit each row separately. |
| `token_level_gae` | one model-emitted token | the same recursion with each reward left where it was earned. State = everything seen before the token; action = the token. Anything the model did not emit (observations, template scaffolding) is state, never action, and the recursion steps over it. |
| `turn_level_gae` | one turn | writes a return only at each turn's first token; needs `value_mask`. |
| `trajectory_grpo` | — | one advantage per episode, normalised within its prompt group. No critic. |

★ **Why not verl's `gae` / `grpo`.** They score one row and open each with
`nextvalues=0`. Under `concat` a row *is* an episode, so that is correct and they are
fine. Under `no_concat` and `compact` a row is one conversation, so they assert that
nothing after the row boundary is worth anything — the agent is never credited across a
turn boundary. Nothing fails; the curves look ordinary. The trainer therefore **refuses
that pairing at startup** (`_vagen_check_estimator_spans_the_layout`), reading the set of
safe names from the registry the estimators populate themselves.

`no_concat_gae` was deleted 2026-08-08: it conflated a layout with an algorithm. What it
did is `turn_level_gae`, which finds turns from the token stream instead of assuming a
row is a turn.

### Adding one

```python
from vagen.custom_advantage import AdvantageInputs, AdvantageOutputs, advantage_estimator

@advantage_estimator("my_algo")
def my_algo(inputs: AdvantageInputs):
    beta = inputs.required_param("beta", "It weights X.")   # +algorithm.beta=0.3
    adv = inputs.zeros()
    for rows in inputs.view.trajectories:      # one episode, its rows in turn order
        ...
    return AdvantageOutputs(advantages=adv, returns=adv + inputs.values)
```

`inputs` carries the per-token tensors (`rewards` — KL-penalised; `scores` — raw;
`values`, `old_log_probs`, `ref_log_probs`, `rollout_log_probs`, `kl()`), the identity
columns (`group_idx`, `traj_idx`, `episode_id`, `conversation_id`, `uid`), and `view`.
`AdvantageOutputs` additionally carries `value_mask` and `policy_mask`. Full contract in
`vagen/custom_advantage/inputs.py`.

Registering is all that is needed to get `tests/test_estimator_contract.py` run against
it: layout equivalence, no advantage on observation tokens, padding duplicates not
double-counted, verl's kwargs tolerated, sentinel returns declared. Every one of those
properties fails silently in training. An estimator with a required hyperparameter needs
an entry in that file's `PARAMS` — forgetting fails with instructions.

★ **verl's own contract is two tensors.** `compute_advantage` does
`advantages, returns = fn(...)` and writes exactly those; the actor reads `advantages`,
the critic reads `returns`, and both mask with `response_mask`. There is no actor/critic
split and no mask in the return value, so `value_mask` and `policy_mask` travel as keys
written into the batch — each backed by a patch in `verl/workers/utils/losses.py`.

## 5. Running things

```bash
# the shared flags -- these select VAGEN's agent loop. Without them verl runs its own
# and the job looks healthy while none of this repo's rollout code executes.
vagen/configs/baseline_vllm.flags

bash examples/train/sokoban/train_ppo_qwen25vl3b.sh          # reads those flags
MODEL=/path/to/local/snapshot bash examples/train/...        # avoids a flaky hub lookup
bash examples/train/... --cfg job --resolve                  # dry-run the config, no GPU
```

A judge endpoint is needed only when `trainer.state_reward.*.enable=True`:
`bash scripts/launch_judge.sh` (~23 GB/card, shares every GPU).

**Reaping matters.** Orphaned vLLM workers holding 60–90 GB/card have bitten this project
repeatedly. Kill everything on the GPUs *except* the judge's process tree.

## 6. Where things are

```
vagen/core/          the contracts: harness, client, runner, tape, env
vagen/harness/       concat / no_concat / compact, and budget.py
vagen/agent_loop/    gym_loop.py (the verl AgentLoop), verl_client.py, multi_output.py
vagen/utils/         image_token_utils.py, concat_val_multi_turn.py, episode_log.py
vagen/trainer/       VagenPPOTrainer + the mixins over verl's SeparateRayPPOTrainer
vagen/custom_advantage/   the algorithm layer (TrajectoryView + the estimators)
logs/                design notes and findings -- see below
```

`../logs/` (outside git) is where the reasoning lives:

| file | what |
|---|---|
| `三个mode-token限制与结束逻辑.md` | the budgets and every termination path |
| `四个token上限-与compact观察.md` | the design discussion the budgets came from |
| `CHANGELOG-overnight.md` | what changed and why, including the mistakes |
| `template-seam.md` | a known defect, pinned as xfail, and why it is not a one-liner |
| `dropped-row-renumbering.md` | conversation numbering + what adopting fully-async needs |

## 7. Where it stands

**Working and verified end to end** — three modes on Sokoban, 6 steps each, validation at
0/2/4/6, episode transcripts to wandb, zero guards fired:

```
concat     1 conversation per episode
no_concat  one per turn
compact    4-5 per 20-turn episode        (max_turns=20, compact_budget=1300)
```

Also working: image-aware truncation, budget-aware generation in all three modes, the
identity chain (group > episode > conversation > turn), per-token rewards reaching the
loss, `value_mask` reaching the critic, all 20 training scripts config-verified with
sokoban and frozenlake actually run.

### Open, ranked

| | severity | where |
|---|---|---|
| The summary is a turn in the GAE recursion | by design | compaction's summary is a policy action -- generated, trained (mask 1), and its zero immediate reward is correct credit assignment. So it is a step, and the turn before it bootstraps through it. Listed because it surprises people, not because it is wrong |
| Compact loss reweighting | deferred | by the project owner; algorithm layer |
| 10 of 15 verl patches could move to the VAGEN layer | cleanliness | audited, hooks identified. The critic mask is explicitly blessed to stay, and the actor mask added 2026-08-08 alongside it for the same reason: a loss-side mask has no hook |
| An empty first generation shrinks the batch | low | the episode contributes no rows and `multi_output._postprocess` only raises when *every* rollout is empty. Under GRPO that quietly shrinks a group |
| `_summary_request_len` over-counts | low | it renders the request as an *opening* turn, so Qwen injects a system block: 39 tokens where the client sends 23. Over-reserving is the safe direction |

## 8. How to work here

- **Write the test so it can fail.** Several checks in this repo's history reported "ok"
  for code that had crashed on import, or asserted an identity (`max(f, x) >= f`). Before
  trusting a new check, mutate the thing it guards and watch it go red.
- **Do not trust a green run over a measurement.** The three-mode runs passed for weeks
  while compact was silently behaving as no_concat.
- **Long output goes to `../logs/`,** not the terminal.
- **Never edit `vagen/` or `verl/` while a training run is in flight** — a later mode in
  the same sweep will pick up the change and the comparison stops being one.
