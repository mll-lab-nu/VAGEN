# VAGEN Architecture

Training backend: a patched fork of verl main -- `JamesKrW/verl` at tag `vagen-260812`
(`27c51e9`) -- carried as the git submodule at `VAGEN/verl` and pip-installed from that
checkout (`pip install --no-deps -e ./verl`). It reports `0.9.0.dev`. The three patches are
listed in `AGENT.md`. This document was written against `release/v0.8.0` as a sibling
checkout; where the two disagree, the submodule is what runs.

> **Start with `AGENT.md`.** It is the five-minute orientation: the layering, the
> invariants that break silently, how to run things, and where the project stands. This
> document is the design rationale behind those decisions, and it is long. Where the two
> disagree, `AGENT.md` is the current one.

This began on 2026-08-02 as a design document, and most of it has since been built. Where
the two disagreed the code was right and the document was not, so the names below are the
ones in the tree. The rationale sections are kept because they still explain *why*; the
status table is rewritten because it was claiming that working code did not exist.

## Names, as built

The design used placeholder names that never made it into the code. Reading the older
sections, substitute:

| document said | the code has | where |
|---|---|---|
| `TokenTape` | `Conversation` | `core/tape.py` |
| `TokenLedger` | `Row` | `core/tape.py` |
| `BatchView` | `TrajectoryView` | `custom_advantage/trajectory.py` |
| `core/ports.py` | `core/client.py` | the data types live with the client |
| `core/harness/` | `core/harness.py` + `harness/` | contract and implementations |

## Scope

| | status |
|---|---|
| Upgrade to verl 0.8 | ✅ done, 2493 lines of fork deleted |
| Env / harness decoupling | ✅ **built.** `core/harness.py`, `core/client.py`, `core/runner.py`, `core/tape.py`. The harness holds no tokenizer, no env and no client; the runner is one loop for training and evaluation |
| `ConcatHarness` / `NoConcatHarness` | ✅ built, a dozen lines each, registered in `harness/__init__.py` |
| `CompactHarness` | ✅ built, with the budget arithmetic in `harness/budget.py` |
| Token accounting / budget checks | ✅ built 2026-08-06. See §12 |
| Image placeholder ↔ frame alignment | ✅ built 2026-08-06. `utils/image_token_utils.py` |
| Algorithm layer | ✅ `default_gae` (baseline), `token_level_gae`, `bi_level_gae_varlam`, `turn_level_gae`, `trajectory_grpo`, all on `TrajectoryView`. `no_concat_gae` deleted 2026-08-08; `default_gae` added 2026-08-09; `bi_level_gae` (the released VAGEN algorithm, for reproduction) 2026-08-10 |
| Row-local estimator under a splitting harness | ✅ refused at startup (`_vagen_check_estimator_spans_the_layout`) |
| VLM beyond Qwen | ⚠️ Qwen2.5-VL, Qwen3-VL and Qwen3.5 train. Processor handling is family-agnostic (`tests/test_vlm_families.py`), but GLM-4.1V and InternVL3 each fail for their own reason -- see `docs/issues.md`. No LLaVA script or config ships. |
| Compact loss reweighting | ❌ deferred, algorithm layer |
| Black-box harness | interface exists (a conversation id is the whole protocol); no adapter written |
| Eval unification / async trainers | package split only. See §13 for what adopting fully-async would take |

---

## 0. Layering

```
                 ┌──────────────────────────────────┐
                 │         EpisodeRunner            │  same code for train & eval
                 └──┬──────────┬──────────┬─────────┘
      ┌─────────────┘          │          └─────────────┐
      ▼                        ▼                        ▼
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Harness    │     │       Env        │     │ InferenceClient  │
│ ★ dependency│     │ step/reward/done │     │  any backend     │
│   free      │     └──────────────────┘     └──────────────────┘
│ Msg only    │
└─────────────┘
      │  (train only) every append also goes to ──►  ┌───────────┐
      └──────────────────────────────────────────────│ TokenTape │──► TokenLedger ──► verl
                                                     └───────────┘
```

**Eval = Runner + Harness + any inference client.**
**Train = the same + a `TokenTape`.** ← the only difference

### Packages and dependency direction

The **rule** below is real and enforced by review: nothing under `vagen/core`, the
environments, or the evaluation path may import verl, torch or ray, so evaluation runs
without a training install. The *names* in the original sketch (`clients/`, `eval/`,
`train/`, `ports.py`, `msg.py`) were never built. What exists:

```
vagen/
├── core/            ★ no verl / no torch / no ray
│   harness.py  client.py  runner.py  tape.py  env.py  env_adapter.py
├── harness/         concat / no_concat / compact, and budget.py   ★ no verl
├── envs/  envs_remote/                                            ★ no verl
├── evaluate/        depends on core + harness + envs   ★ no verl / torch / ray
│   run_eval.py  runner.py  chat_client.py  vision_workflow.py  registry.py
└── agent_loop/  trainer/  custom_advantage/  custom_loss/  custom_filter/
                  depends on all of the above + verl
```

```
vagen.core, vagen.harness, vagen.envs → PIL, numpy, omegaconf
vagen.evaluate                        → the above + openai/anthropic…  ❌ never verl/torch/ray
vagen.agent_loop, vagen.trainer       → + verl, torch, ray
```

There is no `[eval]` extra -- `setup.py` ships `test`, `vllm` and `sglang`. The rule itself
is gated by `tests/test_evaluation_needs_no_training_install.py`, which walks the AST of
every module in those packages and fails on an import of verl, torch or ray at any nesting
depth. ManiSkill's vendored simulator is exempt, with the reason in the test.

---

## 1. What verl already gives us — do NOT rebuild

Checked against `release/v0.8.0`. Six things dropped from an earlier draft:

| Earlier plan | verl already provides |
|---|---|
| `IncrementalTokenizer` with dummy-prefix subtraction + cache | `AgentLoopBase.apply_chat_template(messages, images=…, remove_system_prompt=True)` (`agent_loop.py:272`) and `utils/chat_template.py::initialize_system_prompt` — more robust than a hand-rolled subtraction (derives the prefix by differencing two empty-user templates) |
| `MessageTokenWrapper.settle()` rebuilding tokens from messages | Copy `ToolAgentLoop`'s pattern: one growing `ids` list + one `mask` list, appended as you go, sliced at the end. Cannot drift |
| `TokenSpan` / `SegmentView` structures | Segment boundaries are just cut points on the tape |
| `Msg.provenance` | Sampled ids go straight into the tape |
| `Msg.usage` + a `TokenCounter` protocol | `len(tape.ids)` is exact and free |
| position_ids / attention_mask / padding / `multi_modal_inputs` | `AgentLoopWorker._agent_loop_postprocess` computes all of it, including the generic `processor.get_rope_index` path |

What remains on the token side is **one ~80-line class** (`TokenTape`).

Also free: sticky-session prefix caching (`generate(request_id=…)`), async generation dump with exception propagation, the multi-output agent-loop hook (§8.1), and — when we get to async — transparent partial-rollout resume.

---

## 2. Data types · `core/client.py`

```python
@dataclass
class Msg:
    """A conversation message — the only unit the harness manipulates.

    `content` must be STRUCTURED: a plain str, or
    [{"type": "text", "text": ...}, {"type": "image"}].
    Never a model-specific placeholder literal (Qwen's <|vision_start|>,
    InternVL's <IMG_CONTEXT>, ...). Expanding placeholders is the processor's job.
    This rule is what keeps the harness model-free.

    `images` order must match the order of image entries in `content`.
    """
    role: str                                  # "system" | "user" | "assistant"
    content: str | list[dict]
    images: list[Image] = field(default_factory=list)
    kind: str = ""                             # "system"|"obs"|"response"|"summary"

    @classmethod
    def from_obs(cls, obs: dict, kind: str = "obs") -> "Msg":
        """env's {obs_str, multi_modal_input} -> structured Msg.
        Reuses core/msg.py::compile_text_images_for_order to interleave text and
        images on <image> placeholders (lifted from evaluate/utils/mm_utils.py)."""


@dataclass
class TokenLedger:
    """One training sample.

    ★ Python built-ins + PIL only — no torch.Tensor, or the eval side is forced to
      install torch. Tensor conversion happens in agent_loop/gym_loop.py.
    """
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]                   # len == len(response_ids)
    logprobs: list[float]                      # len == len(response_ids)
    images: list[Image]
    meta: TrajectoryMeta


class TrajectoryMeta(TypedDict):
    """★ The structural contract every algorithm reads (§8.3).

    Semantics of the index fields across modes — note they are well-defined in all
    three from day one, so algorithms written now keep working when compaction lands:

                 segment_idx   turn_idx   global_step_idx   num_segments
      concat        0           0..T-1      = turn_idx           1
      no-concat     0..T-1      0           = segment_idx        T
      compact       0..M-1      0..T_s-1    flattened            M
    """
    group_idx: str                             # prompt / env instance -> GRPO group
    traj_idx: int                              # which of the n rollouts
    segment_idx: int
    num_segments: int
    global_step_idx: list[int]                 # per turn; the timeline GAE runs on
    turn_rewards: list[float]
    token_spans: list[tuple[int, int]]         # ★ turn -> [start, end) in response_ids
    anchor_id: list[str | None]                # optional env-state id (anchor-grouped algos)


@dataclass
class EpisodeResult:
    """Runner's uniform return. Train reads `ledgers`; eval reads the rest."""
    ledgers: list[TokenLedger] | None          # None when no TokenTape was passed
    messages: list[Msg]
    rewards: list[float]
    terminated: bool
    truncated: bool
    num_turns: int
    num_segments: int
    info: dict                                 # accumulated env info (success, ...)
    timing: dict


class InferenceClient(Protocol):
    limits: ClientLimits
    returns_token_ids: bool                    # ★ False ⇒ cannot be used for training
    last_prompt_tokens: int                    # from the last call

    async def chat(self, messages: list[Msg], sampling_params: dict,
                   *, request_id: str) -> Msg:
        """Returns an assistant Msg. If returns_token_ids, it also carries
        `.token_ids` / `.logprobs` verbatim from the server. `request_id` enables
        sticky-session routing for prefix-cache hits; implementations may ignore it."""
```

---

## 3. `Conversation` · `core/tape.py`

```python
class TokenTape:
    """Token-level ledger for one episode. ★ Training only — eval never creates one.

    Mirrors verl's ToolAgentLoop bookkeeping: one growing id list plus a mask that
    starts at the first model response. Segment boundaries are cut points.

    ★ `cut()` is what makes no-concat work, and it is also exactly what compaction
      will need later — the two modes share this mechanism entirely (§9.1).

    Invariant (asserted on every append):
        len(ids) == len(logprobs)  and  len(mask) <= len(ids)

    The encoder is injected, so this class carries no tokenizer and stays importable
    without transformers. In training the injected fn is
    `AgentLoopBase.apply_chat_template(..., remove_system_prompt=True)`.
    """

    def __init__(self, encode: Callable[[Msg], Awaitable[tuple[list[int], list[Image]]]]):
        self.ids: list[int] = []
        self.mask: list[int] = []
        self.logprobs: list[float] = []
        self.images: list[Image] = []

    async def append_context(self, msg: Msg) -> None:
        """Non-generated tokens (system / observation). mask=0, logprob=0.
        Encoded incrementally — never re-tokenizes prior content."""

    def append_response(self, token_ids: list[int], logprobs: list[float]) -> tuple[int, int]:
        """Model-sampled tokens, verbatim from the server. mask=1.
        Returns [start, end) of this turn within the current segment's response —
        stored as `token_spans`, which is what turn-level and bi-level estimators
        index by (§8.3).

        ★ These ids must come straight from the inference response. Never
          decode-then-re-encode: re-tokenization drift silently misaligns the mask.
        """

    def cut(self, *, closed_by: str, seed: list[Msg] | None = None) -> None:
        """Close the current segment; optionally seed the next one.

        no-concat  -> cut(closed_by="turn", seed=[sys, next_obs])
        compaction -> cut(closed_by="compact", seed=[sys, summary_as_user, next_obs])

        ★ Images belong to the segment that holds their tokens. Dropping images and
          dropping their tokens must happen inside this one call — a mismatch between
          image-token count and len(images) makes the processor either crash or
          silently misalign.
        """

    def ledgers(self, meta_fn: Callable[[int], TrajectoryMeta]) -> list[TokenLedger]:
        """Slice cut points into training samples. Segments with no mask==1 token
        are skipped (possible when a cut is immediately followed by done)."""

    def __len__(self) -> int:
        """Current context length in tokens — exact, free, includes the newest
        observation."""
        return len(self.ids)
```

---

## 4. Harness · `core/harness.py` and `harness/`

The harness works in text. It never sees a token, a mask, a reward or a logprob, and it
holds no tokenizer, no client and no env. Each turn it answers one question: **does the
next call continue the current conversation, or start a new one, and seeded with what?**

```python
@dataclass
class Call:
    messages: list[Msg]
    conversation_id: str | None   # None starts a new conversation


class BaseHarness(ABC):
    """Text-space context policy."""

    def begin(self, system: Msg, init_obs: Msg) -> None: ...
    def add_observation(self, obs: Msg) -> None: ...

    @abstractmethod
    def next_call(self) -> Call:
        """What to send now."""

    def accept(self, response: Response) -> str | None:
        """Record the response; return the action text for the environment, or None if
        the harness consumed it itself.

        ★ The `None` branch is what makes compaction ordinary. A summary is a model
        response to a user message like any other -- the harness simply keeps it instead
        of forwarding it, and seeds the next conversation with it.
        """
        self._msgs.append(response.as_msg())
        return response.text
```

The three modes are one axis -- when to drop the conversation id:

| mode | `conversation_id` | seed of a new conversation |
|---|---|---|
| concat | always the current one | — (never starts one) |
| no-concat | `None` every turn | system + latest observation |
| compact | `None` once the budget is hit | system + summary + latest observation |

`ConcatHarness` and `NoConcatHarness` are a handful of lines each. `CompactHarness`
differs only in emitting one extra `Call` -- "summarise the following" -- whose response
it consumes rather than forwards.

**Number of conversations = number of training rows.** One for a concat episode, one per
turn for no-concat, one per compaction span. That is the same fan-out the multi-output
agent loop already produces (§8.1), so nothing downstream has to learn about compaction.

### What this buys

*Eval needs only the harness.* With no tokenizer and no client, the same object drives a
black-box API: `conversation_id` is `previous_response_id` on OpenAI's Responses API, a
session on sglang, and a cached prefix on vLLM. The black-box case of §9.2 stops being a
separate design.

### Edge cases

| Case | Handling |
|---|---|
| a new conversation immediately followed by done → a row with no trainable token | dropped when the tape is materialised. **Easiest one to miss** |
| a single observation exceeds the whole budget | fail loudly; never silently truncate |
| response hits `finish_reason == "length"` | mark the episode `truncated`, not terminated (§6) |

---

## 5. Runner · `core/runner.py`

```python
async def run_episode(env, harness, client, cfg, *, seed) -> EpisodeResult:
    """★ Identical for training and evaluation. Whether tokens are recorded is the
    client's business, not this loop's."""
    obs, _ = await env.reset(seed)
    harness.begin(Msg.system(await env.system_prompt()), Msg.from_obs(obs))

    for _ in range(cfg.max_turns):
        call = harness.next_call()
        response = await client.send(call.messages, call.conversation_id)

        action = harness.accept(response)
        if action is None:
            continue          # the harness kept this one (a summary); no env step

        obs, reward, terminated, truncated, info = await env.step(
            action, response_token_ids=response.token_ids, tokenizer=client.tokenizer
        )
        harness.add_observation(Msg.from_obs(obs))
        if terminated or truncated:
            break
```

Compaction has no branch here. It is a call whose response the harness keeps -- which is
why §4's `accept` returns an `Optional`.

The loop touches no tokens. Everything token-level -- tokenizing what we send, recording
which spans are model output, placing rewards, collecting logprobs, cutting rollout rows
at conversation boundaries -- lives behind `client`, written once (§7).

### Why the tokenizer reaches the env

An env that wants per-token rewards (a reward model, a process scorer) is the natural
place to compute them, and forcing that through a text-only interface would only move
the problem. So it gets both the response's token ids and the tokenizer.

★ **But a returned vector must have the length of `response_token_ids`, asserted.** The
danger was never that the env holds a tokenizer; it is that it *re-encodes* the response
and returns a vector assumed to line up. BPE is not compositional, so a re-encoding can
differ at the boundaries -- the same failure that cost hours here when a prompt was
tokenized twice and diverged by two tokens, with both versions well-formed and nothing
raising. Decoding ids to text is safe; re-encoding is not. The length assertion makes the
unsafe path loud instead of silent.

With `response_token_ids=None`, the env returns a scalar and the tape places it on the
last token.

---

## 6. Env · `core/envs/` — unchanged except one thing

```python
class GymImageEnv(ABC):
    async def system_prompt(self) -> Obs: ...
    async def reset(self, seed: int) -> tuple[Obs, dict]: ...
    async def step(self, action_str: str) -> tuple[Obs, float, bool, bool, dict]:
        """★ AS DESIGNED, and not what `vagen/envs/gym_image_env.py` ships.

        The class an environment author actually subclasses -- the one README.md points at
        -- returns the 4-tuple `(obs, reward, done, info)`. `vagen/core/env.py`'s BaseEnv
        and `vagen/core/runner.py` do use the 5-tuple, and `vagen/core/env_adapter.py`
        bridges between them. Write the 4-tuple; the rationale below is why the inner
        contract has five, not an instruction.

        The split: `done` becomes (terminated, truncated).

        Not about gym compatibility — it is GAE correctness:
          terminated — real MDP terminal state   → bootstrap V = 0
          truncated  — hit max_turns / budget    → **must bootstrap V(s_T)**

        Treating truncation as termination systematically under-values long-horizon
        tasks. Back-compat: expose `done = terminated or truncated` as a property.

        `info` may optionally carry `anchor_id` — a hash of the environment state,
        used later by anchor-grouped algorithms (GiGPO). Envs that omit it simply
        cannot use those; nothing else changes.
        """
    async def close(self) -> None: ...
```

We do **not** add `observe()` / `state_digest()` / `on_compact()`: the harness holds every message, and the last user message *is* the current observation.

> ⚠️ That assumes observations are Markov. True for FrozenLake / Sokoban / navigation / primitive_skill; **spatial_gym's may be incremental**. Relevant to compaction, which has since landed (`CompactHarness`).

---

## 7. Clients · `core/client.py`

One class, written once. It is the only place that knows about tokens.

```python
class InferenceClient(ABC):
    tokenizer: Any | None            # None for closed APIs

    async def send(self, messages: list[Msg], conversation_id: str | None) -> Response:
        """Response: text, conversation_id, token_ids | None, logprobs | None."""
```

Its job, on top of talking to a backend:

* tokenize what we send, so the span it added is known **by construction** rather than
  recovered afterwards -- matching text back to token spans does not work, because BPE
  merges across boundaries;
* record, per conversation, the alternating spans of *context we supplied* (mask 0) and
  *tokens the model produced* (mask 1);
* place the env's reward, scalar or vector, onto those spans;
* keep logprobs alongside;
* emit one rollout row per conversation.

That is the whole token layer. It does not vary with the harness or the env, which is the
point: **the two things that change are the harness and the env, and neither can see a
token.**

| Class | Backend | token ids | v1 |
|---|---|---|---|
| `VerlTITOClient` | verl `LLMServerClient` | ✅ | ✅ training |
| `ChatAPIClient` | existing `ModelAdapter` (~40 lines of wrapping) | ❌ | ✅ eval against closed APIs |
| `LocalEngineClient` | in-process `sgl.Engine` / `vllm.LLM` | ✅ | later |

**Training requires token ids**, so a client without them is rejected at construction
rather than halfway through an episode.

★ The ids come *from the engine*, not from re-tokenizing locally. Established this
cycle: the engine expands multimodal placeholders its own way, and a locally tokenized
prompt can differ from the one that produced the response -- silently, since both are
well-formed. `vllm_async_server` now returns `prompt_token_ids` and the loops adopt them
(§ "VLM families"). Whatever a family's expansion rules are, the sequence trained on is
the sequence sampled from, with no per-family code.

`ChatAPIClient` has no ids and so cannot train; for eval that is fine. Its token counts
come from the provider's `usage`, counted by *their* tokenizer, so a budget-driven
harness will trigger at slightly different points than in training. Log the actual
trigger point.

---

## 8. Training side · `train/`

### 8.1 verl integration — zero fork

`NoConcatHarness` produces T samples per rollout, so verl's one-output-per-agent-loop assumption must be lifted. v0.8.0 already factored the postprocess out, so this is four lines:

```python
# agent_loop/gym_loop.py
class MultiOutputAgentLoopWorker(AgentLoopWorker):
    async def _run_agent_loop(self, sampling_params, trajectory, *, agent_name, trace=True, **kw):
        outputs = await agent_loop.run(sampling_params, **kw)          # may return a list
        if not isinstance(outputs, list): outputs = [outputs]
        return [await self._agent_loop_postprocess(o, trajectory["validate"], **kw) for o in outputs]

    def _postprocess(self, inputs, input_non_tensor_batch=None, validate=False):
        repeats = [len(sub) for sub in inputs]
        flat = [o for sub in inputs for o in sub]
        expanded = {k: np.repeat(v, repeats, axis=0) for k, v in (input_non_tensor_batch or {}).items()}
        return super()._postprocess(flat, input_non_tensor_batch=expanded, validate=validate)

class MultiOutputAgentLoopManager(AgentLoopManager):
    agent_loop_workers_class = ray.remote(MultiOutputAgentLoopWorker)
```
```bash
actor_rollout_ref.rollout.agent.agent_loop_manager_class=vagen.agent_loop.multi_output.MultiOutputAgentLoopManager
```
Official hooks: `workers/config/rollout.py:97`, `ray_trainer.py:930`, `agent_loop.py:1074`.
**Side effect: the 825-line `agent_loop_no_concat.py` fork is deleted outright.** Compaction later needs nothing more here.

### 8.2 Trainer mixin — do not fork `ray_trainer.py`

v0.8.0's `SeparateRayPPOTrainer` breaks `RayPPOTrainer`'s 410-line monolithic `fit()` into ~20 `_fit_*` hooks — and all three async trainers inherit from it, so async comes free later.

```python
class VagenLogicMixin:            # pure logic, bound to no verl method name
    def _vagen_post_advantage(self, batch): ...
    def _vagen_collect_metrics(self, batch): ...
    def _vagen_filter(self, batch): ...

class VagenV0Mixin(VagenLogicMixin):              # binds to v0.8.0's _fit_*
    def _fit_compute_advantage(self, b): return self._vagen_post_advantage(super()._fit_compute_advantage(b))
    def _fit_collect_metrics(self, b):   super()._fit_collect_metrics(b); self._vagen_collect_metrics(b)
    def _fit_experimental(self, b):      super()._fit_experimental(b);    self._vagen_filter(b)

class VagenPPOTrainer(VagenV0Mixin, SeparateRayPPOTrainer): pass
# later, free:  VagenOneStepOffTrainer(VagenV0Mixin, OneStepOffRayTrainer)
```
`vagen/trainer/ppo_trainer.py` + `vagen/trainer/mixin.py`, replacing a 1660-line vendored trainer with a few hundred. The two-layer split exists so we can migrate to main's V1 trainer later by swapping only the binding layer.

### 8.3 ⭐ Algorithm layer — token-level / turn-level / bi-level

Algorithms differ along only four axes:

| Axis | token-PPO | turn-PPO | bi-level | *(later)* GiGPO |
|---|---|---|---|---|
| Credit granularity | token | turn | turn **and** token | turn |
| Baseline | learned V | V at a turn anchor | both levels | group mean, **two groupings** |
| Grouping key | — | — | — | `group_idx` **and `anchor_id`** |
| Needs critic | ✅ | ✅ | ✅ | ❌ |

All of them need the same thing: **the turn structure, and where each turn's tokens live.** Today that gets flattened into a padded tensor and every estimator reconstructs it by hand. Make it a typed view instead.

```python
# train/advantage/view.py
@dataclass
class TurnView:
    """One turn as an algorithm sees it."""
    group_idx: str
    traj_idx: int
    global_step_idx: int
    segment_idx: int
    row: int                        # row index in the padded tensor batch
    token_span: tuple[int, int]     # [start, end) into that row's response_ids
    anchor_pos: int                 # token position used as the value anchor
    reward: float
    value: float | None             # critic V at anchor_pos
    anchor_id: str | None


@dataclass
class BatchView:
    """Structured view over a padded batch. Built once from non_tensor_batch columns,
    then reused by whichever estimator is registered. Estimators are written against
    this, never against DataProto."""
    turns: list[TurnView]
    token_level_rewards: Tensor     # (B, L)
    values: Tensor | None           # (B, L)
    response_mask: Tensor           # (B, L)

    def by_group(self)      -> dict[str, list[TurnView]]: ...
    def by_trajectory(self) -> dict[tuple[str, int], list[TurnView]]: ...  # time-sorted
    def by_anchor(self)     -> dict[str, list[TurnView]]: ...              # for GiGPO later
    def scatter(self, per_turn: dict[TurnView, float]) -> Tensor: ...      # turn value -> (B, L)
```

```python
@register_algo("token_gae", needs_critic=True)
def token_gae(v: BatchView, cfg):
    """Standard per-token GAE. Delegates to verl's core_algos."""


@register_algo("turn_gae", needs_critic=True)
def turn_gae(v: BatchView, cfg):
    """Turn-level GAE: reward and V live at turn granularity; the resulting
    advantage is broadcast across the turn's tokens.

    by_trajectory() is already time-sorted on global_step_idx, so this recursion
    cross compaction boundaries unchanged."""
    adv = torch.zeros_like(v.response_mask, dtype=torch.float32)
    ret = torch.full_like(adv, cfg.ignore_value)
    for traj in v.by_trajectory().values():
        lastgae, next_v = 0.0, 0.0
        for t in reversed(traj):
            delta   = t.reward + cfg.gamma * next_v - t.value
            lastgae = delta + cfg.gamma * cfg.lam * lastgae
            s, e = t.token_span
            adv[t.row, s:e] = lastgae
            ret[t.row, t.anchor_pos] = lastgae + t.value
            next_v = t.value
    return adv, ret


@register_algo("bi_level_gae_varlam", needs_critic=True)
def bi_level_gae_varlam(v: BatchView, cfg):
    """High level: turn_gae over the turn sequence.
    Low level:  per-token GAE inside each turn, using the turn advantage as its
                terminal bootstrap.
    A_token = A_turn + w_low * A_token_within_turn
    """
```

Each estimator is ~20 lines. **Acceptance test for this abstraction: a new estimator that needs more than ~30 lines, or that reaches into `DataProto`, means `BatchView` is missing a field — add the field, do not work around it.**

`needs_critic` is declared on the algorithm, not configured separately; `main_ppo` derives `critic.enable` from it. Today you can select a critic-free estimator and still pay for a critic.

**Migration**: `no_concat_gae` (338 lines) becomes `turn_gae` (~20). Most of its bulk is reconstructing structure `BatchView` now provides — dedup by `(group, traj, turn)`, factorizing string uids, index juggling.

### 8.4 `value_mask`

Turn-level and bi-level estimators write `returns` only at anchor positions, marking the rest with a sentinel. The critic must skip those. Our patch moves from the deleted `dp_critic.py` to `workers/utils/losses.py::value_loss` — note `data.select("values", "returns", "response_mask")` there is a **whitelist**, so `"value_mask"` has to be added to it, and it must survive into the `TrainingWorker`'s TensorDict.

---

## 9. Historical sections, removed

This document carried three sections that had stopped describing the code: a "Deferred" list whose first entry was context compaction (built -- see `vagen/harness/compact.py`, and the scope table in §0 of this file already said so), an effort estimate for phases that have all shipped, and ~400 lines of dated status log including a live-bug postmortem and commit SHAs from a branch that no longer exists.

They are in git history if you want them. What is below is the part that still holds.

## 12. Token accounting · `harness/budget.py`

Six quantities bound a run. Three are enforced by something, three were not until
2026-08-06:

    C     rollout.max_model_len          hard. The engine refuses past it.
    n_p   data.max_prompt_length         a row's prompt region
    n_r   data.max_response_length       a row's response region
    g     response_length_per_turn       one generation. Hard: it is max_new_tokens.
    E     max_env_response_per_turn            one observation, after the processor has
                                         expanded its images
    T     max_turns                      environment steps in an episode

and compaction adds two that have to fit inside those:

    m     trainer.compact_budget         the largest a conversation may become
    k     trainer.compact_summary_budget the largest a summary may be

`S`, the system prompt, comes from the environment and is measured rather than configured.
Every relation that does not mention it is checked before the rollout; the ones that need
it are checked on the first call, which is the earliest they can be.

    no_concat   S + E <= n_p,  g <= n_r,  S + E + g <= C
    concat      T*g + (T-1)*E <= n_r      every turn lands in one response region
    compact     S + k + E <= min(n_p, m)  a conversation opens on a summary
                m + E + g + |req| + k <= min(C, n_r)
                k <= g,  2k <= m          a summary is a generation, and a compression

The response region, not the window, is what bounds a compacted conversation: almost all
of it lands there, so a large `max_prompt_length` hides an overflow rather than absorbing
one.

### The trigger

`m` is the largest a conversation may become, so compaction fires when *one more turn
would not fit*, against a turn cost it **measures** — the last continuation, not the
configured ceiling. On Sokoban `g` is 512 and a real turn costs about 138, so a trigger
charging the ceiling fires after the first turn of every conversation and compaction
silently degenerates into no_concat with a summary attached. The estimate resets with the
conversation: a new one cannot correct an inherited value, because its first turn is an
opening call and openings may not inform it.

Measured on Sokoban vision, for scale: `S`=589, `E`=44–58 with a 96×96 frame, so a
conversation opens at 633 tokens. The `compact_budget=400` used before this existed could
not hold the system prompt, let alone a turn.

### Runtime companions

Static checks are necessary, not sufficient, because `S` and `E` are measured:

| guard | where | when |
|---|---|---|
| `BudgetError` | `harness/budget.py` | before the rollout, from config alone |
| `ContextTooLarge` | `core/client.py` | on the call, against the mode's ceiling |
| `CompactionMakesNoProgress` | `harness/compact.py` | on a *repeat* — one short conversation is data, two is the budget |
| `cap_token_ids` | verl `agent_loop.py` | at the end, refusing to truncate an episode |
| empty-generation retry | `core/client.py` | before the tape, the harness and `env.step` all see it |

## 13. Adopting fully-async policy

The harness, tape, reward path and budget accounting need **no changes**. verl absorbs
rollout interruption a layer below the agent loop:

    verl/experimental/fully_async_policy/fully_async_rollouter.py:52
        """...making rollout interruption invisible to the AgentLoop."""

`FullyAsyncLLMServerClient` resumes from `prompt_ids + token_ids` until the stop reason is
no longer `aborted`, and it is installed at exactly the argument the agent loop reads as
`self.server_manager` (`agent_loop.py:633`). So the resume loop sits *below* `VerlClient`.

What would actually change:

1. **The manager base class.** `MultiOutputAgentLoopManager(AgentLoopManager)` would have
   to compose with `FullyAsyncAgentLoopManager`, which overrides `generate_sequences_single`
   where ours overrides `generate_sequences`. This is the only structural step.
2. **The weight version is dropped on the floor.** Under partial rollout one response can
   span weight updates, and the resume client reports `min_global_steps` / `max_global_steps`
   in `extra_fields`. `VerlClient.generate` reads that dict already but takes only
   `prompt_token_ids`. Off-policy correction needs the rest.
3. **Per-episode failure isolation.** `asyncio.gather` runs with `return_exceptions=False`,
   so one episode's exception kills the batch. `BudgetError` and `ImagePlaceholderMismatch`
   are fatal by nature; `ContextTooLarge`, `CompactionMakesNoProgress` and the
   `cap_token_ids` overflow should drop one episode and continue.
