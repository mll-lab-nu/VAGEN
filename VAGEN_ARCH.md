# VAGEN Architecture

Training backend `verl-project/verl@release/v0.8.0` (bee9f6f4).

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
| Algorithm layer | ✅ `token_level_gae` (baseline), `bi_level_gae`, `turn_level_gae`, `trajectory_grpo`, all on `TrajectoryView`. `no_concat_gae` deleted 2026-08-08 |
| Row-local estimator under a splitting harness | ✅ refused at startup (`_vagen_check_estimator_spans_the_layout`) |
| VLM beyond Qwen | ✅ Qwen / LLaVA / InternVL all training |
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

```
vagen/
├── core/       ★ no verl / no torch / no ray
│   ports.py  tape.py  harness/  runner.py  envspec.py  envs/  envs_remote/  metrics.py
├── clients/    ★ no verl
│   chat_api.py  adapters/            (local_engine.py / managed_server.py: later)
├── eval/       depends on core + clients — **no verl / torch / ray**
└── train/      depends on core + clients + verl
    verl_client.py  gym_loop.py  multi_output.py   (see AGENT.md section 6)
```

```
vagen.core    → PIL, numpy, omegaconf
vagen.clients → vagen.core, openai/anthropic…
vagen.eval    → vagen.core, vagen.clients        ❌ never verl/torch/ray
vagen.train   → + verl, torch, ray
```

CI gate: in a clean env, `pip install vagen[eval] && python -c "import vagen.eval"` must pass.

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
        """★ The only change: split `done` into (terminated, truncated).

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

> ⚠️ That assumes observations are Markov. True for FrozenLake / Sokoban / navigation / primitive_skill; **spatial_gym's may be incremental**. Relevant only once compaction lands — do not fix preemptively.

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
`vagen/ray_trainer.py`: **1660 lines → a few hundred**. The two-layer split exists so we can migrate to main's V1 trainer later by swapping only the binding layer.

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
    will cross compaction boundaries unchanged once compaction lands."""
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


@register_algo("bi_level_gae", needs_critic=True)
def bi_level_gae(v: BatchView, cfg):
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

## 9. Deferred, with the interface left in place

### 9.1 Context compaction

Under §4's contract compaction is not a mechanism, it is a `CompactHarness` that drops
the conversation id when a budget is hit and seeds the next conversation with a summary.
Everything it relies on is already built and already exercised by no-concat:

| Needed for compaction | v1 state |
|---|---|
| starting a new conversation mid-episode | ✅ no-concat does it every turn |
| a row per conversation | ✅ multi-output agent loop (§8.1) |
| `group_idx` / `traj_idx` / `turn_idx` chaining rows into one trajectory | ✅ |
| credit crossing a conversation boundary | ✅ `TrajectoryView` recurses across rows (§8.3) |
| a response the harness consumes instead of forwarding | ✅ `accept() -> None` (§4) |

Two things that were listed here as missing are not:

* **cross-boundary bootstrapping** — built for no-concat, and a compaction boundary is
  the same object.
* **`summary_loss` on/off/aux** — unnecessary. A summary is a model response to a user
  message, so it is trained like any other turn, and its credit arrives the ordinary way:
  downstream rewards propagate back through the trajectory recursion, and the summary
  seeds the next conversation, so its quality moves the whole episode's score. The
  implicit signal is the signal.

One thing to check rather than build:

* **`1/M` weighting.** More compactions means more tokens from one episode. Under
  `loss_agg_mode=token-mean` that is more loss weight, which is a channel worth not
  handing to a trainable compaction decision. Under `seq-mean-token-mean` it is not.
  Confirm which is in use before enabling compaction; no design work needed now.

### 9.2 Black-box harnesses (OpenCode / OpenClaw / Claude Code)

Supported later as a **second mode**, not a fourth harness — the loop is driven from the other side:

```
driven    Runner → Harness → Client                    → TokenTape          ─┐
inverted  ext agent → shim server → TrajectoryRecorder                      ─┴─► list[TokenLedger]
```

Both terminate at `list[TokenLedger]`, so the verl adapter, `BatchView`, every algorithm and the multi-output hook are shared unchanged.

**v1 cost: one rule, zero code.**
> 🔴 Nothing downstream of `TokenLedger` may assume the Runner drove the episode — no reliance on `env.step()` having produced the rewards, no assumption that `segment_idx` came from an explicit cut.

When we build it: **port slime's `TrajectoryManager`** (508 lines, Apache-2.0 — verify before vendoring) rather than writing it. It solves the hard part (reconstructing provenance from string-replayed history, forking on prefix divergence, absorbing re-tokenization drift), is tested against real Claude Code, and its only coupling is to `slime.utils.types.Sample`. Retargeting it to `TokenLedger` is ~1 day; from scratch is ~1 week and buggier.
⚠️ Inverted mode's correctness is inherently weaker: tokens whose sampled origin cannot be proven must drop to `loss_mask=0`.

### 9.3 Async trainers, eval clients

`VagenOneStepOffTrainer(VagenV0Mixin, OneStepOffRayTrainer)` is a one-liner once §8.2 exists. `LocalEngineClient` / `ManagedServerClient` and deleting `vision_workflow.py` follow the same `InferenceClient` protocol. Neither needs design work now.

---

## 10. Engineering requirements

1. **Every class and every feature ships with tests in the same PR.** No tests, no merge.
2. **Test tiers**, kept separable so the fast tier runs on every push:
   - **T1 pure unit** — `Msg`, `TokenTape`, both harnesses, `EpisodeRunner` against a fake client and fake env. No GPU, no network, no transformers. Must run in < 10 s.
   - **T2 contract** — the six invariants below plus the chat-template self-check, against 3 real tokenizers (Qwen2.5-VL, Qwen3, InternVL). CPU only.
   - **T3 integration** — one short end-to-end run per phase, compared against a stored baseline curve.
3. **Store a baseline before touching anything in a phase.** Every phase's exit criterion is "matches baseline".
4. **CI enforces dependency direction** — clean-env import of `vagen.evaluate`.
5. **Fail loudly.** Every silent-degradation path found so far must raise, not warn: verl's `hf_processor` swallowing an unsupported processor into `None`; chat-template prefix instability; image/token count mismatch.
6. **New harness ≈ 50 lines, new algorithm ≈ 20 lines.** If either needs much more, the abstraction leaked — fix the abstraction, don't work around it.
7. 🔴 **Nothing downstream of `TokenLedger` assumes the Runner drove the episode** (§9.2).

### Six hard contracts (T2)

| # | |
|---|---|
| 1 | Model-generated tokens come verbatim from the inference response — **never decode-then-re-encode** |
| 2 | Already-encoded messages are never re-tokenized |
| 3 | `mask==1` positions have real logprobs; `mask==0` positions are 0 |
| 4 | `len(mask) == len(logprobs) == len(response_ids)` |
| 5 | `decode(prompt_ids + response_ids)` is a valid, readable conversation |
| 6 | `len(images)` == the number of images the token sequence expands to |

---

## 11. Plan and effort

Engineer-days for one person familiar with the codebase, **tests included**.

| Phase | Work | Days | Risk |
|---|---|---|---|
| **0** ✅ **DONE** | Branch `vagen-lite-v0.8.0` off `release/v0.8.0`. Re-apply only 2 patches: VLM ValueHead `forward`, and `value_mask` → `workers/utils/losses.py::value_loss` (§8.4). Drop the 3 that landed upstream | 0.5 | — |
| **1** | Upgrade + mixin refactor, done together. Re-copy `main_ppo.py`; `ray_trainer.py` → `VagenLogicMixin`/`VagenV0Mixin`; fix imports and config; delete `agent_loop_no_concat.py` (825) via §8.1 | **8–10** | 🔴 **dominant risk** — 1008 upstream commits; the time actually goes into baseline comparison |
| **2** | VLM generalization: make `utils/tokenizer.py:227`'s silent failure a hard error, add the `get_rope_index` fallback, get InternVL training, upstream the patch | **2** | low · **parallelizable with 3–5** |
| **3** | Package split + `core/ports.py` + `TokenTape` + T2 contract tests | **3.5** | low |
| **4** | `BaseHarness` + Concat/NoConcat + `EpisodeRunner`; split `terminated`/`truncated` across envs; merge the two gym agent loops | **4** | medium |
| **5** | Algorithm layer: `BatchView` + registry; port `no_concat_gae` (338) → `turn_gae` (~20); add `token_gae` and `bi_level_gae` | **3** | medium |
| | | **21–23** | |

### Phase 0 — status (2026-08-02)

Branch `vagen-lite-v0.8.0` @ `upstream/release/v0.8.0` (`bee9f6f4`), three commits:

```
644ab712 [vagen] test: correct the empty-mask expectation in value_mask tests
c37d62bc [vagen] feat: optional value_mask in value_loss
6215068e [vagen] fix: VLM-tolerant forward for trl value-head wrapper
```

Two deliberate improvements over a literal re-apply of the v0.6.1 patches:
- `valuehead_forward_value_only` lifted from a closure inside `apply_monkey_patch` to
  module level, so it is importable and unit-testable.
- It returns `lm_logits=None` instead of synthesising a zero `(bsz, seqlen, vocab)`
  tensor and upcasting it to fp32. Verified `FSDPEngineWithValueHead.prepare_model_outputs`
  (`transformer_impl.py:1322,1342`) is the sole consumer and reads `output[2]` only.
  The old behaviour burned ~2.4 GB per micro-batch for Qwen2.5-VL at seqlen 8k in bf16,
  plus ~4.8 GB after the upcast, purely to discard it.

**Verification** (conda env `verl`: torch 2.8.0+cu128, transformers 4.56.1, trl 0.26.2,
tensordict 0.10.0, pytest 9.1.0 + pytest-asyncio 1.4.0 installed during this phase):

| Check | Result |
|---|---|
| New tests | **14 passed** (8 valuehead + 6 value_mask) |
| Full CPU suite, our branch vs pristine `release/v0.8.0` | **failure sets byte-identical → zero regressions** |
| CPU suite totals | 418 passed, 27 skipped, 35 pre-existing failures |

Baseline frozen at `PHASE0_CPU_BASELINE.txt` — **diff against it at the end of every
later phase**; that is the concrete form of "matches baseline" in the exit criteria.

**Known pre-existing failures (all environmental, none ours):**
- 35 failures/errors: every one traces to the local model cache not existing, so HF
  treats local paths as repo ids and raises `HFValidationError`.
- 1 collection error, `tests/models/test_fused_kernels_ulysses_sp_on_cpu.py`:
  `No module named 'transformers.models.qwen3_5'`. 🔴 **The env is on transformers
  4.56.1; v0.8.0 pins 5.3.0.** Also vllm 0.11.0 vs 0.20.2 and sglang 0.5.2 vs 0.5.12.
  **Upgrading these is part of Phase 1 and is a real chunk of its risk** — the current
  green run does not prove v0.8.0 works on its intended dependency set.

### VLM families beyond Qwen (2026-08-03)

| model | mrope | image token | layout | steps | prompt = engine's |
|---|---|---|---|---|---|
| Qwen2.5-VL-3B | yes | `<\|image_pad\|>` | concat | 2/2 | ✅ |
| llava-interleave-0.5B | no | `<image>` | concat | 3/3 | ✅ |
| InternVL3-1B-hf | no | `<IMG_CONTEXT>` | concat | 3/3 | ✅ |
| InternVL3-1B-hf | no | `<IMG_CONTEXT>` | no-concat | 3/3 | ✅ |

LLaVA's zero score is a 0.5B model rambling to the response cap, not a pipeline fault.

**The two-form problem.** A multimodal prompt exists in two forms: the agent loop
tokenizes one with the HF processor, which expands an image into its full placeholder
run, and hands the engine a version with the runs collapsed, because the engine expands
them itself. Nothing forces the two expansions to agree. When they disagree the engine
samples from one sequence while training computes log-probs over another -- both
well-formed, neither side seeing both, the loss finite. Nothing reports it.

Closed by having the engine return the prompt it actually ran and adopting it, so the
training sequence *is* the sampling sequence by construction, for any family, with no
per-family expansion rules to keep in step. PR #6578 needs a ~200-line custom processor
to predict what the engine will do; asking it removes the need to predict.

Adoption differs per layout because the bookkeeping does:

* **no-concat** rebuilds its prompt each turn, so nothing has been measured against the
  old positions -- a straight replacement.
* **concat** carries one prompt across turns and measures its mask by appending counts,
  so the mask moves with the prompt. Tractable because only the newest observation can
  change length: earlier regions were adopted on a previous turn and are already in the
  engine's form, and the dedup preceding the round trip is the inverse of the engine's
  expansion, so re-expansion is idempotent. The delta lands entirely on a trailing run
  of zeros. That assumption is asserted after every adoption, not trusted.

⚠️ **Two corrections to earlier entries here.**

1. InternVL was written up as blocked on tiling configuration. It was not -- the
   placeholder-dedup fix cleared it. After a fix that plausibly explains several
   failures, re-run all of them, not just the one in hand.
2. InternVL was then written up as training end to end, 3/3 steps, score 0.156. It was
   training on a sequence two tokens longer than it sampled from, and only looked fine.
   The +2 was the `<img>`/`</img>` wrapper, which the deduplicated prompt already carried
   and vLLM's expansion added again. The length check caught it; adoption fixed it.

**Four defects behind the general case**, all of one shape -- something Qwen satisfied:

1. `hf_processor` raised for unrecognised processors, inside a broad `except`, so the
   processor became `None` -- indistinguishable from a text-only model.
2. `_compute_position_ids` called `get_rope_index` unconditionally.
3. **Placeholder dedup was gated on `Qwen2VLImageProcessor` appearing in the class
   name**, so every other family shipped the HF-expanded run and vLLM expanded it again.
4. The value-head forward did not request `return_dict`, which verl's fused-kernel
   forwards refuse.

Plus, VAGEN-side, the gym loops forwarded `mm_processor_kwargs` to neither the tokenizer
nor the engine, so a preprocessing knob configured only one half.

**Environment.** `mistral_common` must be `<1.10` -- 1.11 moved `ImageChunk` out of
`protocol.instruct.messages` while vLLM 0.11's `pixtral.py` still imports it there, which
makes `llava.py` unimportable. Pinned to 1.9.0; only surfaces once a LLaVA-family model
is used. transformers 4.56 also leaves `_no_split_modules` unset on InternVL and LLaVA,
so FSDP layer classes are named in config (`Qwen2DecoderLayer` plus the vision tower's).

**Relation to verl PR #6578** (open). It reaches the same two conclusions independently
-- 1D position ids for non-mrope models, and the two prompt forms -- and resolves the
second by predicting the engine's expansion per model. Its `llm_config` fallbacks in
`get_max_position_embeddings` and the `num_attention_heads` lookup are genuine gaps here
too, unhit only because `-hf` configs expose `text_config`; worth porting if original
`internvl_chat` checkpoints are ever needed.

### Legacy removed, validation validated (2026-08-03)

`vagen/ray_trainer.py` (1668 lines) and `vagen/agent_loop/agent_loop_no_concat.py` (825)
are gone. Both were verl 0.6 copies with small changes threaded through, which froze
those subsystems at 0.6. Everything they carried is now an override; verl's own
`_validate`, `_dump_generations` and `fit()` are used unmodified. Net across the branch:
**+2542 / −2747** over 14 commits.

Two pieces were dropped rather than ported, having become unreachable:
`_assign_group_and_traj_idx` (the manager does it, and earlier) and
`_post_process_no_concat_batch` with its `alignment_indices` helper. The second was
based on a wrong assumption of mine -- verl's `union` already tolerates the row-count
change, because `_get_gen_batch` leaves the tensor batch empty.

Validation had been off for every run until now, so the fold-back was the last code path
with only unit tests behind it. Both layouts now run it:

| layout | `val-core/sokoban/reward/mean@1` |
|---|---|
| no-concat (folded) | 0.6598 |
| concat (native) | 0.6586 |

That near-equality is the real check. Validation scores one policy on one val set, and
the layout governs only how *training* rows are formed, so the two should agree; a fold
that stitched trajectories together wrongly would show up as a gap. The instrumentation
confirms the fold is doing work rather than being a no-op: 32 validation rollouts
produced 119 rows, folded back to 32 for `_validate`.

### Algorithm layer — layout and algorithm decoupled (2026-08-03)

`no_concat_gae` conflated two things: a *layout* (one row per turn) and an *algorithm*
(turn-level GAE). They are orthogonal. `TrajectoryView` groups rows by
`(group_idx, traj_idx)` and orders them by `turn_idx`, so an estimator sees a trajectory
as an ordered list of rows plus the positions that are model output -- a concat
trajectory being one whose list has length one.

| estimator | layout | steps | score | vf_loss | adv/max |
|---|---|---|---|---|---|
| `gae` (verl) | concat | 5/5 | 0.275 → 0.231 | 6.95 → 1.43 | 4.87 |
| `no_concat_gae` (turn-level) | no-concat | 5/5 | 0.097 → 0.159 | 1.03 → 0.105 | 2.46 |
| `token_level_gae` | no-concat | 5/5 | 0.094 → 0.138 | 7.86 → 0.82 | 4.61 |
| `token_level_gae` | concat | 5/5 | 0.163 → 0.300 | 3.14 → 0.59 | 3.93 |
| `trajectory_grpo` | no-concat | 5/5 | 0.074 → 0.141 | n/a (no critic) | **1.732** |

`trajectory_grpo`'s 1.732 is an exact check rather than a plausible number: four trajectories
per group with one success gives mean 0.25 and population std 0.433, so the winner's
normalised advantage is (1 − 0.25)/0.433 = √3.

**Token-level needs no `value_mask`.** It supervises every model-output token, so
`critic/returns/min` is 0 rather than −100. The sentinel machinery serves turn-level GAE
alone -- a third of the grid, not all of it.

Three defects surfaced while wiring the grid, all of the same shape: something worked
under one layout by accident.

* `turn_idx` does not exist under concat. Defaulting it to zero is what lets one
  estimator serve both.
* verl's `_postprocess` drops `input_non_tensor_batch` entirely when streaming reward is
  enabled. no-concat survived because its loop emits `group_idx` per turn through
  `extra_fields`; concat did not, so the estimators raised `KeyError` under concat only.
* Validation needs one output row per input row, which the split layout breaks. Folding
  the turns back in the manager keeps verl's `_validate` usable unmodified.

**Observability.** How many rows a rollout produced was invisible: every no-concat row
reports `num_turns=1` by construction, and a batch of one-turn episodes has the same
shape as a batch of multi-turn ones. Inferring it from `perf/total_num_tokens` produced a
*wrong* conclusion -- that the fan-out was not happening at all. The worker now prints
the distribution directly, which shows 1–5 rows per episode as intended.

### ✅ v0.8.0 path validated end-to-end (2026-08-03)

Sokoban, Qwen2.5-VL-3B, 4× B200, vllm, 5 steps each. Both legs clean.

| | concat | no-concat |
|---|---|---|
| steps | 5/5 | 5/5 |
| `critic/vf_loss` | 6.9 → 1.4 | 1.03 → 0.105 |
| `critic/vpred_mean` | −3.0 → 1.1 | oscillates ±2, ends 0.17 |
| `critic/returns/min` | 0 | **−100** |
| `num_turns/mean` | ~4.8 | **1** |

The no-concat leg is the decisive one. `returns/min = −100` shows the sentinels are
present, while `vf_loss` stays at O(1) and `vpred_mean` oscillates around zero instead of
running toward the sentinel -- so `value_mask` reaches the critic on this path. Together
with `num_turns = 1` per row it confirms the whole chain: multi-output flattening, the
group/traj/turn bookkeeping, the adapted estimator signature, and the verl-side
`value_loss` patch.

**Nine blockers cleared to get here**, of which two were upstream defects
(`_create_critic_class` reading pre-refactor config names; `compute_advantage` handing
the raw containers only to a hard-coded GDPO) and one was an environment race (agent loop
workers calling `copy_to_local` concurrently, whose only symptom was a much later
`processor is None` because `hf_processor` swallows the exception).

Engine note: v0.8.0's sglang extra pins `sglang==0.5.8` **and `torch==2.9.1`**, which this
env cannot satisfy without a full rebuild. vllm 0.11.0 is already inside the supported
range, and the code under test is engine-agnostic, so the validation runs on vllm. The
sglang upgrade belongs in a fresh env, not this one.

Still gating deletion of the legacy files: `_validate` and `_flush_image_dumps` were off
for these runs and remain unported.

### Phase 1 — status (2026-08-03)

**Three trainer lineages coexist in v0.8.0.** This decides where the mixin binds:

| entrypoint | trainer | `_fit_*` hooks |
|---|---|---|
| `main_ppo.py` | `RayPPOTrainer` | no — monolithic `fit()`; **`@deprecated`** |
| `main_ppo_sync.py` | `PPOTrainer` (standalone, TransferQueue + ReplayBuffer) | no |
| async paths | `SeparateRayPPOTrainer(RayPPOTrainer)` | **yes**, ~20 of them |

The hooks exist *only* on `SeparateRayPPOTrainer` (`experimental/separation/ray_trainer.py`),
which is also what `OneStepOffRayTrainer` / `FullyAsyncTrainer` / `FullyAsyncRollouter`
inherit. So `VagenPPOTrainer(VagenV0Mixin, SeparateRayPPOTrainer)` is the right base —
it is simultaneously the hook-bearing class and the async lineage. Note this diverges
from the new default sync entrypoint, which is a standalone class with no hooks.

⚠️ Hook signatures are **not** uniform across that lineage: `FullyAsyncTrainer` declares
`_fit_save_checkpoint(self, force=False)` and calls it with `force=True`, while
`SeparateRayPPOTrainer` declares it no-arg. Overrides must forward `*args/**kwargs` or
they break the composition they are supposed to enable.

**Landed**

| item | result |
|---|---|
| `MultiOutputAgentLoopWorker` / `Manager` | 2 method overrides, no copied upstream code |
| gym loops → v0.8.0 `AgentLoopBase` | `init_class` removed upstream, so both loops' copies were dead code |
| shared `VagenGymAgentLoopBase` | module-level caches replace the class-level `init_class` ones |
| no-concat index bookkeeping | pure core in `logic.py` + guards the originals lacked |
| mixin `_fit_save_checkpoint` | forwards `*args/**kwargs` |
| tests | 67 VAGEN + 16 verl-side, all passing |

Two things made the agent-loop replacement small: nothing between `agent_loop.run()` and
`_agent_loop_postprocess` inspects the returned value, so a list can travel unnoticed to
a single flattening point; and `AgentLoopManager.__init__` guards its worker class with
`if not hasattr(...)`, an explicit subclass hook. `generate()` and the `_target_`
registration format were unchanged between 0.6 and 0.8, so no call sites moved.

**Remaining cutover** — `vagen/ray_trainer.py` (1668 lines) and
`agent_loop/agent_loop_no_concat.py` (825) are still on disk because `main_ppo.py`
imports the former. Inventory of the vendored trainer: 14 methods are overrides of
upstream (mostly verbatim copies) and 7 are VAGEN-only, of which `_validate`,
`_flush_image_dumps` and the resource-pool helpers still need porting. The two that
turn-level GAE depends on are already extracted.

Not yet validated on GPUs: the v0.8.0 training path end-to-end. The baselines above all
ran on the pre-0.8 stack. A short sokoban smoke test is the gate for deleting the two
legacy files.

### Environment findings (2026-08-02, an 8-GPU host)

Bringing up an actual sokoban run surfaced several things that block Phase 1:

**Stale installs.** Both editable installs in the conda `verl` env point at
an older checkout path, **which no longer exists**. `import verl` therefore
only resolves when cwd happens to contain a `verl/` directory. Worse, `VAGEN/verl` is an
uninitialised submodule (empty dir) that Python treats as a namespace package and which
**shadows the real verl** whenever you run from the VAGEN root — and VAGEN's hydra
searchpath (`file:../../verl/verl/trainer/config`) points into that same empty dir. Either
initialise the submodule or run from a neutral cwd with `PYTHONPATH` + a
`hydra.searchpath` override; `baseline_runs/run.sh` does the latter.

**Missing dependencies** (installed during this session, all via `pip`):

| Package | Why | Note |
|---|---|---|
| `pytest-asyncio` 1.4.0 | in verl's `requirements-test.txt`; without it 40 CPU tests fail spuriously | |
| `torchao` **0.13.0** | sglang imports `float8_dynamic_activation_float8_weight` unconditionally in `apply_torchao_config_to_model` | 🔴 **pin it** — 0.17.0 renamed the symbol and breaks sglang 0.5.2 |
| `ninja` | flashinfer JIT-compiles kernels at first use | must be **on `PATH`**, not just importable |

**B200 specifics** — sglang's default `fa3` attention backend is unsupported on Blackwell.
VAGEN's `docs/issues.md` already documents the fix; it must be on every command line:
```
+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer
+actor_rollout_ref.rollout.engine_kwargs.sglang.mm_attention_backend=triton_attn
```

🔴 **Two verl runs cannot share a node.** `verl/workers/rollout/utils.py::get_free_port`
binds an ephemeral port with `SO_REUSEPORT`, so two concurrent runs can be handed the
*same* port; sglang's TCPStore then binds it without `SO_REUSEPORT` and dies with
`EADDRINUSE`. Observed directly when running two 4-GPU jobs on 8 GPUs. **Run baselines
sequentially** — do not assume 8 GPUs means two parallel jobs.

**`verl080` env — built, per `docker/Dockerfile.stable.sglang`.** Kept separate from `verl`
on purpose: upgrading in place would destroy the ability to reproduce the baseline, which
runs the *old* stack (verl @ `3fe0a299`, transformers 4.56.1).

```
torch 2.11.0+cu130 · transformers 5.3.0 · trl 0.27.0 · sglang 0.5.12
tensordict 0.13.0 · ray 2.56.1 · hydra 1.3.4 · kernels 0.10.5 · torchao 0.17.0
```

Two conflicts worth recording, both resolved the way verl itself resolves them:

- 🔴 **`kernels` must be pinned `<0.11`.** transformers 5.3.0 declares
  `kernels<0.11,>=0.10.2`, but only inside *extras*, so pip happily installed 0.16.0 —
  which then rejects transformers' own `LayerRepository(...)` construction with
  `ValueError: Either a revision or a version must be specified`, breaking
  `import transformers` (and therefore `import sglang`) entirely.
- **sglang 0.5.12 declares `transformers==5.6.0` and `torchao==0.17.0`.** verl's dockerfile
  deliberately overrides the former (`FROM lmsysorg/sglang:v0.5.12` then
  `RUN pip install transformers==5.3.0`), so pip's conflict warning here is expected, not a
  mistake. torchao does follow sglang (0.17.0) — note this is the opposite of the *old*
  env, where sglang 0.5.2 needs torchao **0.13.0**; the symbol
  `float8_dynamic_activation_float8_weight` moved between the two.

### 🔴 Live bug found by the baseline: `value_mask` never reached the critic

Two independent layers, both required; fixing either alone changes nothing.

1. **Predicate never matched.** `ray_trainer.py:1516` gated on
   `adv_estimator in ["no_concat_gae_last", "no_concat_gae_first"]`, but the registered
   names are `no_concat_gae_last` and `no_concat_gae` — `..._first` has never existed,
   and `no_concat_gae` is what every no-concat script passes.
2. **Key filtered out one layer down.** Even once written, `dp_critic.py:197` selects a
   whitelist (`input_ids, responses, response_mask, attention_mask, position_ids, values,
   returns`) before the micro-batch loop, so `value_mask` was dropped and
   `.get("value_mask", response_mask)` fell back to the full mask.

Consequence: turn-level GAE writes a real return at one anchor token per turn and leaves
the rest at the −100 sentinel, so **the critic was trained to predict −100 almost
everywhere.**

A/B on sokoban (identical config, 3B, 4× B200, only value_mask differs):

| metric | buggy | fixed |
|---|---|---|
| `critic/vf_loss` | 568 → 5.7e-5 | **1.3 → ~0.01** |
| `critic/vpred_mean` (final) | **−98.76** | **0.39** |
| `critic/vf_explained_var` | −0.016 → **0.94** | ~0 |

`vpred_mean = −98.76` after 100 steps is the direct evidence: the critic converged
almost exactly onto the −100 sentinel, and `vf_loss` 5.7e-5 is it fitting that constant
perfectly.

`vpred_mean → −77.6` is the direct evidence: the critic's own predictions were racing
towards the sentinel.

⚠️ **Monitoring trap.** The broken run's `vf_explained_var` reads **0.94** — it looks
excellent. It is explaining the variance of a constant-dominated target (mostly −100),
which is trivially explainable. The healthy run sits near 0, which is honest for 9 steps.
**Watching explained_var alone would never have surfaced this.** Watch `vpred_mean` and
the absolute scale of `vf_loss` instead.

Fixed structurally rather than by correcting a string: estimators declare that they emit
sentinels at their registration site (`custom_advantage/registry.py`), and
`needs_value_mask()` reads that registry — the predicate cannot drift from the
registration again. On v0.8.0 the second layer is already handled (Phase 0 added
`"value_mask"` to `losses.py::value_loss`'s `select`, and nothing else on that path
filters keys — verified `_update_critic` dispatches the whole TensorDict and
`left_right_2_no_padding` only nests `input_ids`/`position_ids`/`routed_experts`/
`teacher_*`, leaving `values`/`returns`/`response_mask`/`value_mask` dense and
width-aligned).

🔴 **All previous no-concat PPO results need re-evaluating** — the critic was broken, so
the advantages were too.

**Milestones**

| | Phases | Days | Exit |
|---|---|---|---|
| **M1 — upgrade** | 0–2 | 10.5–12.5 | existing experiments reproduce on v0.8.0; InternVL trains; 825 lines gone |
| **M2 — decouple + algorithms** | 3–5 | 10.5 | concat & no-concat share one harness; a new advantage estimator is ~20 lines |

**Notes**
- Phase 1 dominates and is the least predictable; a bad baseline comparison can double it. Everything after is mostly mechanical.
- Phases 3–5 are largely *extraction*: incremental tokenization already exists in `gym_agent_loop.py`, and the multi-turn loop exists twice. The risk is behavioural drift — that is what the baseline comparisons are for.
- M1 alone already delivers value: current features on v0.8.0, InternVL, 825 fewer lines.
- Duplicate code removed: `gym_agent_loop*.py` (731) + `agent_loop_no_concat.py` (825) + `no_concat_gae` bulk (~300) ≈ **1850 lines**, against roughly **700 lines** of new shared code.

---

## Appendix · verl 0.8 upgrade cheat sheet

**Where we are**: `d7426381` (2025-11-14, ≈ v0.6.1) + 4 commits. **Target**: `release/v0.8.0` HEAD `bee9f6f4`. Gap: **1008 commits, 422 files under `verl/`**.

⚠️ `release/v0.8.0` **is not an ancestor of `main`** — it carries 40 fixes main lacks (#7077, #7065, #7031, #7026, #6980). **Pin the branch head, not the tag.** main is 250 commits ahead with 4 more BREAKING changes and moves daily; there is no v0.9 branch yet.

**Our 5 patches**: 3 landed upstream and are deletable (VLM `hidden_size`; sglang & vllm per-turn `max_tokens`). VLM ValueHead `forward` stays (worth upstreaming). `value_mask` moves (§8.4).

> ⚠️ **One semantic difference in the per-turn `max_tokens` patch that did land upstream.**
> Ours clamped an explicitly-passed per-turn budget by `config.response_length`:
> `min(max_new_tokens, response_length, model_headroom)`.
> Upstream only clamps in its *default* branch — an explicit `max_new_tokens` is passed
> through and clamped by model context alone (`vllm_async_server.py`, `async_sglang_server.py`).
> Benign for every current VAGEN config (per-turn `response_length_per_turn` ≤
> `data.max_response_length` everywhere), and our agent loops never pass `None`, so the
> upstream `min(None, …)` hazard does not apply either. But if `response_length_per_turn`
> is ever set above `max_response_length`, generation now overruns the response tensor
> width and the agent loop silently truncates it (`response_mask[: self.response_length]`)
> — wasted compute plus dropped tokens, with no error.
> **Fix at config-validation time in VAGEN (Phase 1), not by re-patching verl** — the goal
> is fewer fork patches, and this belongs to our config anyway.

**Will ImportError**: `workers.fsdp_workers` / `megatron_workers` / `workers.roles` → `workers.engine_workers`. Deleted outright: `experimental.dataset.sampler` (`AbstractSampler`, `AbstractCurriculumSampler`), `dynamicgen_dataset`, `prometheus_utils`, `trainer.ppo.reward.compute_reward(_async)`, `experimental.reward.RewardManagerWorker`, `transferqueue_utils.create_transferqueue_client`. Moved: `ResourcePoolManager` → `single_controller.ray`.
**API**: `AgentLoopBase.__init__` gains required `dataset_cls` / `data_config`; `init_class()` deleted; `AgentLoopManager` → `await .create(config, llm_client=…)`.

**Config**: `critic.model.fsdp_config.*` → **`critic.fsdp.*`**; `rollout.skip_rollout` → `rollout.skip.enable`; `reward_model.*` → `reward.*`.

**VLM**: v0.8.0 already made the multimodal path registry-driven (`build_multimodal_processor_inputs` + `processor.get_rope_index`). Registry at `utils/tokenizer.py:205` covers Qwen2/2.5/3-VL, GLM4V, Kimi-VL — **no InternVL (main has none either)**.
🔴 v0.8.0's GLM4V entry is broken: it matches `Glm4vImageProcessor`, but the top-level class is `Glm4vProcessor`, so it falls to `case _` → raises → gets swallowed by `try/except` → `processor = None` → **silently degrades to text-only training**. Fixed on main by #6873.

```bash
git remote add upstream https://github.com/verl-project/verl.git
git fetch upstream 'refs/heads/release/v0.8.0:refs/remotes/upstream/release/v0.8.0' main
git diff d62da495 3fe0a299                                        # our patches
git log --format='%h|%ad|%s' --date=short v0.6.0..v0.8.0 | grep -i BREAKING
```
Comparison worktrees already created: `/tmp/verlwt/{v060, pin, v080}`

**Key source locations**
```
experimental/separation/ray_trainer.py          SeparateRayPPOTrainer + ~20 _fit_* hooks
experimental/agent_loop/agent_loop.py           :272 apply_chat_template(remove_system_prompt=)
                                                :600 single-output hard-code (we override)
                                                :631 _agent_loop_postprocess (padding/pos_ids/mm)
                                                :1074 agent_loop_workers_class subclass hook
experimental/agent_loop/tool_agent_loop.py      the incremental pattern TokenTape copies
utils/chat_template.py                          initialize_system_prompt / extract_system_prompt_and_generation
workers/config/rollout.py                       :93 agent_loop_config_path  :97 manager_class
workers/utils/losses.py                         :57 ppo_loss  :147 value_loss (value_mask lands here)
models/transformers/monkey_patch.py:348         ValueHead patch site
utils/tokenizer.py                              :102 build_multimodal_processor_inputs  :205 registry
```

---

## 12. Token accounting · `harness/budget.py`

Six quantities bound a run. Three are enforced by something, three were not until
2026-08-06:

    C     rollout.max_model_len          hard. The engine refuses past it.
    n_p   data.max_prompt_length         a row's prompt region
    n_r   data.max_response_length       a row's response region
    g     response_length_per_turn       one generation. Hard: it is max_new_tokens.
    E     env_response_length            one observation, after the processor has
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
