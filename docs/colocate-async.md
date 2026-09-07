# Colocated asynchronous training

VAGEN preserves its established V0 trainer by default. Setting `trainer.use_v1=true`
selects verl V1's `colocate_async` scheduler: rollouts keep running in the background;
as soon as enough prompt groups finish, the trainer samples them from TransferQueue and
starts the next update instead of waiting for the slowest episode.

## Boundary

The scheduler changes how completed episodes move into training, not how an episode is
executed:

```text
colocate_async scheduler
        -> GymLoop / run_episode
        -> OpenAI-like harness and rollout client
        -> environment (the token-reward producer)
        -> TransferQueue adapter
        -> VAGEN advantage estimator and PPO update
```

Harnesses stay backend-neutral and do not import torch, verl, or TransferQueue. The
environment's reward implementation is the one intentional token-level exception: it
may consume the engine's response token ids and tokenizer to return an already aligned
`per_token_reward`. The adapter validates and transports that vector unchanged; it does
not re-tokenize text or recompute reward.

Every completed conversation row carries its episode/group/trajectory/turn identity,
response spans, images, and policy-version interval. The V1 trainer materializes those
columns before computing advantages and writes `advantages`, `returns`, `value_mask`,
and `turn_id` back to TransferQueue.

## Configuration

The shipped defaults keep V0 selected:

```yaml
trainer:
  use_v1: false

transfer_queue:
  enable: false

actor_rollout_ref:
  rollout:
    free_cache_engine: false
    agent:
      agent_loop_manager_class: vagen.training.agent_loop.multi_output.MultiOutputAgentLoopManager
```

Opt into colocate_async with one command-line override:

```bash
bash examples/train/sokoban/train_default_gae_qwen25vl3b.sh trainer.use_v1=true
```

The entrypoint derives the complete V1 configuration from that knob:

```yaml
trainer:
  use_v1: true
  v1:
    trainer_mode: colocate_async
    colocate_async:
      num_warmup_batches: 1
      max_inflight_steps: 2

transfer_queue:
  enable: true

actor_rollout_ref:
  rollout:
    free_cache_engine: true
    agent:
      agent_loop_manager_class: vagen.training.agent_loop.tq.VagenAgentLoopManagerTQ
```

`TransferQueue>=0.1.9` is required. That is the first supported release with queue
checkpoint/restore, so finished rows and in-flight prompt markers remain consistent with
a resumed trainer checkpoint.

`max_inflight_steps` limits a still-running training group by policy versions. At the
default value of 2, a group may span two versions; on the third it is cancelled as one
atomic UID group, marked failed, evicted, and replaced by the async replay buffer. It is
separate from `trainer.v1.sampler.max_off_policy_threshold`, which governs already
completed groups. Validation is never cancelled by this limit.

An empty episode is also a failed group. This matters for GRPO: no sibling trajectory
from that UID is trained after one session fails or produces no row.

## Filters and fallback

The built-in `reward_variance` and `reward_variance_top_p` filters run on completed TQ
keys before actor/critic model work. `reward_variance_top_p` keeps its advantage scaling.
A custom legacy `DataProto` filter is not automatically safe in V1; implement it as a
custom V1 replay-buffer sampler so it can select keys before model work starts.

Switch back to the default V0 path explicitly with:

```bash
trainer.use_v1=false
```

The entrypoint then disables TransferQueue, restores `MultiOutputAgentLoopManager`, and
turns `free_cache_engine` off. Both modes execute the same `GymLoop` and
`run_episode()` implementation.
