# Repository structure rules

This document defines the durable package boundaries for VAGEN. Paths should make two
facts clear: which subsystem owns code, and whether the code is a shared contract or a
selectable implementation.

## Invariants and extension axes

Stable orchestration belongs in specifically named modules. The main invariant packages
are `rollout`, `training`, and `evaluation`.

A family that expects multiple selectable implementations is an extension axis:

```text
axis/
├── __init__.py              # public facade and registry
├── _common/                 # contracts and genuinely shared helpers
└── implementation/
    ├── __init__.py
    └── implementation.py
```

`_common` must not import a concrete implementation. Core orchestration imports an axis
through its public facade rather than importing a concrete implementation directly.
The implementation directory must also own the selectable implementation's actual
control flow. A registered function that only forwards to `_common._compute_*` is not a
real implementation boundary; keep only genuinely reused primitives in `_common`.

## Canonical package layout

```text
vagen/
├── algorithms/
│   ├── _common/
│   └── <algorithm>/
├── envs/
│   ├── _common/
│   └── <environment>/
├── harness/
│   ├── _common/
│   └── <harness>/
├── models/
│   ├── _common/
│   └── <model_family>/
├── rollout/                 # framework-independent recording and lifecycle seams
│   ├── client.py
│   ├── runner.py
│   ├── scoring.py
│   └── trajectory.py
├── evaluation/
│   └── backends/
│       ├── _common/
│       └── <backend>/
└── training/                # VERL-specific integration
    ├── agent_loop/
    ├── trainer/
    ├── filters/
    ├── metrics/
    └── losses/
```

The responsibilities are:

- `algorithms`: advantage estimators and their declared training capabilities.
- `envs`: environment contracts, shared wrappers, remote protocol, and implementations.
- `harness`: complete reusable agent loops. Each implementation owns
  `run_episode(client, env)` and is shared unchanged by training and evaluation.
- `models`: model-family rendering and multimodal token handling. Chat-template details
  belong here, never in a VERL or evaluation transport.
- `rollout`: backend-neutral message routing, token/trajectory recording, reward
  attribution, and environment lifetime management.
- `evaluation`: configuration, execution, recording, summaries, and backend plugins.
- `training`: integration with VERL workers, datasets, trainers, losses, filters, and
  training-only metrics.

The `verl/` checkout is an external submodule. VAGEN may adapt its public interfaces but
must not treat it as part of the `vagen` package hierarchy.

## Dependency direction

Dependencies flow toward contracts:

```text
training -----> rollout <----- evaluation
   |               |               |
   +----------> public axis facades+
                    |
             concrete implementation -> axis/_common
```

- Training-only packages must not be imported by standalone evaluation.
- Environment implementations may import `envs._common`, not another environment.
- Harness implementations may import `harness._common` and stable rollout contracts,
  not training or evaluation. They use `client.create(messages)` and `client.size(...)`;
  they do not manipulate conversation ids or training rows.
- Rollout orchestration selects environments, harnesses, and models through their
  facades or registries.
- Configuration should use registered names. Dynamic Python paths are supported only
  where they are an intentional public extension mechanism.

## Runtime boundaries

- `training/agent_loop/verl_client.py` is a VERL transport only. Model-family rendering
  is selected through `vagen.models`.
- `evaluation/backends/` contains hosted-provider transports. Its contract is named
  `EvaluationBackend` so it cannot be confused with `vagen.models.ModelAdapter`.
- `envs/_common/adapter.py` is the compatibility boundary for existing text-oriented gym
  environments. Harnesses pass a complete response object; the adapter unwraps text and
  token metadata for legacy implementations.
- `rollout/scoring.py` is the mandatory reward seam. It associates a reward with the
  exact model `call_id`, so a harness cannot accidentally omit or misfile credit.

The former `core`, `agent_loop`, `trainer`, `custom_advantage`, `custom_filter`,
`custom_metric`, `custom_loss`, `evaluate`, and `envs_remote` package trees have been
removed. Do not recreate them as compatibility mirrors.

## Repository-level content

```text
examples/<phase>/<environment>/  # maintained configs and launchers
docs/common/                     # durable project rules
docs/dates/YY-MM-DD/             # dated reports and incidents
exps/<project>/<run>/            # ignored generated artifacts
```

Generated checkpoints, rollout dumps, W&B files, caches, and evaluation results must not
be committed or placed inside source packages.

## Change checklist

When adding or moving an implementation:

1. Classify it as invariant code, shared axis code, or one concrete implementation.
2. Update the axis facade and registry.
3. Update imports, dynamic import strings, Hydra targets, worker `external_lib` values,
   CLI entry points, tests, examples, and documentation together.
4. Preserve optional-dependency lazy loading; one unavailable environment or backend
   must not make unrelated implementations unavailable.
5. Run registry/import smoke tests, focused tests, the full unit suite, and
   `git diff --check`.
