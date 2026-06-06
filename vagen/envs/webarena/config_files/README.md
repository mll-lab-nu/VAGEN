# Task config files

Three JSON corpora used by the WebArena env to look up per-seed tasks.
Kept out of git (large + just data); fetch them from the hosting location
below before running the env.

| File | Tasks | Sites | Use |
|---|---:|---|---|
| `normalized_test.json` | 165 | shopping_admin 36, map 31, shopping 46, reddit 24, gitlab 34, wikipedia 4 | Full WebArena-Lite test set (BC paper's baseline) |
| `normalized_test_nomap.json` | 134 | (above minus map) | Same test set with the 31 map tasks dropped — convenient when the map docker (port 3000) is not deployed |
| `normalized_train.json` | 647 | shopping_admin 148, map 97, shopping 146, reddit 105, gitlab 170, wikipedia 19 | Training task pool for PPO/GRPO RL |

## Where to get them

**TODO**: pick one and update this section once uploaded:

- [ ] Hugging Face Hub: `huggingface.co/datasets/<org>/vagen-webarena-tasks`
- [ ] GitHub Release asset on `mll-lab-nu/VAGEN`
- [ ] Internal S3 / shared drive

Once uploaded, the canonical fetch should be one shell command that
drops the three JSONs into this directory.
