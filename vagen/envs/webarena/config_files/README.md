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

Hosted on Google Drive:
https://drive.google.com/drive/folders/1gvLX9CZ29ELWKeoMcequ51RgNmpjLJyy?usp=sharing

Download all three JSONs into this directory:

```
vagen/envs/webarena/config_files/
├── normalized_test.json
├── normalized_test_nomap.json
└── normalized_train.json
```

You can fetch them programmatically with `gdown`:

```bash
pip install gdown
cd vagen/envs/webarena/config_files
gdown --folder https://drive.google.com/drive/folders/1gvLX9CZ29ELWKeoMcequ51RgNmpjLJyy
```
