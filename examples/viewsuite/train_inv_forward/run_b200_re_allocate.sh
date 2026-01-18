set -euo pipefail
bash /home/ubuntu/patch.sh
bash /home/ubuntu/projects/viewsuite/ViewSuite/scripts/download_checkpoints.sh
bash /home/ubuntu/projects/viewsuite/VAGEN/examples/viewsuite/train_inv_forward/train_grpo_qwen25vl7b_b200_424.sh