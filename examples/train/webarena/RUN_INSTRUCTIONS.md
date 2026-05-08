# Running WebAgent-R1 (Qwen2.5-3B SFT + GRPO) on 2× RTX PRO 6000 Blackwell

End-to-end runbook. Assumes the smoke-test fixes from earlier are still in
place: verl `.pth` patched, `VAGEN-WEBAGENT/verl/` symlink, etc.

## 1. SSH tunnel (login node)

The webarena Docker host is remote. Tunnel six ports to localhost:

```bash
ssh -i ~/.ssh/id_ed25519 -p <REMOTE_PORT> -o StrictHostKeyChecking=accept-new \
    root@<REMOTE_HOST> \
    -L 7770:localhost:7770 -L 7780:localhost:7780 \
    -L 9999:localhost:9999 -L 8023:localhost:8023 \
    -L 8888:localhost:8888 -L 4399:localhost:4399 \
    -N &
```

Port 3000 (map) is intentionally NOT tunneled. The yaml's `seed_list`
already excludes all 97 map tasks from training and 31 map tasks from val.

## 2. Two webarena env servers (login node)

Train server uses `normalized_train.json` (647 tasks); val server uses
`normalized_test.json` (165 tasks). They must be separate processes
because the handler locks the task file at startup.

Always **kill leaked sessions + chromium first**, otherwise an old context
holding a browser thread will deadlock new connects on the same browser.

```bash
# Clean
pkill -9 -f "vagen.envs.webarena.serve" || true
pkill -9 -u $USER -f chromium || true
sleep 3

cd /work/nvme/bgig/ryu4/VAGEN-WEBAGENT
source /u/ryu4/miniconda3/etc/profile.d/conda.sh
conda activate webarena
source vagen/envs/webarena/setup_vars.sh

# Train server (port 8002, 8x8=64 contexts)
nohup env PYTHONPATH=. python -m vagen.envs.webarena.serve \
  --task_config_file=vagen/envs/webarena/config_files/normalized_train.json \
  --n_browsers=8 --max_contexts_per_browser=8 \
  --port=8002 --auth_cache_dir=./.wa_auth \
  > examples/evaluate/webarena/logs/webarena_train_server.log 2>&1 &

# Val server (port 8003, 4x8=32 contexts)
nohup env PYTHONPATH=. python -m vagen.envs.webarena.serve \
  --task_config_file=vagen/envs/webarena/config_files/normalized_test.json \
  --n_browsers=4 --max_contexts_per_browser=8 \
  --port=8003 --auth_cache_dir=./.wa_auth \
  > examples/evaluate/webarena/logs/webarena_val_server.log 2>&1 &

# Wait for ready
until curl -s --fail http://localhost:8002/health >/dev/null \
   && curl -s --fail http://localhost:8003/health >/dev/null; do sleep 3; done
echo "both servers ready"
```

## 3. SLURM job (24h, 2× RTX PRO 6000)

```bash
salloc --partition=<your_partition> \
       --time=24:00:00 \
       --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=192G \
       --gres=gpu:rtx_pro_6000:2
```

Or sbatch equivalent. Get the JOBID + NODELIST from `squeue -u $USER`.

## 4. Verify GPU node can reach login03

```bash
srun --jobid=<jobid> --overlap bash -c '
  curl -s -o /dev/null -w "train server %{http_code}\n" http://dt-login03:8002/health
  curl -s -o /dev/null -w "val server %{http_code}\n"   http://dt-login03:8003/health
'
```
Both should return 200.

## 5. Launch training

```bash
export WANDB_API_KEY=<your_key>   # or remove 'wandb' from logger in the sh

srun --jobid=<jobid> --overlap bash -c "
  source /u/ryu4/miniconda3/etc/profile.d/conda.sh
  conda activate vagen
  export WANDB_API_KEY=$WANDB_API_KEY
  cd /work/nvme/bgig/ryu4/VAGEN-WEBAGENT
  exec bash examples/train/webarena/train_webarena_grpo_qwen25_3b.sh
" 2>&1 | tee /work/nvme/bgig/ryu4/VAGEN-WEBAGENT/examples/train/webarena/train_full.log
```

## 6. Resume from checkpoint after interruption

verl auto-resumes if `actor.checkpoint.save_contents` includes `optimizer`
and `extra`. On restart, it reads `latest_checkpointed_iteration.txt`
under `default_local_dir`. Just re-run the same command — it will pick
up where it left off (with one wasted rollout for the first new step).

## 7. Hourly health checks (recommended for long runs)

In a separate shell, run:
```bash
while true; do
  echo "=== $(date) ==="
  curl -s http://localhost:8002/health -o /dev/null -w 'train: %{http_code}\n'
  curl -s http://localhost:8003/health -o /dev/null -w 'val:   %{http_code}\n'
  ssh root@<REMOTE_HOST> 'docker ps --format {{.Names}}\\t{{.Status}}' 2>&1 | head -10
  sleep 1800
done
```
