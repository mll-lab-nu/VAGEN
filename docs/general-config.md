# Configuration Explanation

We use the service-based architecture as an enhanced approach for managing environments in VAGEN. This document explains the key configuration parameters for setting up and running experiments with the service architecture.

## Experiment Configuration
### Algorithm
```
algorithm.adv_estimator=bi_level_gae
algorithm.high_level_gamma=1.0
algorithm.kl_ctrl.kl_coef=0.001
```
`algorithm`:

- `adv_estimator`: Sets the advantage estimation method. Set to `bi_level_gae` for TRICO's cross-turn credit assignment.
- `high_level_gamma`: Discount factor for turn-level advantage calculations. Value of 1.0 means no discount across turns.
- `kl_ctrl.kl_coef`: Coefficient for KL divergence penalty. Controls policy deviation from reference model. Default 0.001

### Data
```
data.train_files=data/svg-vision-debug/train.parquet
data.val_files=data/svg-vision-debug/test.parquet
data.train_batch_size=16
data.max_prompt_length=1024
data.max_response_length=648
data.max_trajectory_length=1800
data.image_key=images
data.truncation=error
```

`data`:

- `train_files`: Path to training data in parquet format.
- `val_files`: Path to validation data.
- `train_batch_size`: Number of training examples per batch.
- `max_prompt_length`: Maximum token length for environment prompts.
- `max_response_length`: Maximum token length for model responses.
- `max_trajectory_length`: Maximum combined length of full interaction trajectory.
- `image_key`: Key used to access image data in inputs.
- `truncation`: Behavior when sequences exceed maximum length.

### Actor-Rollout-Reference Model
```
actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-3B-Instruct
actor_rollout_ref.model.use_remove_padding=True
actor_rollout_ref.model.enable_gradient_checkpointing=True
```

`actor_rollout_ref.model`:

- `path`: Base VLM model path.
- `use_remove_padding`: Whether to skip computation on padded tokens.
- `enable_gradient_checkpointing`: Trades computation time for reduced memory usage.

```
actor_rollout_ref.actor.optim.lr=1e-6
actor_rollout_ref.actor.ppo_mini_batch_size=32
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
actor_rollout_ref.actor.use_kl_loss=False
actor_rollout_ref.actor.kl_loss_coef=0.001
actor_rollout_ref.actor.kl_loss_type=mse
```
`actor_rollout_ref.actor`:

- `optim.lr`: Actor model learning rate.
- `ppo_mini_batch_size`: Number of samples per PPO update batch.
- `ppo_micro_batch_size_per_gpu`: Micro-batch size per GPU for actor updates.
- `use_kl_loss`: Whether to use KL loss in actor updates.
- `kl_loss_coef`: Weight for KL loss term if enabled.
- `kl_loss_type`: Type of KL loss calculation.

```
actor_rollout_ref.actor.fsdp_config.param_offload=False
actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
```
`actor_rollout_ref.actor.fsdp_config`:

- `param_offload`: Whether to offload parameters to CPU.
- `optimizer_offload`: Whether to offload optimizer states.

```
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
actor_rollout_ref.rollout.tensor_model_parallel_size=2
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.gpu_memory_utilization=0.2
actor_rollout_ref.rollout.enable_chunked_prefill=False
actor_rollout_ref.rollout.enforce_eager=False
actor_rollout_ref.rollout.free_cache_engine=False
actor_rollout_ref.rollout.n=1
actor_rollout_ref.rollout.top_p=0.95
actor_rollout_ref.rollout.temperature=0.7
```
`actor_rollout_ref.rollout`:

- `log_prob_micro_batch_size_per_gpu`: Micro-batch size for log probability calculations.
- `tensor_model_parallel_size`: Number of GPUs for tensor parallelism.
- `name`: Backend implementation for model deployment.
- `gpu_memory_utilization`: Target GPU memory utilization.
- `enable_chunked_prefill`: Whether to enable chunked context processing.
- `enforce_eager`: Whether to use eager execution mode.
- `free_cache_engine`: Whether to aggressively free cache.
- `n`: Number of rollout sequences per input.
- `top_p`: Nucleus sampling parameter for controlling output diversity.
- `temperature`: Sampling temperature for generation randomness.

```
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
actor_rollout_ref.ref.fsdp_config.param_offload=True
```
`actor_rollout_ref.ref`:

- `log_prob_micro_batch_size_per_gpu`: Micro-batch size for reference model.
- `fsdp_config.param_offload`: Whether to offload reference model parameters.

### Critic Model

```
critic.optim.lr=1e-5
critic.ppo_micro_batch_size_per_gpu=1
```
`critic`:

* `optim.lr`: Critic model learning rate. Typically higher than actor learning rate.
* `ppo_micro_batch_size_per_gpu`: Micro-batch size for critic updates. Controls memory usage during critic training.

```
critic.model.use_remove_padding=True
critic.model.path=Qwen/Qwen2.5-VL-3B-Instruct
critic.model.enable_gradient_checkpointing=True
```
`critic.model`:

* `use_remove_padding`: Whether to remove padding in critic inputs. Improves efficiency in critic computation.
* `path`: Base model for critic. Uses the same model architecture as actor.
* `enable_gradient_checkpointing`: Enables memory-saving for critic. Trades computation time for reduced memory usage.

```
critic.model.fsdp_config.param_offload=False
critic.model.fsdp_config.optimizer_offload=False
```
`critic.model.fsdp_config`:

* `param_offload`: Whether to offload critic parameters. False keeps parameters in GPU memory.
* `optimizer_offload`: Whether to offload critic optimizer states. False keeps optimizer states in GPU memory.

### Trainer
```
trainer.critic_warmup=0
trainer.logger=['console','wandb']
trainer.project_name='vagen_new'
trainer.experiment_name='trico_svg_vision_service'
trainer.n_gpus_per_node=4
trainer.nnodes=1
trainer.save_freq=70
trainer.test_freq=20
trainer.total_training_steps=200
trainer.val_before_train=True
trainer.val_generations_to_log_to_wandb=8
```
`trainer`:

- `critic_warmup`: Number of initial steps for critic-only training.
- `logger`: Logging destinations for experiment tracking.
- `project_name`: Project name for logging organization.
- `experiment_name`: Specific experiment identifier.
- `n_gpus_per_node`: Number of GPUs to use per node.
- `nnodes`: Number of compute nodes for distributed training.
- `save_freq`: Checkpoint saving frequency in steps.
- `test_freq`: Validation frequency in steps.
- `total_training_steps`: Total number of training iterations.
- `val_before_train`: Whether to run validation before training.
- `val_generations_to_log_to_wandb`: Number of generations to log in wandb.

### Rollout Manager
```
rollout_manager.max_turns=2
rollout_manager.window_size=3
rollout_manager.use_multi_turn_reward=False
rollout_manager.use_loss_mask=True
rollout_manager.use_gae_mask=True
rollout_manager.n_trajectory=8
```
`rollout_manager`:

- `max_turns`: Maximum number of interaction turns per episode.
- `window_size`: Context window size for previous interactions.
- `use_multi_turn_reward`: Whether to use turn-level rewards.
- `use_loss_mask`: Enables selective token masking for policy optimization.
- `use_gae_mask`: Applies masking to advantage calculations.
- `n_trajectory`: Number of parallel trajectories for batch processing.

## Service Architecture Configuration
### Environment Configuration File (env_config.yaml)

```yaml
env1:
    env_name: frozenlake   # Name of registered environment service
    env_config:    # Specific configs for your enviornment
        render_mode: text
    train_size: 10000   # Number of training environments 
    test_size: 512   # Number of testing environments

env2:
    env_name: svg   # Name of registered environment service
    env_config:     # Specific configs for your enviornment
        dino_weight: 3.0
        dino_only: False
    train_size: 10000   # Number of training environments 
    test_size: 512   # Number of testing environments
```
**env_name**: Specifies which registered environment service to use. Options include:

- `frozenlake`: Simple grid navigation environment
- `sokoban`: Visual puzzle environment with box pushing
- `svg`: SVG-based environment with reward model integration
- `navigation`: Visual navigation task for embodied AI

**env_config**: Environment-specific configuration (full configs should be defined in `env/your_env/configs`)

**train_size**: Number of training environments to generate

**test_size**: Number of testing environments to generate

### Service Configuration File (service_config.yaml)

```yaml
# UNDER DEVELOPMENT
```