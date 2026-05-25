# 2×L40 resource work: code changes and training logic

This document summarizes changes made so SimpleVLA-RL can run the LIBERO OpenVLA-OFT LoRA launcher on **two NVIDIA L40 GPUs (~46 GB each)** without the common failure modes we hit (broken `flash_attn`, DP micro-batch shrinking to zero, TF vs PyTorch VRAM contention, and stale Ray processes).

## Files touched

### Launch scripts

- [`examples/run_openvla_oft_rl_libero_lora.sh`](../examples/run_openvla_oft_rl_libero_lora.sh)  
  - **`ray stop --force`** before launch to clear orphaned workers that reserve GPU memory.  
  - **`MUJOCO_GL=egl`** for headless LIBERO / MuJoCo EGL.  
  - **`VERL_FSDP_SUMMON_OFFLOAD_CPU=0`**: rollout must not use CPU-offloaded full params (`summon_full_params(offload_to_cpu=True)` mismatched embeddings vs CUDA inputs during `generate_action_verl`).  
  - **Micro-batches scaled to `NUM_GPUS`**: `ppo_micro_batch_size`, `rollout.log_prob_micro_batch_size`, `ref.log_prob_micro_batch_size` — FSDP divides these by world size; using `1` on 2 GPUs became **0** and broke updates.  
  - **Memory knobs**: `param_offload=True`, `grad_offload`, `optimizer_offload`, `traj_mini_batch_size=1`, rollout `max_prompt_length=256` (aligned with data).  
  - **Conservative LoRA on L40**: `lora_rank=16`, `lora_alpha=16` (raises success rate vs rank 32 on ~46 GB cards during long LIBERO rollouts + PPO update).

- [`examples/run_openvla_oft_rl_libero_lora_smoke.sh`](../examples/run_openvla_oft_rl_libero_lora_smoke.sh)  
  Fast path: short horizon (`max_episode_steps`), `test_freq=1`, `wandb offline`, etc.

- [`tests/test_libero_lora_smoke.sh`](../tests/test_libero_lora_smoke.sh)  
  Runs smoke script; **`ray stop`** before training; greps log for validation + actor metrics.

### Core training / rollout

- [`verl/trainer/ppo/ray_trainer.py`](../verl/trainer/ppo/ray_trainer.py)  
  - **`compute_entropy` is skipped when `entropy_coeff == 0`** (same PPO grads; one fewer full forward).

- [`verl/workers/fsdp_workers.py`](../verl/workers/fsdp_workers.py)  
  - **OpenVLA-OFT** (`openvla-oft`): tries `flash_attention_2` only if **`flash_attn` imports cleanly**, then loads with FA2 or falls back.  
  - **OpenVLA** (`openvla`): same try/fallback pattern (avoids hard dependency on a broken `flash_attn` wheel).  
  - Imports **`RobDataParallelPPOActor` from `dp_rob`** so the actor package does not eagerly import PRIME/`dp_prime`.  
  - **`update_actor`**: `gc.collect()` + `empty_cache()` before loading FSDP params from CPU when offload is enabled (reduces peak during param reload).

### Rollout and LIBERO utils

- [`verl/workers/rollout/rob_rollout.py`](../verl/workers/rollout/rob_rollout.py)  
  - **`max_episode_steps`** optional override (Hydra) with fallback to suite defaults; Robotwin path uses `.get(..., 800)`.  
  - **`_fsdp_summon_context()`**: gated by **`VERL_FSDP_SUMMON_OFFLOAD_CPU`** (default in code kept safe for rollout: no CPU offload unless env says so — scripts set `0` for correctness).  
  - **TensorFlow**: `tf.config.set_visible_devices([], "GPU")` in preprocessing so LIBERO image ops do not carve GPU VRAM beside PyTorch.

- [`verl/utils/libero_utils.py`](../verl/utils/libero_utils.py)  
  - Early **`tf.config.set_visible_devices([], "GPU")`** so TF JPEG/resize stays off the GPU across processes.

### Config default

- [`verl/trainer/config/ppo_trainer.yaml`](../verl/trainer/config/ppo_trainer.yaml)  
  - **`actor_rollout_ref.rollout.max_episode_steps: null`** placeholder for overrides (e.g. smoke tests).

### Actor imports (broken `flash_attn` environments)

- [`verl/workers/actor/dp_actor.py`](../verl/workers/actor/dp_actor.py), [`dp_rob.py`](../verl/workers/actor/dp_rob.py), [`dp_prime.py`](../verl/workers/actor/dp_prime.py): optional **`flash_attn.bert_padding` imports** (try/except) so worker import does not explode if the CUDA extension mismatches PyTorch.

- [`verl/workers/actor/__init__.py`](../verl/workers/actor/__init__.py)  
  - Lazy-safe imports: **`dp_rob`** always loads; **`dp_actor` / `dp_prime`** in try blocks so **`from verl.workers.actor import RobDataParallelPPOActor`** does not pull PRIME paths that force `flash_attn`.

## Did we change SimpleVLA-RL training *logic*?

**Mostly no.** Algorithmically it is still the same GRPO/PPO LIBERO pipeline: rollouts → rewards/advantages → `update_actor` in `RobDataParallelPPOActor`.

**Minor behavioral deltas (engineering, not a new RL algorithm):**

1. **`entropy_coeff == 0`**: we no longer run the extra **`compute_entropy`** forward each step — only metrics that depended on entropy logging differ; gradients and PPO objective are unchanged when entropy weight is zero.

2. **TensorFlow hidden from GPUs**: preprocessing runs on CPU; **policy and sim still run on CUDA** as before — only TF no longer reserves GPU memory.

3. **FSDP / Hydra correctness**: fixing micro-batch sizes after DP division fixes a **bug** that made effective micro-batch zero on multi-GPU; that restores intended training, not a new method.

4. **`max_episode_steps`**: optional cap only affects **episode length during rollout**, not the PPO formula (useful for smoke tests).

If you reinstall a **`flash_attn` build matching your PyTorch + CUDA**, FA2 loading will be attempted again for compatible models without code changes beyond the fallback already in place.

## Operational notes on this machine

- Uninstall incompatible **`flash_attn`** if `transformers`/workers fail at import (`undefined symbol` against `libtorch`). The code falls back to standard attention.  
- After any crash, run **`ray stop --force`** (the main script does this preemptively now) before relaunching.  
- **`train_batch_size=1`** combined with **`num_trials`/filtering can yield empty minibatches** in LIBERO rollout (observed empty `pad_sequence`); keep **`train_batch_size >= 2`** for stability unless you adapt the loop.
