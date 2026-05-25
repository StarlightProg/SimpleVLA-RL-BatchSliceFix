#!/usr/bin/env bash
# Find largest train/val batch sizes that complete 1 train step + 1 validation on 2xL40.
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate simplevla-cpython

REPO_ROOT="/home/akornaev/workspace/vla/SimpleVLA-RL-BatchSliceFix"
cd "${REPO_ROOT}"

RESULTS="${RESULTS:-/tmp/l40_batch_sweep_results.txt}"
LOG_DIR="${LOG_DIR:-/tmp/l40_batch_sweep_logs}"
mkdir -p "${LOG_DIR}"
: > "${RESULTS}"

NUM_GPUS=2
# Shorter episodes for sweep speed; final 512-step check uses FULL_EPISODES=1.
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-128}"
FULL_EPISODE_VERIFY="${FULL_EPISODE_VERIFY:-1}"

run_probe() {
  local train_bs="$1"
  local val_bs="$2"
  local ppo_mini="$3"
  local n_samples="$4"
  local tag="$5"
  local max_steps="${6:-${MAX_EPISODE_STEPS}}"

  local log="${LOG_DIR}/${tag}.log"
  echo "=== PROBE ${tag}: train_bs=${train_bs} val_bs=${val_bs} ppo_mini=${ppo_mini} n_samples=${n_samples} max_steps=${max_steps} ===" | tee -a "${RESULTS}"

  ray stop --force 2>/dev/null || true
  sleep 2

  set +e
  TEST_ONE_STEP=1 \
  SKIP_CKPT_OVERWRITE=1 \
  NUM_TRIALS_PER_TASK=1 \
  TRAIN_BATCH_SIZE="${train_bs}" \
  VAL_BATCH_SIZE="${val_bs}" \
  PPO_MINI_BATCH_SIZE="${ppo_mini}" \
  N_SAMPLES="${n_samples}" \
  MAX_EPISODE_STEPS="${max_steps}" \
  bash examples/run_openvla_oft_rl_libero_lora.sh >"${log}" 2>&1
  local rc=$?
  set -e

  if [[ ${rc} -eq 0 ]] \
    && grep -q "validation generation end" "${log}" \
    && grep -qE "timing/update_actor|actor/pg_loss" "${log}" \
    && ! grep -qiE "OutOfMemoryError|CUDA out of memory" "${log}"; then
    echo "PASS ${tag}" | tee -a "${RESULTS}"
    return 0
  fi
  echo "FAIL ${tag} (exit=${rc})" | tee -a "${RESULTS}"
  grep -iE "OutOfMemory|CUDA out of memory|RayTaskError|Error executing" "${log}" | tail -5 >> "${RESULTS}" || true
  return 1
}

# ppo_mini_batch_size must be >= NUM_GPUS and divisible by NUM_GPUS after FSDP split uses config/ world
align_ppo_mini() {
  local train_bs="$1"
  local mini=$(( train_bs < 4 ? 4 : train_bs ))
  if (( mini % NUM_GPUS != 0 )); then
    mini=$(( (mini / NUM_GPUS + 1) * NUM_GPUS ))
  fi
  echo "${mini}"
}

# One-time ckpt utils patch before sweep
bash examples/overwrite_vla_ckpt_utils.sh /home/akornaev/workspace/vla/openvla_model

echo "L40 batch sweep started $(date -Is)" | tee -a "${RESULTS}"
echo "max_episode_steps=${MAX_EPISODE_STEPS} (set FULL_EPISODE_VERIFY=0 to skip 512-step confirm)" | tee -a "${RESULTS}"

BEST_TRAIN=2
BEST_VAL=2
BEST_MINI=4
BEST_NS=1

# Phase 1: maximize train_batch_size (n_samples=1)
for tb in 2 4 6 8 10 12 14 16; do
  pm=$(align_ppo_mini "${tb}")
  if run_probe "${tb}" "${tb}" "${pm}" 1 "tb${tb}_ns1"; then
    BEST_TRAIN=${tb}
    BEST_VAL=${tb}
    BEST_MINI=${pm}
  else
    echo "Stopping train_batch sweep at ${tb}" | tee -a "${RESULTS}"
    break
  fi
done

# Phase 2: maximize n_samples at best train batch
for ns in 2 3 4 5 6; do
  pm=$(align_ppo_mini "${BEST_TRAIN}")
  if run_probe "${BEST_TRAIN}" "${BEST_VAL}" "${pm}" "${ns}" "tb${BEST_TRAIN}_ns${ns}"; then
    BEST_NS=${ns}
  else
    echo "Stopping n_samples sweep at ${ns}" | tee -a "${RESULTS}"
    break
  fi
done

# Phase 3: try larger val_batch (validation VRAM)
BEST_VAL_FINAL=${BEST_VAL}
for vb in "${BEST_VAL}" $((BEST_VAL + 2)) $((BEST_VAL + 4)) $((BEST_VAL + 6)); do
  pm=$(align_ppo_mini "${BEST_TRAIN}")
  if run_probe "${BEST_TRAIN}" "${vb}" "${pm}" "${BEST_NS}" "tb${BEST_TRAIN}_ns${BEST_NS}_vb${vb}"; then
    BEST_VAL_FINAL=${vb}
  else
    break
  fi
done

# Phase 4: optional full 512-step episode confirm
if [[ "${FULL_EPISODE_VERIFY}" == "1" ]]; then
  pm=$(align_ppo_mini "${BEST_TRAIN}")
  if ! run_probe "${BEST_TRAIN}" "${BEST_VAL_FINAL}" "${pm}" "${BEST_NS}" \
    "tb${BEST_TRAIN}_ns${BEST_NS}_vb${BEST_VAL_FINAL}_ep512" "512"; then
    echo "WARN: config passed at max_episode_steps=${MAX_EPISODE_STEPS} but failed at 512 steps" | tee -a "${RESULTS}"
    # Step down one notch for safe production values
    if [[ ${BEST_NS} -gt 1 ]]; then BEST_NS=$((BEST_NS - 1)); fi
    if [[ ${BEST_TRAIN} -gt 2 ]]; then BEST_TRAIN=$((BEST_TRAIN - 2)); fi
    BEST_VAL_FINAL=${BEST_TRAIN}
    pm=$(align_ppo_mini "${BEST_TRAIN}")
  fi
fi

{
  echo ""
  echo "======== RECOMMENDED (2x L40) ========"
  echo "data.train_batch_size=${BEST_TRAIN}"
  echo "data.val_batch_size=${BEST_VAL_FINAL}"
  echo "data.n_samples=${BEST_NS}"
  echo "actor_rollout_ref.actor.ppo_mini_batch_size=${pm}"
  echo "Apply these in examples/run_openvla_oft_rl_libero_lora.sh"
  echo "Logs: ${LOG_DIR}"
} | tee -a "${RESULTS}"

echo "Sweep complete. Results: ${RESULTS}"
