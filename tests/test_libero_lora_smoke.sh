#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate simplevla-cpython

REPO_ROOT="/home/akornaev/workspace/vla/SimpleVLA-RL-BatchSliceFix"
cd "${REPO_ROOT}"

ray stop --force 2>/dev/null || true
sleep 2

LOG_FILE="/tmp/simplevla_smoke.log"
rm -f "${LOG_FILE}"

bash examples/run_openvla_oft_rl_libero_lora_smoke.sh 2>&1 | tee "${LOG_FILE}"

grep -q "validation generation end" "${LOG_FILE}"
grep -qE "actor/pg_loss|timing/update_actor" "${LOG_FILE}"
echo "Smoke test passed."
