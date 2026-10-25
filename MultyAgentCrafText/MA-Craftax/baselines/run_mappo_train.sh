#!/usr/bin/env bash
# Launch MAPPO (multi-agent proximal policy optimization) training on CrafText / Craftax-Coop.
#
# Usage (from MA-Craftax repo root):
#   bash baselines/run_mappo_train.sh
#   bash baselines/run_mappo_train.sh mappo_rnn.yaml
#
# Optional Hugging Face Hub auto-upload of checkpoints:
#   export HF_HUB_UPLOAD=1
#   export HF_HUB_REPO_ID=your-user/mappo-craftext-checkpoints
#   export HF_TOKEN=hf_...
#   bash baselines/run_mappo_train.sh
#
# Config keys (baselines/config/mappo_rnn.yaml) also accepted:
#   HF_HUB_UPLOAD, HF_HUB_REPO_ID, HF_HUB_REPO_TYPE, HF_HUB_PRIVATE,
#   HF_HUB_PATH_PREFIX, CHECKPOINT_INTERVAL

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG_FILE="${1:-mappo_rnn.yaml}"

echo "[run_mappo_train] cwd=$ROOT"
echo "[run_mappo_train] config=baselines/config/${CONFIG_FILE}"
if [[ "${HF_HUB_UPLOAD:-0}" =~ ^(1|true|yes|on)$ ]]; then
  echo "[run_mappo_train] HF Hub upload ON repo=${HF_HUB_REPO_ID:-<unset>}"
else
  echo "[run_mappo_train] HF Hub upload OFF (local checkpoints only)"
fi

exec python baselines/my_mappo.py --config_file "$CONFIG_FILE"
