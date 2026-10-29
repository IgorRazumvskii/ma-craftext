#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?Set HF_TOKEN with dataset write access}"
: "${HF_DATASET_REPO_ID:?Set HF_DATASET_REPO_ID to owner/dataset}"
checkpoint=/checkpoint/mappo_model_latest.pkl
if [[ ! -f "$checkpoint" ]]; then
  echo "Checkpoint file not found: $checkpoint" >&2
  exit 1
fi

run_dir="$(mktemp -d /staging/collect_XXXXXXXX)"
run_name="$(basename "$run_dir")"
echo "Temporary files: $run_dir"
echo "HF dataset: $HF_DATASET_REPO_ID"
echo "If collection fails, files in the staging volume are kept."

python baselines/collect_expert_trajectories_mappo.py \
  --checkpoint "$checkpoint" \
  --num-trajectories "${COLLECT_NUM_TRAJECTORIES:-4}" \
  --num-envs-collect "${COLLECT_NUM_ENVS:-2}" \
  --rollout-steps "${COLLECT_ROLLOUT_STEPS:-128}" \
  --max-steps "${COLLECT_MAX_STEPS:-2000}" \
  --seed "${COLLECT_SEED:-0}" \
  --compact-observations \
  --output "$run_dir/$run_name.npz" \
  --hf-upload --hf-stream-upload \
  --hf-repo-id "$HF_DATASET_REPO_ID" "$@"

rmdir "$run_dir"
echo "Collection finished; temporary run directory removed."
