#!/usr/bin/env bash
set -euo pipefail

checkpoint=/checkpoint/mappo_model_latest.pkl
if [[ ! -f "$checkpoint" ]]; then
  echo "Checkpoint file not found: $checkpoint" >&2
  exit 1
fi

run_dir="$(mktemp -d /datasets/collect_XXXXXXXX)"
run_name="$(basename "$run_dir")"
echo "Local dataset directory: $run_dir"
echo "Collection output is stored on the server."

python baselines/collect_expert_trajectories_mappo.py \
  --checkpoint "$checkpoint" \
  --num-trajectories "${COLLECT_NUM_TRAJECTORIES:-100000}" \
  --num-envs-collect "${COLLECT_NUM_ENVS:-2}" \
  --rollout-steps "${COLLECT_ROLLOUT_STEPS:-128}" \
  --shard-size-mib "${COLLECT_SHARD_SIZE_MIB:-32}" \
  --max-steps "${COLLECT_MAX_STEPS:-100}" \
  --seed "${COLLECT_SEED:-0}" \
  --compact-observations \
  --output "$run_dir/$run_name.parquet" "$@"

echo "Collection finished: $run_dir"
