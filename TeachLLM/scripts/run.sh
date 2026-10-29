#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/train:$ROOT_DIR/eval:$ROOT_DIR/data:${PYTHONPATH:-}"

if [[ -d "$VENV_DIR" ]]; then
  source "$VENV_DIR/bin/activate"
fi

MODE="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$MODE" in
  train)
    exec python "$ROOT_DIR/train/train_sft_qwen_3.py" "$@"
    ;;
  eval-accuracy)
    exec python "$ROOT_DIR/eval/eval_qwen_action_accuracy.py" "$@"
    ;;
  rollout)
    exec python "$ROOT_DIR/visualize_qwen_env_rollout.py" "$@"
    ;;
  help|-h|--help)
    cat <<'EOF'
Usage:
  ./scripts/run.sh train [train_sft_qwen_3.py args...]
  ./scripts/run.sh eval-accuracy [eval_qwen_action_accuracy.py args...]
  ./scripts/run.sh rollout [visualize_qwen_env_rollout.py args...]

Examples:
  ./scripts/run.sh train --help
  ./scripts/run.sh train --data_dir /path/to/dataset --output_dir ./runs/sft-qwen
  ./scripts/run.sh eval-accuracy --help
  ./scripts/run.sh rollout --help
EOF
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Use: ./scripts/run.sh --help"
    exit 1
    ;;
esac
