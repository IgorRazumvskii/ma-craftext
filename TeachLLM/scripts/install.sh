#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
WITH_EVAL_DEPS=0
WITH_COMET=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-eval-deps)
      WITH_EVAL_DEPS=1
      shift
      ;;
    --with-comet)
      WITH_COMET=1
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--with-eval-deps] [--with-comet]"
      exit 1
      ;;
  esac
done

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$ROOT_DIR/requirements.txt"

if [[ "$WITH_EVAL_DEPS" -eq 1 ]]; then
  python -m pip install "jax[cpu]>=0.4.30"
fi

if [[ "$WITH_COMET" -eq 1 ]]; then
  python -m pip install "comet_ml>=3.33.0"
fi

echo "Environment is ready."
echo "Activate with: source \"$VENV_DIR/bin/activate\""
