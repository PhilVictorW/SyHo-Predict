#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TOKENIZERS_PARALLELISM=false
exec /home/wangchaoyang/envs/LaWAM/bin/python "$SCRIPT_DIR/evaluate.py" "$@"
