#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec /home/wangchaoyang/envs/LaWAM/bin/python -B launch.py "$@"
