#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
exec ./train.sh --execute --after-eval
