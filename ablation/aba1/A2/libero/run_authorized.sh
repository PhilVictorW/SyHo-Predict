#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
bash launch.sh --execute 2>&1 | tee -a launcher.log
