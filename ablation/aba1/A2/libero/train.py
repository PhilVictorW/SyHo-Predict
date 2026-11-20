#!/usr/bin/env python3
"""A2 LIBERO train entrypoint. Importing this file starts no work."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ablation.aba1.libero_common.training import main

if __name__ == '__main__':
    main('A2', Path(__file__).with_name('config.json'))
