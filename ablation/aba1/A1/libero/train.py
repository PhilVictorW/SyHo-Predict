#!/usr/bin/env python3
"""A1 fixed-step pipeline. Default is a dry run; --execute starts both stages."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from a1_local.launcher import main
if __name__ == '__main__':
    main()
