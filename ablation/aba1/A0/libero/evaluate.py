#!/usr/bin/env python3
import sys
from pathlib import Path
WORK=Path(__file__).resolve().parent
sys.path.insert(0,str(WORK.parents[3]))
from a0_single_latent.evaluation import main
if __name__=="__main__":
    main("A0",WORK/"config.json")
