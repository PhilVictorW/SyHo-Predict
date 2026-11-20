#!/usr/bin/env python3
import sys,faulthandler
faulthandler.enable()

from pathlib import Path
WORK=Path(__file__).resolve().parent
sys.path.insert(0,str(WORK.parents[3]))
from a0_single_latent.training import main
if __name__=="__main__":
    if "--smoke" in sys.argv:faulthandler.dump_traceback_later(90,repeat=True)
    main()
