from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from a1_local.preflight import main
if __name__=="__main__":main()
