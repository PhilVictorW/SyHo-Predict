"""A1 LIBERO-Plus entrypoint. No work on import."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from a1_eval.evaluation import main
if __name__=="__main__":main("A1",Path(__file__).with_name("config.json"))
