"""Reuse the verified current-only student server, with an isolated checkpoint root."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from student_eval_common import TRAIN
from goal_local import eval_server

if __name__=='__main__':
    eval_server.SNAPSHOTS=TRAIN
    eval_server.main()
