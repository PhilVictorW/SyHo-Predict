"""Use the unchanged Goal/Local trainer with the indexed RLDS sample backend."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from indexed_future import IndexedFutureDataset
from goal_local import train as trainer

if __name__ == '__main__':
    trainer.FutureDataset = IndexedFutureDataset
    trainer.main()
