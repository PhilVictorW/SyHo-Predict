"""Replay the recorded reference instructions while retaining Round10 noise mapping."""
import json
import os
from pathlib import Path
from robotwin.round10 import bridge


def instruction(*, task_name, instruction_type, episode_info, seed):
    table = json.loads(Path(os.environ['ROUND11_REFERENCE_INSTRUCTIONS']).read_text())
    return table[str(int(seed))]


bridge._select_instruction = instruction
if __name__ == '__main__': raise SystemExit(bridge.main())
