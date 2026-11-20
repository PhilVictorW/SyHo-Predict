"""Expose the exception hidden by LIBERO-plus's reset retry loop (read-only)."""
import faulthandler
import json
import sys
from pathlib import Path

from rollout import _imports

faulthandler.enable()
faulthandler.dump_traceback_later(30, repeat=True)
libero, get_adapter, core = _imports()
task_row = json.loads(Path(sys.argv[1]).read_text())
suite = libero.benchmark.get_benchmark_dict()[task_row['suite']]()
task = suite.get_task(task_row['task_id'])
print('[诊断] 开始创建环境', flush=True)
env, _ = get_adapter('libero_plus').build_env(task, core.LIBERO_ENV_RESOLUTION, 234)
print('[诊断] 环境创建成功；直接调用底层 reset，显示被原包装器吞掉的异常', flush=True)
try:
    env.env.reset()
    print('[诊断] 底层 reset 成功', flush=True)
finally:
    env.close()
