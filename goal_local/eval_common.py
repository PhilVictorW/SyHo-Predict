"""Shared, torch-free helpers for the isolated student LIBERO evaluation queue."""
import collections,io,json,pickle,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASELINE=ROOT.parent/'LaWAM_aba2'
TRAIN=ROOT/'runs/goal_local/stage_II_warm250_20260921'
SNAPSHOTS=ROOT/'runs/goal_local/student_eval_snapshots'
OUTPUT=ROOT/'results/goal_local/libero_40x10'
SSD=Path('/home/wangchaoyang/.cache/lawam_goal_local_eval_20260922')
SUITES=('libero_spatial','libero_object','libero_goal','libero_10')
PYTHON='/home/wangchaoyang/envs/LaWAM/bin/python'
SIM_PYTHON='/home/wangchaoyang/envs/LIBERO4LaWAM/bin/python'
def atomic_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False));tmp.replace(path)
class Placeholder:
 def __init__(self,*args,**kwargs):pass
 def __setstate__(self,state):pass
class Metadata(pickle.Unpickler):
 def find_class(self,module,name):
  return collections.OrderedDict if module=='collections' and name=='OrderedDict' else Placeholder
 def persistent_load(self,pid):return Placeholder()
def model_metadata(path):
 with zipfile.ZipFile(path) as z:
  name=next(n for n in z.namelist() if n.endswith('/data.pkl'))
  d=Metadata(io.BytesIO(z.read(name))).load()
 return {k:d[k] for k in ('format','mode','step','config','extra')}
def read_json(path,default=None):
 try:return json.loads(Path(path).read_text())
 except (OSError,ValueError):return default
