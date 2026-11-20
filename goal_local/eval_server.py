"""Local student-only inference service, isolated from the training process."""
import argparse,base64,gc,json,os,threading,time,traceback
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import numpy as np
import torch
from .model import GoalLocalPolicy
from .eval_common import SNAPSHOTS,atomic_json
class Engine:
 def __init__(self,ready):
  self.ready=Path(ready);self.lock=threading.Lock();self.model=None;self.checkpoint=None;self.step=None;self.calls=0
 def load(self,path):
  p=Path(path).resolve()
  if not p.is_relative_to(SNAPSHOTS.resolve()) or not (p/'eval_snapshot_complete.json').exists():raise ValueError('Only archived student snapshots are allowed')
  if self.checkpoint==str(p):return self.info()
  if self.model is None:self.model=GoalLocalPolicy.load(p).to('cuda:0').eval().requires_grad_(False)
  else:
   payload=torch.load(p/'model.pt',map_location='cpu',mmap=True,weights_only=False)
   if payload['mode']!='student' or payload['config']!=self.model.cfg:raise ValueError('Student checkpoint architecture mismatch')
   self.model.load_state_dict(payload['state_dict'],strict=True);self.model.source_step=payload['step'];del payload
  if self.model.mode!='student' or self.model.cfg['flow']['use_state']:raise ValueError('Expected student without state input')
  self.checkpoint=str(p);self.step=self.model.source_step;self.calls=0;gc.collect()
  info=self.info();atomic_json(self.ready,info);print(json.dumps(info),flush=True);return info
 def info(self):return dict(pid=os.getpid(),port=getattr(self,'port',None),checkpoint=self.checkpoint,step=self.step,calls=self.calls,physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),teacher_loaded=False,uses_future=False,uses_state=False)
 @torch.inference_mode()
 def infer(self,request):
  if self.model is None or str(Path(request['checkpoint']).resolve())!=self.checkpoint:raise ValueError('Inference checkpoint mismatch')
  if set(request)!={'checkpoint','main','wrist','instruction','noise_seed'}:raise ValueError('Unexpected policy inputs')
  def image(value):
   raw=base64.b64decode(value,validate=True)
   if len(raw)!=256*256*3:raise ValueError('Expected 256x256 uint8 RGB image')
   return torch.from_numpy(np.frombuffer(raw,dtype=np.uint8).reshape(256,256,3).copy()).permute(2,0,1)[None]
  main,wrist=image(request['main']),image(request['wrist'])
  generator=torch.Generator(device='cuda:0').manual_seed(int(request['noise_seed']))
  noise=torch.randn((1,50,32),generator=generator,device='cuda:0',dtype=torch.bfloat16)
  begin=time.monotonic();a=self.model.predict_action(main,wrist,[request['instruction']],noise=noise)[0].float().cpu().numpy()
  if a.shape!=(8,7) or not np.isfinite(a).all():raise ValueError('Invalid student action output')
  # Equivalent to baseline client clipping normalized first-six actions before denormalizing.
  low=np.asarray(self.model.statistics['franka']['action']['min'][:6]);high=np.asarray(self.model.statistics['franka']['action']['max'][:6])
  a[:,:6]=np.clip(a[:,:6],low,high)
  self.calls+=1
  return dict(actions=a.tolist(),step=self.step,shape=[8,7],normalized=False,inference_seconds=time.monotonic()-begin)
def main():
 p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=0);p.add_argument('--ready',required=True);a=p.parse_args();torch.set_num_threads(2)
 engine=Engine(a.ready)
 class Handler(BaseHTTPRequestHandler):
  def log_message(self,*args):pass
  def send(self,code,result):
   data=json.dumps(result).encode();self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
  def do_GET(self):
   if self.path!='/health':return self.send(404,dict(error='unknown endpoint'))
   self.send(200,engine.info())
  def do_POST(self):
   try:
    length=int(self.headers.get('Content-Length',0))
    if not 0<length<2_000_000:raise ValueError('Invalid request length')
    request=json.loads(self.rfile.read(length))
    with engine.lock:
     if self.path=='/load':result=engine.load(request['checkpoint'])
     elif self.path=='/infer':result=engine.infer(request)
     else:raise ValueError('Unknown endpoint')
    self.send(200,result)
   except Exception as e:
    traceback.print_exc();self.send(500,dict(error=repr(e)))
 server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler);engine.port=server.server_address[1];atomic_json(a.ready,engine.info());server.serve_forever()
if __name__=='__main__':main()
