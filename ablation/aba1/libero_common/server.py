"""Local student-only inference service, isolated from the training process."""
import argparse,base64,gc,json,os,threading,time,traceback
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import numpy as np
import torch
from .model import GoalLocalPolicy
from .config import ABA, atomic_json, file_sha256, digest_json
class Engine:
 def __init__(self,ready,variant,device='cuda:0'):
  self.variant=variant;self.device=torch.device(device);self.sha=None
  self.ready=Path(ready);self.lock=threading.Lock();self.model=None;self.checkpoint=None;self.step=None;self.calls=0
 def load(self,path):
  p=Path(path).resolve()
  if not p.is_relative_to((ABA/self.variant/'libero/runs').resolve()) or not (p/'checkpoint_complete.json').is_file():
   raise ValueError('Expected a complete checkpoint under this experiment run directory')
  if self.checkpoint==str(p):return self.info()
  if self.model is not None:raise ValueError('One immutable checkpoint per server; restart to change models')
  marker=json.loads((p/'checkpoint_complete.json').read_text())
  if marker['step']<1:raise ValueError('Cannot evaluate initialization as a trained policy')
  sha=file_sha256(p/'model.pt')
  if sha!=marker['model_sha256']:raise ValueError('Checkpoint checksum mismatch')
  self.model=GoalLocalPolicy.load(p).to(self.device).eval().requires_grad_(False)
  if self.model.cfg['ablation']!=self.variant:raise ValueError('Checkpoint ablation mismatch')
  self.sha=sha
  if self.model.mode!='student' or self.model.cfg['flow']['use_state']:raise ValueError('Expected student without state input')
  self.checkpoint=str(p);self.step=self.model.source_step;self.calls=0;gc.collect()
  info=self.info();atomic_json(self.ready,info);print(json.dumps(info),flush=True);return info
 def info(self):return dict(pid=os.getpid(),port=getattr(self,'port',None),checkpoint=self.checkpoint,step=self.step,calls=self.calls,physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),teacher_loaded=False,uses_future=False,uses_state=False,variant=self.variant,model_sha256=self.sha,config_sha256=digest_json(self.model.cfg) if self.model is not None else None,local_slots=self.model.kl if self.model is not None else None,goal_slots=self.model.kg if self.model is not None else None,interaction=self.model.cfg['interaction'] if self.model is not None else None)
 @torch.inference_mode()
 def infer(self,request):
  if self.model is None or str(Path(request['checkpoint']).resolve())!=self.checkpoint:raise ValueError('Inference checkpoint mismatch')
  if set(request)!={'checkpoint','main','wrist','instruction','noise_seed'}:raise ValueError('Unexpected policy inputs')
  def image(value):
   raw=base64.b64decode(value,validate=True)
   if len(raw)!=256*256*3:raise ValueError('Expected 256x256 uint8 RGB image')
   return torch.from_numpy(np.frombuffer(raw,dtype=np.uint8).reshape(256,256,3).copy()).permute(2,0,1)[None]
  main,wrist=image(request['main']),image(request['wrist'])
  generator=torch.Generator(device=self.device).manual_seed(int(request['noise_seed']))
  noise=torch.randn((1,50,32),generator=generator,device=self.device,dtype=torch.bfloat16)
  begin=time.monotonic();a=self.model.predict_action(main,wrist,[request['instruction']],noise=noise)[0].float().cpu().numpy()
  if a.shape!=(8,7) or not np.isfinite(a).all():raise ValueError('Invalid student action output')
  # Equivalent to baseline client clipping normalized first-six actions before denormalizing.
  low=np.asarray(self.model.statistics['franka']['action']['min'][:6]);high=np.asarray(self.model.statistics['franka']['action']['max'][:6])
  a[:,:6]=np.clip(a[:,:6],low,high)
  self.calls+=1
  return dict(actions=a.tolist(),step=self.step,shape=[8,7],normalized=False,inference_seconds=time.monotonic()-begin)
def main():
 p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=0);p.add_argument('--ready',required=True);p.add_argument('--variant',choices=['A0','A1','A2','A3'],required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(2)
 engine=Engine(a.ready,a.variant,a.device)
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
