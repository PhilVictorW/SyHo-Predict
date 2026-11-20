"""Adapter for the unchanged LIBERO rollout core; accepts current images only."""
import base64,json,urllib.request
from pathlib import Path
import numpy as np
from PIL import Image

def rpc(url,endpoint,payload=None,timeout=1200):
 data=None if payload is None else json.dumps(payload).encode()
 req=urllib.request.Request(url+endpoint,data=data,headers={'Content-Type':'application/json'})
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 with opener.open(req,timeout=timeout) as response:return json.load(response)
class StudentClient:
 def __init__(self,checkpoint,url):
  self.checkpoint=str(Path(checkpoint).resolve());self.url=url;self.episode_seed=0;self.reset('')
 def configure_episode(self,suite_index,task_id,episode_idx):self.episode_seed=(suite_index+1)*10_000_000+task_id*10_000+episode_idx*100
 def reset(self,task_description,**kwargs):self.task_description=task_description;self.actions=None;self.cursor=0;self.chunk=0
 def close(self):pass
 def step(self,example,step=0,return_intermediates=False,**kwargs):
  if return_intermediates:raise ValueError('Similarity rendering is not supported by student adapter')
  if self.actions is None or self.cursor==len(self.actions):
   def encode(images):
    x=np.asarray(images[-1],dtype=np.uint8)
    if x.ndim!=3 or x.shape[-1]!=3:raise ValueError('Expected RGB HWC image')
    if x.shape[:2]!=(256,256):x=np.asarray(Image.fromarray(x).resize((256,256),Image.Resampling.BILINEAR))
    return base64.b64encode(np.ascontiguousarray(x).tobytes()).decode()
   # The baseline core collects state, but it is deliberately never sent to this policy.
   result=rpc(self.url,'/infer',dict(checkpoint=self.checkpoint,main=encode(example['primary_image']),wrist=encode(example['wrist_image']),instruction=str(example['lang']),noise_seed=self.episode_seed+self.chunk))
   a=np.asarray(result['actions'],dtype=np.float32)
   if result.get('normalized') is not False or a.shape!=(8,7) or not np.isfinite(a).all():raise ValueError('Invalid denormalized action chunk')
   self.actions=a;self.cursor=0;self.chunk+=1
  a=self.actions[self.cursor].copy();self.cursor+=1
  # No second denormalization. Baseline core retains its exact gripper conversion.
  return {'raw_action':{'world_vector':a[:3],'rotation_delta':a[3:6],'open_gripper':a[6:7]}}
