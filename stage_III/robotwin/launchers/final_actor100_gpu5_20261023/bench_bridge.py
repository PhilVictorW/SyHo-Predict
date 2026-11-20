"""Standard Round10 benchmark environment + immutable Stage III HTTP policy."""
import hashlib, json, os, random, sys, time
from pathlib import Path
import numpy as np
from eval_common import read, write
from stage_III.robotwin.rpc import call, array_encode, array_decode
import resumable_round10_bridge as bridge
from examples.Robotwin.eval_files import model2robotwin_interface as interface

class ActorTransport:
    def __init__(self, host, port):
        os.environ['NO_PROXY']=os.environ['no_proxy']='127.0.0.1,localhost,::1'
        self.url=f'http://127.0.0.1:{int(port)}'
        if host!='127.0.0.1': raise ValueError('Only loopback is allowed')
        self.expected=read(os.environ['STAGE3_POLICY_IDENTITY'])
        self.metadata=call(self.url,'/health',{},timeout=30)
        self.check(self.metadata)
    def check(self, info):
        for key in ('student_sha256','actor_sha256'):
            if info.get(key)!=self.expected[key]: raise ValueError('Wrong frozen policy: '+key)
    def get_server_metadata(self):
        return dict(arm='compatible',horizon=36,ckpt_path=self.expected['base_checkpoint'],
                    stage3_student_sha256=self.expected['student_sha256'],
                    stage3_actor_sha256=self.expected['actor_sha256'])
    def predict_action(self, request):
        examples=request['examples']
        if len(examples)!=1: raise ValueError('One environment per lane is required')
        ex=examples[0];images=ex['primary_image']+ex['wrist_image']
        req=dict(instruction=ex['lang'],policy='actor_mean',noise_seed=int(ex['_noise_seed']),
                 gate_seed=int(ex['_noise_seed']),student_sha256=self.expected['student_sha256'],
                 actor_sha256=self.expected['actor_sha256'],
                 **{k:array_encode(np.asarray(im,dtype=np.uint8)) for k,im in zip(('main','left','right'),images)})
        reply=call(self.url,'/predict',req,timeout=2400)
        self.check(reply)
        actions=array_decode(reply['actions'])
        if actions.shape!=(36,16) or not np.isfinite(actions).all():
            raise ValueError('Invalid Stage III action chunk')
        gate=reply.get('gate') or {}
        with (Path(os.environ['ROBOTWIN_EVAL_ROOT'])/'gate_trace.jsonl').open('a') as f:
            f.write(json.dumps(dict(time=time.time(),noise_seed=req['noise_seed'],
                         **{k:gate.get(k) for k in ('eta','old_alpha','old_beta')}))+'\n')
        return {'data':{'normalized_actions':actions[None]}}
    def close(self): pass

class ActivityConnection:
    def __init__(self, conn):
        self.conn=conn;self.seen={};self.last_write=0
        self.progress={'last_progress_at':time.time()}
    def send(self, msg):
        kind=msg.get('type')
        signature=(kind,msg.get('episode_id'),msg.get('seed'))
        score=int(msg.get('step',msg.get('steps',0)) or 0)
        meaningful=kind in ('candidate_valid','action_request','episode_done')
        now=time.time()
        if meaningful and score>self.seen.get(signature,-1):
            self.progress['last_progress_at']=now
            self.seen[signature]=score
        self.progress.update(phase=kind,updated_at=now,pid=os.getpid(),
                             episode=msg.get('episode_id'),seed=msg.get('seed'),
                             env_steps=msg.get('step',msg.get('steps',0)))
        if now-self.last_write>=2 or kind in ('candidate_valid','episode_done','slot_error'):
            write(Path(os.environ['ROBOTWIN_EVAL_ROOT'])/'activity.json',self.progress)
            self.last_write=now
        return self.conn.send(msg)
    def recv(self): return self.conn.recv()
    def close(self): return self.conn.close()

_original_slot=bridge._slot_worker_main
def slot_worker(slot_id, conn, **kw):
    # SAPIEN3's deprecated SapienRenderer ignores its device argument.
    # Bind the actual RenderSystem inside this owned simulator process only.
    import sapien
    original_init=sapien.Scene.__init__
    job=Path(os.environ['ROBOTWIN_EVAL_ROOT'])
    expected=read(job.parent.parent/'manifest.json')['gpu_pci']
    def scene_init(self, systems=None):
        if systems is None:
            renderer=sapien.render.RenderSystem('cuda:0')
            device=renderer.device
            if device.pci_string!=expected or device.cuda_id!=0:
                raise RuntimeError('Simulator is not on requested physical GPU5')
            write(job/'render_device.json',dict(pid=os.getpid(),physical_gpu=5,
                  pci=device.pci_string,cuda_id=device.cuda_id,time=time.time()))
            systems=[sapien.physx.PhysxCpuSystem(),renderer]
        return original_init(self,systems)
    sapien.Scene.__init__=scene_init
    # Observe real simulator messages; printing logs does not renew the deadline.
    return _original_slot(slot_id,ActivityConnection(conn),**kw)

def instruction(**kw):
    random.seed(int(kw['seed']))
    return bridge._select_instruction_native(**kw)

interface.WebsocketClientPolicy=ActorTransport
bridge._slot_worker_main=slot_worker
bridge._select_instruction=instruction

if __name__=='__main__':
    raise SystemExit(bridge.main())
