"""Frozen interaction policy service. Actor reload is serialized between groups."""
from __future__ import annotations
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from .common import (DEFAULT_CONFIG,atomic_json,configure_gpu,load_config,load_checkpoint,
                     require_execute,sha256,local_output)
from .rpc import array_decode,array_encode

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--config",default=str(DEFAULT_CONFIG));p.add_argument("--student")
    p.add_argument("--gpu");p.add_argument("--port",type=int,required=True)
    p.add_argument("--ready",required=True);p.add_argument("--execute",action="store_true")
    a=p.parse_args();require_execute(a);cfg=load_config(a.config);configure_gpu(a.gpu)
    import torch
    import numpy as np
    from .base_loader import load_base,load_student
    from .gate_actor import GateActor
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    if a.student:
        model,payload=load_student(cfg,a.student)
        student_hash=sha256(a.student)
    else:
        model=load_base(cfg);student_hash="round10_baseline"
    model.eval().requires_grad_(False)
    model.teacher.eval().requires_grad_(False)
    model.teacher.flow.checkpoint_blocks=False;model.checkpoint_blocks=False
    def forbidden(*args,**kwargs):raise AssertionError("Real future is forbidden online")
    model.teacher.policy.encode_future_images=forbidden
    model.teacher.policy.future.forward=forbidden
    state=dict(actor=None,actor_hash=None)
    lock=threading.Lock()
    @torch.no_grad()
    def predict(req):
        if req["student_sha256"]!=student_hash:raise ValueError("Student version mismatch")
        if req.get("actor_sha256")!=state["actor_hash"]:raise ValueError("Actor version mismatch")
        views=[array_decode(req[k]) for k in ("main","left","right")]
        if any(v.shape!=(256,256,3) or v.dtype!=np.uint8 for v in views):
            raise ValueError("Expected three uint8 256x256 RGB images")
        tensors=[torch.from_numpy(x.copy()).permute(2,0,1) for x in views]
        obs=dict(current_main=tensors[0][None],current_wrist=torch.stack(tensors[1:])[None],
                 instruction=[str(req["instruction"])])
        seed=int(req["noise_seed"])
        with torch.random.fork_rng(devices=[0]):
            torch.cuda.manual_seed(seed)
            noise=torch.randn(1,50,32,device="cuda",dtype=torch.bfloat16).float()
        mode=req["policy"]
        kw={}
        if mode=="baseline":kw["baseline"]=True
        elif mode=="fixed":kw["eta"]=float(req["eta"])
        elif mode in ("actor_sample","actor_mean"):
            if state["actor"] is None:raise ValueError("No Actor loaded")
            kw.update(actor=state["actor"],gate_seed=int(req["gate_seed"]),mean=mode=="actor_mean")
        else:raise ValueError("Unknown policy mode")
        with torch.autocast("cuda",dtype=torch.bfloat16,cache_enabled=False):
            actions,record=model.predict(obs,noise,**kw)
        actions=actions.float().cpu().numpy()[0]
        if actions.shape!=(36,16) or not np.isfinite(actions).all():
            raise FloatingPointError("Invalid policy action")
        gate=None
        if mode.startswith("actor"):
            gate={k:(v[0].float().cpu().tolist() if k=="features" else float(v[0].float().cpu()))
                  for k,v in record.items()}
        return dict(actions=array_encode(actions),gate=gate,student_sha256=student_hash,
                    actor_sha256=state["actor_hash"])
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            try:
                length=int(self.headers.get("Content-Length","0"))
                if not 0<length<16*1024*1024:raise ValueError("Invalid request size")
                req=json.loads(self.rfile.read(length))
                with lock:
                    if self.path=="/health":
                        out=dict(ready=True,student_sha256=student_hash,actor_sha256=state["actor_hash"])
                    elif self.path=="/actor":
                        path=local_output(req["path"])
                        data=load_checkpoint(path)
                        if data.get("kind")!="actor" or data["student_sha256"]!=student_hash or data["control_site"]!="vlm_goal_local":
                            raise ValueError("Actor belongs to a different Student/control site")
                        actor=GateActor(cfg["rl"]["hidden_dim"],cfg["rl"]["concentration"])
                        actor.load_state_dict(data["actor"],strict=True)
                        state.update(actor=actor.cuda().eval().requires_grad_(False),actor_hash=sha256(path))
                        out=dict(actor_sha256=state["actor_hash"],student_sha256=student_hash)
                    elif self.path=="/predict":out=predict(req)
                    else:raise ValueError("Unknown route")
                encoded=json.dumps(out,allow_nan=False).encode()
                self.send_response(200)
            except Exception as exc:
                encoded=json.dumps(dict(error=repr(exc),error_kind="model_error")).encode()
                self.send_response(500)
            self.send_header("Content-Type","application/json");self.send_header("Content-Length",str(len(encoded)))
            self.end_headers();self.wfile.write(encoded)
    # Bind before reporting ready; a port conflict must not masquerade as a server.
    srv=HTTPServer(("127.0.0.1",a.port),Handler)
    atomic_json(local_output(a.ready),dict(ready=True,pid=__import__("os").getpid(),
                       port=a.port,student_sha256=student_hash,actor_sha256=None))
    try:srv.serve_forever()
    finally:srv.server_close()
if __name__=="__main__":main()
