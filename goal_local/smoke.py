"""Real SFT, real LIBERO images, full-width forward/backward and deployment verification."""
import argparse,json,gc,time
from pathlib import Path
import torch
from torch.utils.data import default_collate
from .model import GoalLocalPolicy,make_stage_two
from .data import FutureDataset
from .roles import remove_old_queries


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:2');p.add_argument('--output',default='artifacts/goal_local/smoke');p.add_argument('--backward',action='store_true');p.add_argument('--roundtrip',action='store_true');args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(2026)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path('configs/goal_local/default.json').read_text())
    report={}
    def log(name,value):
        report[name]=value;print(name,json.dumps(value),flush=True);(out/'report.json').write_text(json.dumps(report,indent=2))
    model=GoalLocalPolicy.from_baseline(cfg)
    log('initialization',model.initialization_report)
    model.to(args.device).eval()
    ds=FutureDataset(cfg['dataset_root'],model.statistics,'all')
    batch=default_collate([ds.sample(ds.episodes[0],0)])
    # Full FP32 distinguishes architecture equivalence from BF16 shape/kernel rounding.
    model.qwen.float()
    with torch.no_grad():
        inputs=model.encode_current_inputs(batch['current_main'],batch['current_wrist'],batch['instruction'])
        current=model.current_embeddings(inputs)
        embed=model.qwen.get_input_embeddings()(inputs['input_ids']).clone()
        embed[current[2]]=model.act_query.to(embed.dtype);embed[current[3]]=model.flow_action_query.to(embed.dtype)
        original=model.qwen(inputs_embeds=embed,attention_mask=inputs['attention_mask'],pixel_values=inputs['pixel_values'],image_grid_thw=inputs['image_grid_thw'],use_cache=False).last_hidden_state
        base,_=remove_old_queries(original,current[1],current[2],current[3])
        encoded=model.encode(batch,inputs)
        discrepancy=float((base-encoded['current']).abs().max())
        log('current_baseline_fp32_max_abs_diff',discrepancy)
        torch.testing.assert_close(base,encoded['current'],atol=.001,rtol=.001)
    del current,original,base,encoded,embed
    model.qwen.bfloat16()
    with torch.autocast('cuda',dtype=torch.bfloat16),torch.no_grad():
        encoded=model.encode(batch,inputs)
        dino=model.current_dino(batch['current_main']);conditions=model.action_conditions(batch,encoded)
        log('shapes',dict(input_ids=list(inputs['input_ids'].shape),current=list(encoded['current'].shape),goal=list(encoded['goal'].shape),local=list(encoded['local'].shape),dino=list(dino.shape),base=list(conditions[0].shape),goal_action=list(conditions[3].shape),actions=list(batch['actions'].shape)))
        loss=model.action_loss(batch,encoded);log('stage_I_flow',float(loss))
        assert torch.isfinite(loss)
    del encoded,conditions,dino,loss;gc.collect();torch.cuda.empty_cache()
    if args.backward:
        model.train()
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(batch)['loss']
        loss.backward()
        names=['future.goal.query','future.local.query','act_query','flow_action_query','flow.goal_proj.weight','flow.local_proj.weight','flow.plan']
        grads={name:float(dict(model.named_parameters())[name].grad.float().norm()) for name in names}
        assert all(v>0 for v in grads.values())
        assert all(p.grad is None for p in model.qwen.visual.parameters())
        assert all(p.grad is None for p in model.dino.parameters())
        log('stage_I_gradient_norms',grads)
        # One small SGD step is a connectivity smoke, not a trained posterior teacher.
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:p.add_(p.grad,alpha=-1e-6)
        model.zero_grad(set_to_none=True)
        del loss;gc.collect();torch.cuda.empty_cache()
    if args.roundtrip:
        model.save(out/'posterior',step=0,extra={'smoke_only':True})
        del model;gc.collect();torch.cuda.empty_cache()
        model=GoalLocalPolicy.load(out/'posterior').to(args.device)
        log('posterior_roundtrip',True)
    system=make_stage_two(model).to(args.device).eval()
    teacher=system.teacher;student=system.student
    current_only={k:batch[k] for k in ('current_main','current_wrist','instruction')}
    with torch.autocast('cuda',dtype=torch.bfloat16),torch.no_grad():
        before=teacher.encode(batch)['goal'].clone()
        full=student.encode(batch);minimal=student.encode(current_only)
        torch.testing.assert_close(full['local'],minimal['local'],atol=0,rtol=0)
        losses=system(batch);log('stage_II_losses',{k:float(v) for k,v in losses.items()})
        assert all(torch.isfinite(v) for v in losses.values())
        tail=default_collate([ds.sample(ds.episodes[0],ds.episodes[0]['length']-1)])
        missing=system(tail);log('all_future_missing_losses',{k:float(v) for k,v in missing.items()})
        assert float(missing['local_distill'])==0 and float(missing['goal_distill'])==0
    del full,minimal,losses,missing,tail;gc.collect();torch.cuda.empty_cache()
    if args.backward:
        system.train()
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=system(batch)['loss']
        loss.backward()
        names=['q_goal','q_local','gate.proj_l.weight','gate.proj_g.weight','gate.null_key','gate.scales']
        grads={name:float(dict(student.named_parameters())[name].grad.float().norm()) for name in names}
        assert all(v>0 for v in grads.values());assert all(p.grad is None for p in teacher.parameters())
        log('stage_II_gradient_norms',grads)
        torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        with torch.no_grad():
            for p in student.parameters():
                if p.grad is not None:p.add_(p.grad,alpha=-1e-6)
        student.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16),torch.no_grad():after=teacher.encode(batch)['goal']
        torch.testing.assert_close(before,after,atol=0,rtol=0);log('teacher_unchanged_after_student_update',True)
        del loss,after
    student.eval();noise=torch.randn(1,50,32,device=args.device)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        actions=student.predict_action(batch['current_main'],batch['current_wrist'],batch['instruction'],noise=noise)
    assert actions.shape==(1,8,7) and torch.isfinite(actions).all()
    log('deployment_actions',dict(shape=list(actions.shape),finite=True))
    if args.roundtrip:
        student.save(out/'student',step=0,extra={'smoke_only':True})
        del system,teacher,model,student;gc.collect();torch.cuda.empty_cache()
        loaded=GoalLocalPolicy.load(out/'student').to(args.device).eval()
        assert not hasattr(loaded,'future')
        with torch.autocast('cuda',dtype=torch.bfloat16):reloaded=loaded.predict_action(batch['current_main'],batch['current_wrist'],batch['instruction'],noise=noise)
        torch.testing.assert_close(actions,reloaded,atol=0,rtol=0)
        log('student_roundtrip_teacher_free',True)
    log('peak_allocated_GiB',torch.cuda.max_memory_allocated(args.device)/2**30)
    log('passed',True)

if __name__=='__main__':main()
