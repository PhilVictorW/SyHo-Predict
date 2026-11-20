"""Round11 student wrapper; reuse the verified Round10 deployment preprocessing."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from robotwin.round10.server import Policy, ROOT, R, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candidate', choices=['query_only', 'query_gate', 'action_adapter'], required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '2'
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.25)
    torch.use_deterministic_algorithms(True)
    cfg = json.loads((ROOT/'robotwin/configs/stage_I.json').read_text())
    policy = Policy('compatible', cfg['baseline_checkpoint'])
    adapter = a.candidate == 'action_adapter'
    path = (ROOT/'robotwin/round11/action_adapter_200/step_000200/adapter.pt' if adapter else
            ROOT/f'robotwin/round11/paired_query_pilot_500/{a.candidate}/step_000500/student.pt')
    assert (path.parent/'checkpoint_complete.json').exists()
    payload = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    if adapter:
        assert payload['format'] == 'robotwin_current_action_adapter_v1' and payload['step'] == 200
        assert payload['manifest']['base_config'] == cfg
        assert payload['manifest']['recipe'] == json.loads((ROOT/'robotwin/round7/candidate_recipe.json').read_text())
        named = dict(policy.teacher.named_parameters())
        weights = payload['adapter']
        assert set(weights) == set(payload['manifest']['trainable_names'])
        assert all(n.startswith(('policy.flow.goal_', 'policy.flow.local_')) for n in weights)
    else:
        assert payload['format'] == 'robotwin_paired_query_gate_v1'
        assert payload['step'] == 500 and payload['arm'] == a.candidate
        named = dict(policy.model.named_parameters())
        weights = payload['trainables']
        assert set(weights) == set(payload['manifest']['arms'][a.candidate]['trainable_names'])
    with torch.no_grad():
        for name,value in weights.items(): named[name].copy_(value)
    del named, payload, weights
    fixture = np.load(R/'fixture.npz')
    info = json.loads((R/'fixture.json').read_text())
    example = dict(primary_image=[fixture['main']], wrist_image=[fixture['left'], fixture['right']],
                   lang=info['instruction'], embodiment_id=1, action_hz=30., _noise_seed=290001)
    x = policy.predict_action([example])['normalized_actions']
    y = policy.predict_action([example])['normalized_actions']
    assert np.array_equal(x, y), float(np.max(np.abs(x-y)))
    from robotwin.round6.train import physical_metrics
    batch = torch.load(R/'fixture_batch.pt', map_location='cpu', weights_only=False)
    torch.manual_seed(800)
    measured = physical_metrics(policy.candidate(example, torch.randn(1,50,32,device='cuda')),
                                batch, policy.teacher.policy.statistics)
    if adapter:
        prior = json.loads((ROOT/'robotwin/round11/action_adapter_200/validation_000200.json').read_text())
        expected = prior['rows'][0]['metrics']
    else:
        prior = json.loads((ROOT/'robotwin/round11/paired_query_pilot_500/validation_000500.json').read_text())
        expected = prior['rows'][a.candidate][0]['metrics']
    error = max(abs(measured[k]-expected[k]) for k in measured)
    assert error < 1e-6, error
    check = dict(passed=True, repetition_max_abs=0., offline_reproduction_max_abs=error)
    write(a.output/'preflight.json', check)
    meta = dict(arm='compatible', candidate=a.candidate, checkpoint=str(path), horizon=36,
                action_hz=30, uses_future=False, uses_state=False, server_type='round11_student_pilot')
    write(a.output/'ready.json', dict(pid=os.getpid(), ready=True, time=time.time(), preflight=check, **meta))
    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
    WebsocketPolicyServer(policy, host='127.0.0.1', port=a.port, metadata=meta).serve_forever()


if __name__ == '__main__': main()
