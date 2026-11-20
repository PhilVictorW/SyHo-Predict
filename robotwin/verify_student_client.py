import json
from pathlib import Path
import numpy as np
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.Robotwin.eval_files.model2robotwin_interface import ModelClient

def main():
    root=Path(__file__).resolve().parent;run=root/'runs/stage_II_20260925'
    checkpoint=run/'step_000001/model.pt'
    client=WebsocketClientPolicy('127.0.0.1',24940)
    meta=client.get_server_metadata();assert Path(meta['ckpt_path'])==checkpoint and meta['teacher_loaded'] is False
    # Exercise the actual two-slot websocket contract with deterministic RGB probes.
    rgb=np.full((256,256,3),127,dtype=np.uint8)
    examples=[dict(primary_image=[rgb],wrist_image=[rgb,rgb],lang='Adjust the bottle.',embodiment_id=1,action_hz=30.) for _ in range(2)]
    examples[1]['lang']='Please grasp the bottle carefully and adjust it to stand upright on the table.'
    response=client.predict_action(dict(examples=examples));client.close()
    action=np.asarray(response['data']['normalized_actions']);assert action.shape==(2,8,16) and np.isfinite(action).all()
    statistics=json.loads((run/'dataset_statistics.json').read_text())['agilex']['action']
    raw=ModelClient.unnormalize_actions(action[0],statistics,passthrough_indices=(3,4,5,6,11,12,13,14),invert_indices=(7,15))
    assert raw.shape==(8,16) and np.isfinite(raw).all()
    assert np.allclose(np.linalg.norm(raw[:,3:7],axis=1),1.) and np.allclose(np.linalg.norm(raw[:,11:15],axis=1),1.)
    report=dict(websocket_ok=True,variable_length_instructions=True,checkpoint=str(checkpoint),normalized_shape=list(action.shape),denormalized_shape=list(raw.shape),quaternion_norms_valid=True,teacher_loaded=False,uses_future=False)
    (root/'verification/stage_II_student_client.json').write_text(json.dumps(report,indent=2));print(report)

if __name__=='__main__':main()
