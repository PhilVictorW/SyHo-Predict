"""CPU-only checks for delayed launch, final budget and full task accounting."""
import copy,importlib.util,json,sys,unittest
from pathlib import Path
WORK=Path(__file__).resolve().parent
sys.path.insert(0,str(WORK.parents[3]))
from after_training_full_eval import gate
from a1_eval.runtime import validate_final_budget
from a1_eval.evaluation import summarize
from ablation.aba1.plus_common.config import load_config,checkpoint_path,ACTION_PROTOCOL
from ablation.aba1.plus_common.preflight import check_manifest

class AutoEval(unittest.TestCase):
    def setUp(self):
        self.cfg=load_config(WORK/"config.json","A1")
        self.teacher=dict(complete=True,stage="I",step=3000)
        self.student=dict(complete=True,stage="II",step=2500)
    def test_training_is_never_a_launch_signal(self):
        for stage in ("I","II"):
            self.assertEqual(gate(dict(phase="training",stage=stage),self.teacher,self.student,0,[],True),"waiting_training")
        self.assertEqual(gate(dict(phase="failed"),self.teacher,self.student,1,[],True),"waiting_training_recovery")
    def test_final_commit_and_successful_exit_are_both_required(self):
        ready=dict(phase="completed")
        self.assertEqual(gate(ready,self.teacher,self.student,0,[123],True),"waiting_training_exit")
        self.assertEqual(gate(ready,self.teacher,self.student,None,[],True),"waiting_successful_pipeline_exit")
        self.assertEqual(gate(ready,self.teacher,self.student,1,[],True),"waiting_successful_pipeline_exit")
        self.assertEqual(gate(ready,self.teacher,self.student,0,[],False),"waiting_final_checkpoints")
        self.assertEqual(gate(ready,self.teacher,dict(self.student,step=2000),0,[],True),"invalid_final_checkpoint")
        self.assertEqual(gate(ready,dict(self.teacher,step=5000),self.student,0,[],True),"invalid_final_checkpoint")
        self.assertEqual(gate(ready,self.teacher,self.student,0,[],True),"ready")
    def test_only_own_teacher3000_student2500_is_accepted(self):
        meta=dict(mode="student",step=2500,extra=dict(stage="II",teacher_step=3000,
            source_paths=dict(teacher=str(checkpoint_path(self.cfg,"I",3000)))))
        marker=dict(step=2500,cumulative_step=5500)
        validate_final_budget(meta,marker,self.cfg)
        with self.assertRaises(ValueError):validate_final_budget(meta,dict(marker,cumulative_step=7500),self.cfg)
        for change in ("step","teacher","source"):
            bad=copy.deepcopy(meta)
            if change=="step":bad["step"]=2000
            if change=="teacher":bad["extra"]["teacher_step"]=4000
            if change=="source":bad["extra"]["source_paths"]["teacher"]="/different/teacher"
            with self.assertRaises(ValueError):validate_final_budget(bad,marker,self.cfg)
    def test_full10030_and_incomplete_result_denominator(self):
        self.assertEqual(check_manifest(self.cfg)["tasks"],10030)
        tasks=json.loads(Path(self.cfg["full_eval_manifest"]).read_text())["tasks"]
        protocol=dict(tasks=tasks,checkpoint="student2500",model_sha256="test",protocol_sha256="test",scope="full")
        rows=[dict(suite=t["suite"],task_id=t["task_id"],category=t["category"],checkpoint="student2500",
            model_sha256="test",protocol_sha256="test",episode_idx=0,action_protocol_id=ACTION_PROTOCOL["id"],
            success=i%2==0,num_actions=8) for i,t in enumerate(tasks)]
        partial=summarize(protocol,rows[:-1]);self.assertIsNone(partial["success_rate"])
        complete=summarize(protocol,rows);self.assertTrue(complete["complete"])
        self.assertEqual(complete["expected"],10030);self.assertEqual(complete["success_rate"],.5)
        with self.assertRaises(ValueError):summarize(protocol,rows+[rows[0]])

if __name__=="__main__":unittest.main(verbosity=2)
