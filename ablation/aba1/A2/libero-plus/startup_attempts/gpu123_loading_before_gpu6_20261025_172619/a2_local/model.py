"""A2 student initialization from its configured final teacher budget."""
import copy
from ablation.aba1.plus_common.config import validate_config
from ablation.aba1.plus_common.model import DistillationSystem,initialize_new_modules
from .config_runtime import stage_steps

def make_stage_two(posterior,cfg=None,initial_path=None):
    cfg=copy.deepcopy(cfg or posterior.cfg)
    validate_config(cfg)
    teacher_step=stage_steps(cfg,"I")
    if (cfg["ablation"]!="A2" or initial_path is not None or posterior.cfg!=cfg
        or posterior.mode!="posterior" or getattr(posterior,"source_step",None)!=teacher_step):
        raise ValueError("A2 student requires its own configured final teacher")
    student=type(posterior)(cfg,copy.deepcopy(posterior.qwen),copy.deepcopy(posterior.dino),
                           posterior.processor,posterior.statistics,"student")
    initialize_new_modules(student,queries_only=True)
    shared={k:v for k,v in posterior.state_dict().items() if not k.startswith("future.")}
    missing,unexpected=student.load_state_dict(shared,strict=False)
    expected={"q_goal"}
    if unexpected or set(missing)!=expected:raise ValueError((missing,unexpected))
    student.initialization_report=dict(teacher_step=teacher_step,paired=None,new_queries=sorted(expected),
        teacher_student_trainable_storage_shared=False)
    posterior.flow=None
    posterior.requires_grad_(False);posterior.frozen_teacher=True;posterior.eval()
    return DistillationSystem(student,posterior)
