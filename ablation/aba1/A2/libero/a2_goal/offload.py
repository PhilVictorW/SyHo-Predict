"""Stream frozen Teacher modules without changing tensors or the forward graph."""
import torch


def stream_frozen_teacher(teacher, device):
    if not teacher.frozen_teacher or any(p.requires_grad for p in teacher.parameters()):
        raise ValueError("Only an independently frozen Teacher may be streamed")
    teacher.cpu()
    # encode() uses act_query.device for input placement. All tensors retain dtype.
    teacher.act_query.data = teacher.act_query.data.to(device)
    teacher.flow_action_query.data = teacher.flow_action_query.data.to(device)
    lm = teacher.qwen.language_model
    lm.rotary_emb.to(device)
    lm.norm.to(device)
    modules = [teacher.qwen.get_input_embeddings(), teacher.qwen.visual, teacher.future, *lm.layers]
    def load(module, args):
        if torch.is_grad_enabled():
            raise RuntimeError("Streaming Teacher forward must be under no_grad")
        module.to(device)
    def unload(module, args, output):
        module.cpu()
    handles = []
    for module in modules:
        handles.append(module.register_forward_pre_hook(load))
        handles.append(module.register_forward_hook(unload, always_call=True))
    teacher._stream_handles = handles
    teacher._stream_device = device
    return teacher
