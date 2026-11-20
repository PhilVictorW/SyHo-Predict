# Motus Policy for RoboTwin

import torch
import torch.nn as nn
import numpy as np
import cv2
import atexit
import json
import math
import time
from pathlib import Path
import sys
import os
import logging
from typing import List, Dict, Any, Optional
from collections import deque
import yaml
from PIL import Image
from transformers import AutoProcessor
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add model paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "models"))

from models.motus import Motus, MotusConfig

# Add bak path for T5EncoderModel
BAK_ROOT = str((Path(__file__).parent / "bak").resolve())
if BAK_ROOT not in sys.path:
    sys.path.insert(0, BAK_ROOT)

from wan.modules.t5 import T5EncoderModel
from utils.image_utils import resize_with_padding

logger = logging.getLogger(__name__)


class QuantizationProfiler:
    """Collect per-policy-call latency, action throughput, and estimated FLOPs."""

    FLOP_CATEGORIES = (
        "video_generation_dit",
        "intention_understanding_expert",
        "action_expert",
    )

    def __init__(self, model: nn.Module, output_root: str, task_name: str, device: str, metadata: Optional[Dict[str, Any]] = None):
        self.model = model
        self.output_dir = Path(output_root) / task_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.json"
        self.device = device
        self.metadata = metadata or {}
        self.created_at = time.strftime("%Y-%m-%d %H:%M:%S")

        self.active = False
        self.call_count = 0
        self.current_flops = self._zero_flops()
        self.flop_totals = self._zero_flops()
        self.flop_last = self._zero_flops()
        self.flop_max = self._zero_flops()
        self.stats = {
            "cuda_latency_ms": self._new_stat(),
            "wall_time_ms": self._new_stat(),
            "action_throughput_hz_effective": self._new_stat(),
            "action_throughput_hz_cuda_only": self._new_stat(),
            "actions_per_policy_call": self._new_stat(),
        }
        self.recent_calls = []
        self.hooks = []
        self._orig_einsum = None
        self._orig_wan_model_flash_attention = None
        self._orig_wan_attention_flash_attention = None

        self._register_hooks()
        self.flush()
        atexit.register(self.flush)

    def _zero_flops(self) -> Dict[str, int]:
        data = {name: 0 for name in self.FLOP_CATEGORIES}
        data["total"] = 0
        return data

    def _new_stat(self) -> Dict[str, Optional[float]]:
        return {"count": 0, "total": 0.0, "min": None, "max": None, "mean": None, "last": None}

    def _update_stat(self, name: str, value: float):
        if value is None or not math.isfinite(float(value)):
            return
        stat = self.stats[name]
        value = float(value)
        stat["count"] += 1
        stat["total"] += value
        stat["last"] = value
        stat["min"] = value if stat["min"] is None else min(float(stat["min"]), value)
        stat["max"] = value if stat["max"] is None else max(float(stat["max"]), value)
        stat["mean"] = stat["total"] / stat["count"]

    def _register_hooks(self):
        roots = [
            ("video_generation_dit", getattr(getattr(self.model, "video_model", None), "wan_model", None)),
            ("intention_understanding_expert", getattr(self.model, "vlm_model", None)),
            ("intention_understanding_expert", getattr(self.model, "und_expert", None)),
            ("action_expert", getattr(self.model, "action_expert", None)),
        ]
        seen = set()
        for category, root in roots:
            if root is None:
                continue
            for module in root.modules():
                key = (category, id(module))
                if key in seen:
                    continue
                seen.add(key)
                if isinstance(module, nn.Linear):
                    self.hooks.append(module.register_forward_hook(self._make_linear_hook(category)))
                elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    self.hooks.append(module.register_forward_hook(self._make_conv_hook(category)))
                elif "Attention" in module.__class__.__name__ and not module.__class__.__name__.startswith("Wan"):
                    self.hooks.append(module.register_forward_hook(self._make_attention_module_hook(category)))

    def _make_linear_hook(self, category: str):
        def hook(module, inputs, output):
            if not self.active or not inputs or not torch.is_tensor(inputs[0]):
                return
            shape = tuple(inputs[0].shape)
            if len(shape) == 0:
                return
            token_count = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
            self._add_flops(category, 2 * token_count * module.in_features * module.out_features)
        return hook

    def _make_conv_hook(self, category: str):
        def hook(module, inputs, output):
            if not self.active:
                return
            out = self._first_tensor(output)
            if out is None or out.dim() < 3:
                return
            spatial = int(np.prod(tuple(out.shape[2:])))
            kernel = int(np.prod(module.kernel_size))
            channels = module.in_channels // module.groups
            flops = 2 * int(out.shape[0]) * module.out_channels * spatial * channels * kernel
            self._add_flops(category, flops)
        return hook

    def _make_attention_module_hook(self, category: str):
        def hook(module, inputs, output):
            if not self.active or not inputs or not torch.is_tensor(inputs[0]):
                return
            hidden = inputs[0]
            if hidden.dim() < 3:
                return
            batch, seq_len = int(hidden.shape[0]), int(hidden.shape[1])
            num_heads = int(getattr(module, "num_heads", getattr(module, "num_attention_heads", 1)))
            head_dim = int(getattr(module, "head_dim", max(int(hidden.shape[-1]) // max(num_heads, 1), 1)))
            self._add_flops(category, 4 * batch * num_heads * seq_len * seq_len * head_dim)
        return hook

    def _first_tensor(self, value):
        if torch.is_tensor(value):
            return value
        if isinstance(value, (list, tuple)):
            for item in value:
                tensor = self._first_tensor(item)
                if tensor is not None:
                    return tensor
        if isinstance(value, dict):
            for item in value.values():
                tensor = self._first_tensor(item)
                if tensor is not None:
                    return tensor
        return None

    def _add_flops(self, category: str, flops: int):
        if not self.active or flops <= 0:
            return
        if category not in self.current_flops:
            return
        self.current_flops[category] += int(flops)

    def _patch_ops(self):
        if self._orig_einsum is None:
            self._orig_einsum = torch.einsum

            def counted_einsum(*args, **kwargs):
                if self.active and args and args[0] == "BTD,KNDE->KBTNE" and len(args) >= 3:
                    left, right = args[1], args[2]
                    if torch.is_tensor(left) and torch.is_tensor(right) and left.dim() == 3 and right.dim() == 4:
                        b, tokens, dim = [int(x) for x in left.shape]
                        k, heads, right_dim, head_dim = [int(x) for x in right.shape]
                        if dim == right_dim:
                            category = "action_expert"
                            und_dim = getattr(getattr(self.model, "config", None), "und_expert_hidden_size", None)
                            if und_dim is not None and dim == int(und_dim):
                                category = "intention_understanding_expert"
                            self._add_flops(category, 2 * k * b * tokens * heads * head_dim * dim)
                return self._orig_einsum(*args, **kwargs)

            torch.einsum = counted_einsum

        try:
            import wan.modules.model as wan_model_module
            if self._orig_wan_model_flash_attention is None:
                self._orig_wan_model_flash_attention = wan_model_module.flash_attention

                def counted_model_flash_attention(q, k, v, *args, **kwargs):
                    self._count_flash_attention(q, k, kwargs.get("q_lens"), kwargs.get("k_lens"))
                    return self._orig_wan_model_flash_attention(q, k, v, *args, **kwargs)

                wan_model_module.flash_attention = counted_model_flash_attention
        except Exception:
            pass

        try:
            import wan.modules.attention as wan_attention_module
            if self._orig_wan_attention_flash_attention is None:
                self._orig_wan_attention_flash_attention = wan_attention_module.flash_attention

                def counted_attention_flash_attention(q, k, v, *args, **kwargs):
                    self._count_flash_attention(q, k, kwargs.get("q_lens"), kwargs.get("k_lens"))
                    return self._orig_wan_attention_flash_attention(q, k, v, *args, **kwargs)

                wan_attention_module.flash_attention = counted_attention_flash_attention
        except Exception:
            pass

    def _restore_ops(self):
        if self._orig_einsum is not None:
            torch.einsum = self._orig_einsum
            self._orig_einsum = None
        try:
            import wan.modules.model as wan_model_module
            if self._orig_wan_model_flash_attention is not None:
                wan_model_module.flash_attention = self._orig_wan_model_flash_attention
                self._orig_wan_model_flash_attention = None
        except Exception:
            pass
        try:
            import wan.modules.attention as wan_attention_module
            if self._orig_wan_attention_flash_attention is not None:
                wan_attention_module.flash_attention = self._orig_wan_attention_flash_attention
                self._orig_wan_attention_flash_attention = None
        except Exception:
            pass

    def _count_flash_attention(self, q, k, q_lens=None, k_lens=None):
        if not self.active or not torch.is_tensor(q) or not torch.is_tensor(k) or q.dim() != 4 or k.dim() != 4:
            return
        heads = int(q.shape[2])
        head_dim = int(q.shape[3])
        if q_lens is not None and k_lens is not None and torch.is_tensor(q_lens) and torch.is_tensor(k_lens):
            q_vals = q_lens.detach().cpu().tolist()
            k_vals = k_lens.detach().cpu().tolist()
            token_pairs = sum(int(a) * int(b) for a, b in zip(q_vals, k_vals))
        else:
            token_pairs = int(q.shape[0]) * int(q.shape[1]) * int(k.shape[1])
        self._add_flops("video_generation_dit", 4 * token_pairs * heads * head_dim)

    def begin_call(self):
        self.current_flops = self._zero_flops()
        self.active = True
        self._patch_ops()

    def abort_call(self, error: Optional[str] = None):
        self.active = False
        self._restore_ops()
        if error:
            self.metadata["last_error"] = str(error)
            self.flush()

    def end_call(self, cuda_latency_ms: float, wall_time_ms: float, action_count: int):
        self.active = False
        self._restore_ops()
        self.call_count += 1

        self.current_flops["total"] = sum(self.current_flops[name] for name in self.FLOP_CATEGORIES)
        self.flop_last = dict(self.current_flops)
        for key, value in self.current_flops.items():
            self.flop_totals[key] += int(value)
            self.flop_max[key] = max(int(self.flop_max.get(key, 0)), int(value))

        action_count = int(action_count)
        wall_seconds = max(float(wall_time_ms) / 1000.0, 1e-12)
        cuda_seconds = max(float(cuda_latency_ms) / 1000.0, 1e-12)
        effective_hz = action_count / wall_seconds
        cuda_hz = action_count / cuda_seconds

        self._update_stat("cuda_latency_ms", cuda_latency_ms)
        self._update_stat("wall_time_ms", wall_time_ms)
        self._update_stat("action_throughput_hz_effective", effective_hz)
        self._update_stat("action_throughput_hz_cuda_only", cuda_hz)
        self._update_stat("actions_per_policy_call", action_count)

        self.recent_calls.append({
            "index": self.call_count,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cuda_latency_ms": float(cuda_latency_ms),
            "wall_time_ms": float(wall_time_ms),
            "action_count": action_count,
            "action_throughput_hz_effective": effective_hz,
            "action_throughput_hz_cuda_only": cuda_hz,
            "flops": dict(self.current_flops),
        })
        self.recent_calls = self.recent_calls[-20:]
        self.flush()

    def flush(self):
        try:
            mean_flops = {
                key: (self.flop_totals[key] / self.call_count if self.call_count else None)
                for key in self.flop_totals
            }
            payload = {
                "task_name": self.output_dir.name,
                "created_at": self.created_at,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "device": self.device,
                "inference_call_count": self.call_count,
                "metrics": self.stats,
                "flops": {
                    "unit": "FLOPs",
                    "totals": self.flop_totals,
                    "mean_per_policy_call": mean_flops,
                    "max_per_policy_call": self.flop_max,
                    "last_policy_call": self.flop_last,
                },
                "recent_policy_calls": self.recent_calls,
                "method": {
                    "cuda_latency_ms": "CUDA events around Motus.model.inference_step after model input tensors are prepared.",
                    "action_throughput_hz_effective": "action_count divided by wall-clock time for one policy inference call.",
                    "flops": "Estimated from Linear/Conv forward hooks plus WAN flash-attention and MoT einsum estimates.",
                },
                "metadata": self.metadata,
            }
            tmp_path = self.metrics_path.with_suffix(".json.tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, self.metrics_path)
        except Exception as exc:
            logger.warning(f"Failed to flush quantization metrics: {exc}")


class MotusPolicy:
    """
    Motus Policy wrapper for RoboTwin evaluation.
    Implements the joint video-action diffusion model for robotic control.
    """
    
    def __init__(self, checkpoint_path: str, config_path: str, wan_path: str, vlm_path: str, device: str = "cuda", log_dir: Optional[str] = None, task_name: Optional[str] = None, save_images: bool = False, quantization_dir: Optional[str] = None):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.wan_path = wan_path
        self.vlm_path = vlm_path
        self.task_name = task_name or os.environ.get('TASK_NAME') or os.environ.get("MOTUS_QUANTIZATION_TASK") or "default_task"
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config_dict = yaml.safe_load(f)
        
        # Initialize model WITHOUT loading pretrained backbones
        self.model = self._load_model()
        quantization_dir = quantization_dir or os.environ.get("MOTUS_QUANTIZATION_DIR")
        self.quant_profiler = None
        if quantization_dir:
            self.quant_profiler = QuantizationProfiler(
                self.model,
                quantization_dir,
                self.task_name,
                device,
                metadata={
                    "checkpoint_path": checkpoint_path,
                    "wan_path": wan_path,
                    "vlm_path": vlm_path,
                    "num_inference_timesteps": self.config_dict.get("model", {}).get("inference", {}).get("num_inference_timesteps"),
                    "action_chunk_size": getattr(self.model.config, "action_chunk_size", None),
                },
            )

        # Initialize T5 encoder for language embeddings (WAN text encoder)
        self.t5_encoder = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=device,
            checkpoint_path=os.path.join(self.wan_path, 'models_t5_umt5-xxl-enc-bf16.pth'),
            tokenizer_path=os.path.join(self.wan_path, 'google', 'umt5-xxl'),
        )

        # Initialize VLM processor from vlm_path (for tokenization only, weights from checkpoint)
        self.vlm_processor = AutoProcessor.from_pretrained(self.vlm_path, trust_remote_code=True)
        
        # Initialize observation cache
        self.obs_cache = deque(maxlen=1)
        self.action_cache = deque()
        
        # Model state
        self.current_state = None
        self.current_state_norm = None
        self.is_first_step = True
        self.prev_action = None

        # Load normalization stats
        self._load_normalization_stats()
        
        # Initialize image saving
        self.save_images = save_images
        base_log_dir = log_dir or os.environ.get('LOG_DIR') or str(Path(__file__).resolve().parent.parent / "logs")
        self.save_dir = Path(base_log_dir) / "images" / self.task_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.episode_count = 0
        self.step_count = 0

        logger.info("Motus Policy initialized successfully")

    def set_instruction(self, instruction: str):
        """Set the current instruction for the policy."""
        self.current_instruction = instruction
        logger.info(f"Instruction set: {instruction}")

    def _load_model(self) -> Motus:
        """Load the Motus model without pretrained backbones, then load checkpoint."""
        logger.info(f"Initializing Motus model from config (no pretrained backbones)")

        config = self._create_model_config()
        
        # Initialize model from config WITHOUT loading pretrained weights
        model = Motus(config)
        model = model.to(self.device)
        
        # Load checkpoint weights
        try:
            logger.info(f"Loading checkpoint from {self.checkpoint_path}")
            model.load_checkpoint(self.checkpoint_path, strict=False)
            logger.info("Model checkpoint loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
        
        model.eval()
        return model
    
    def _create_model_config(self) -> MotusConfig:
        """Create model configuration from yaml config - inference mode."""
        common = self.config_dict['common']
        model_cfg = self.config_dict['model']

        # Use paths passed to constructor
        vae_path = os.path.join(self.wan_path, "Wan2.2_VAE.pth")
        vlm_checkpoint_path = self.vlm_path

        hidden_size = model_cfg['action_expert']['hidden_size']
        ffn_multiplier = model_cfg['action_expert']['ffn_dim_multiplier']

        config = MotusConfig(
            # Paths for config loading only (no weights loaded)
            wan_checkpoint_path=self.wan_path,
            vae_path=vae_path,
            wan_config_path=self.wan_path,
            video_precision='bfloat16',
            vlm_checkpoint_path=vlm_checkpoint_path,
            
            # Understanding expert config
            und_expert_hidden_size=512,
            und_expert_ffn_dim_multiplier=4,
            und_expert_norm_eps=1e-5,
            und_layers_to_extract=None,
            vlm_adapter_input_dim=2048,
            vlm_adapter_projector_type="mlp3x_silu",
            
            # Model architecture
            num_layers=30,
            action_state_dim=common['state_dim'],
            action_dim=common['action_dim'],
            action_expert_dim=hidden_size,
            action_expert_ffn_dim_multiplier=ffn_multiplier,
            action_expert_norm_eps=1e-6,
            
            # Training config
            global_downsample_rate=common['global_downsample_rate'],
            video_action_freq_ratio=common['video_action_freq_ratio'],
            num_video_frames=common['num_video_frames'],
            video_loss_weight=1.0,
            action_loss_weight=1.0,
            
            # Inference config
            batch_size=1,
            video_height=common['video_height'],
            video_width=common['video_width'],
            
            # Don't load pretrained backbones - will load full model from checkpoint
            load_pretrained_backbones=False,
            training_mode='finetune',
        )

        return config
    
    def update_obs(self, observation: Dict[str, Any]):
        """Update observation cache with new observation."""
        # Extract visual observations
        if 'observation' in observation:
            obs_data = observation['observation']
            if 'head_camera' in obs_data and 'left_camera' in obs_data and 'right_camera' in obs_data:
                head_img = obs_data['head_camera']['rgb']
                left_img = obs_data['left_camera']['rgb']
                right_img = obs_data['right_camera']['rgb']
                
                left_img_resized = cv2.resize(left_img, (160, 120))
                right_img_resized = cv2.resize(right_img, (160, 120))
                bottom_row = np.concatenate([left_img_resized, right_img_resized], axis=1)
                image = np.concatenate([head_img, bottom_row], axis=0)
            else:
                raise ValueError("Missing camera data")
        elif 'head_camera' in observation:
            image = observation['head_camera']
        elif 'image' in observation:
            image = observation['image']
        else:
            raise ValueError("No visual observation found")

        target_size = (self.config_dict['common']['video_height'],
                      self.config_dict['common']['video_width'])

        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        else:
            image_tensor = image

        if image_tensor.shape[-2:] != target_size:
            image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            resized_np = resize_with_padding(image_np, target_size)
            if resized_np.dtype == np.uint8:
                resized_np = resized_np.astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(resized_np).permute(2, 0, 1).unsqueeze(0)
        
        self.obs_cache.append(image_tensor.to(self.device))

        # Extract robot state
        state = observation['joint_action']['vector']

        if isinstance(state, np.ndarray):
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        else:
            state_tensor = state.float().unsqueeze(0) if state.dim() == 1 else state.float()

        self.current_state = state_tensor.to(self.device)
        self.current_state_norm = self._normalize_actions(self.current_state).to(self.device)
    
    def get_action(self, instruction: str = None) -> List[np.ndarray]:
        """Get action predictions from the model."""
        if len(self.obs_cache) == 0:
            raise ValueError("No observations in cache. Call update_obs first.")
        
        if self.current_state is None:
            raise ValueError("No robot state available. Call update_obs first.")
        
        current_frame = self.obs_cache[-1]

        # Encode instruction with T5
        scene_prefix = ("The whole scene is in a realistic, industrial art style with three views: "
                        "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
                        "The aloha robot is currently performing the following task: ")
        instruction = f"{scene_prefix}{self.current_instruction}"
        t5_out = self.t5_encoder([instruction], self.device)
        if isinstance(t5_out, torch.Tensor):
            t5_list = [t5_out.squeeze(0)] if t5_out.dim() == 3 else [t5_out]
        elif isinstance(t5_out, list):
            t5_list = t5_out
        else:
            raise ValueError("Unexpected T5 encoder output format")

        # Build VLM inputs
        first_frame_pil = self._tensor_to_pil_image(current_frame.squeeze(0).cpu())
        vlm_inputs = self._preprocess_vlm_messages(instruction, first_frame_pil)

        # Run inference
        num_inference_steps = self.config_dict['model']['inference']['num_inference_timesteps']
        profiler = self.quant_profiler
        start_event = end_event = None
        wall_start = None
        if profiler is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            profiler.begin_call()
            wall_start = time.perf_counter()
            start_event.record()
        try:
            with torch.no_grad():
                predicted_frames, predicted_actions = self.model.inference_step(
                    first_frame=current_frame,
                    state=self.current_state,
                    num_inference_steps=num_inference_steps,
                    language_embeddings=t5_list,
                    vlm_inputs=[vlm_inputs],
                )
            if profiler is not None and start_event is not None and end_event is not None:
                end_event.record()
                torch.cuda.synchronize()
                cuda_latency_ms = start_event.elapsed_time(end_event)
                wall_time_ms = (time.perf_counter() - wall_start) * 1000.0
                profiler.end_call(cuda_latency_ms, wall_time_ms, int(predicted_actions.shape[1]))
        except Exception as exc:
            if profiler is not None:
                profiler.abort_call(str(exc))
            raise

        # Save frame grid
        if predicted_frames is not None:
            if predicted_frames.dim() == 5:
                if predicted_frames.shape[1] == 3:
                    predicted_frames_viz = predicted_frames.permute(0, 2, 1, 3, 4)
                else:
                    predicted_frames_viz = predicted_frames
                
                condition_frame_viz = current_frame.squeeze(0)
                predicted_frames_viz = predicted_frames_viz.squeeze(0)
                
                self._save_frame_grid(condition_frame_viz, predicted_frames_viz)
                self.step_count += 1

        actions_real = predicted_actions.squeeze(0).cpu().numpy()
        self.prev_action = actions_real[-1].copy()
        self.action_cache.extend(actions_real)

        return actions_real

    def _tensor_to_pil_image(self, tensor_chw: torch.Tensor) -> Image.Image:
        """Convert [C, H, W] tensor to PIL Image."""
        if tensor_chw.dtype != torch.float32:
            tensor_chw = tensor_chw.float()
        tensor_chw = tensor_chw.clamp(0, 1)
        np_img = (tensor_chw.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(np_img, mode='RGB')

    def _preprocess_vlm_messages(self, instruction: str, image: Image.Image) -> Dict[str, torch.Tensor]:
        """Build VLM inputs."""
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': instruction},
                    {'type': 'image', 'image': image},
                ]
            }
        ]
        text = self.vlm_processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        encoded = self.vlm_processor(text=[text], images=[image], return_tensors='pt')
        vlm_inputs = {
            'input_ids': encoded['input_ids'].to(self.device),
            'attention_mask': encoded['attention_mask'].to(self.device), 
            'pixel_values': encoded['pixel_values'].to(self.device),
            'image_grid_thw': encoded.get('image_grid_thw', None)
        }
        if vlm_inputs['image_grid_thw'] is not None:
            vlm_inputs['image_grid_thw'] = vlm_inputs['image_grid_thw'].to(self.device)
        return vlm_inputs

    def _load_normalization_stats(self):
        """Load action normalization stats."""
        try:
            stat_path = Path(__file__).parent / 'utils' / 'stat.json'
            with open(stat_path, 'r') as f:
                stat_data = yaml.safe_load(f) if stat_path.suffix in ['.yml', '.yaml'] else None
        except Exception:
            stat_data = None
        if stat_data is None:
            import json as _json
            with open(Path(__file__).parent / 'utils' / 'stat.json', 'r') as f:
                stat_data = _json.load(f)

        stats = stat_data.get('robotwin2')
        if stats is None:
            raise ValueError('Normalization stats not found')
        self.action_min = torch.tensor(stats['min'], dtype=torch.float32, device=self.device)
        self.action_max = torch.tensor(stats['max'], dtype=torch.float32, device=self.device)
        self.action_range = self.action_max - self.action_min

    def _normalize_actions(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize to [0,1]."""
        shape = x.shape
        x_flat = x.reshape(-1, shape[-1])
        norm = (x_flat - self.action_min.unsqueeze(0)) / self.action_range.unsqueeze(0)
        return norm.reshape(shape)

    def _denormalize_actions(self, y: torch.Tensor) -> torch.Tensor:
        """Denormalize from [0,1]."""
        shape = y.shape
        y_flat = y.reshape(-1, shape[-1])
        denorm = y_flat * self.action_range.unsqueeze(0) + self.action_min.unsqueeze(0)
        return denorm.reshape(shape)
    
    def _create_frame_grid(self, condition_frame: torch.Tensor, predicted_frames: torch.Tensor) -> Image.Image:
        """Create horizontal grid."""
        def tensor_to_numpy(tensor):
            if tensor.dim() == 3:
                tensor = tensor.permute(1, 2, 0)
            tensor = tensor.detach().cpu().float()
            tensor = torch.clamp(tensor, 0, 1)
            return (tensor.numpy() * 255).astype(np.uint8)
        
        condition_np = tensor_to_numpy(condition_frame)
        predicted_np = []
        num_pred_frames = predicted_frames.shape[0]
        for i in range(num_pred_frames):
            frame_np = tensor_to_numpy(predicted_frames[i])
            predicted_np.append(frame_np)
        
        while len(predicted_np) < 4:
            predicted_np.append(predicted_np[-1] if predicted_np else condition_np)
        
        all_frames = [condition_np] + predicted_np[:4]
        grid_image = np.concatenate(all_frames, axis=1)
        
        return Image.fromarray(grid_image)
    
    def _save_frame_grid(self, condition_frame: torch.Tensor, predicted_frames: torch.Tensor):
        """Save frame grid to disk."""
        if not self.save_images:
            return
        
        try:
            grid_image = self._create_frame_grid(condition_frame, predicted_frames)
            filename = f"episode_{self.episode_count:04d}_step_{self.step_count:04d}.png"
            save_path = self.save_dir / filename
            grid_image.save(save_path)
            logger.info(f"Saved frame grid to {save_path}")
        except Exception as e:
            logger.warning(f"Failed to save frame grid: {e}")


def encode_obs(observation):
    """Post-Process Observation"""
    return observation


def get_model(usr_args):
    """
    Initialize Motus model.
    
    Args:
        usr_args: Arguments from eval script (must include wan_path and vlm_path)
    """
    checkpoint_path = usr_args.get('ckpt_setting')
    wan_path = usr_args.get('wan_path')  # Passed from eval.sh or auto_eval.sh
    vlm_path = usr_args.get('vlm_path')  # Passed from eval.sh or auto_eval.sh
    
    if not wan_path:
        raise ValueError("wan_path not provided in usr_args")
    
    if not vlm_path:
        raise ValueError("vlm_path not provided in usr_args")
    
    policy_dir = Path(__file__).parent
    config_path = policy_dir / "utils" / "robotwin.yml"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    policy = MotusPolicy(
        checkpoint_path=checkpoint_path,
        wan_path=wan_path,
        vlm_path=vlm_path,
        config_path=str(config_path),
        device=device,
        log_dir=usr_args.get('log_dir'),
        task_name=usr_args.get('task_name'),
        save_images=bool(usr_args.get('save_images', False)),
        quantization_dir=usr_args.get('quantization_dir')
    )
    
    return policy


def eval(TASK_ENV, model, observation):
    """Evaluation function."""
    obs = encode_obs(observation)
    
    instruction = TASK_ENV.get_instruction()
    model.set_instruction(instruction)
    model.update_obs(obs)

    actions = model.get_action()
    
    for action in actions:
        TASK_ENV.take_action(action, action_type='qpos')


def reset_model(model):  
    """Reset model cache at episode start."""
    model.obs_cache.clear()
    model.action_cache.clear()
    model.current_state = None
    model.is_first_step = True
    model.prev_action = None
    model.episode_count += 1
    model.step_count = 0
    logger.info(f"Model reset completed for episode {model.episode_count}")