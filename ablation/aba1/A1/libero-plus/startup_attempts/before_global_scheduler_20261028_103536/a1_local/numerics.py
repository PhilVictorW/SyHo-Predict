"""A1-local SDPA backend selection; attention masks/equations are unchanged."""
import torch

def configure_attention_backends():
    # The original second training microbatch reproduces NaNs in
    # ScaledDotProductEfficientAttentionBackward0 on this installed runtime.
    # Masked calls fall back to math SDPA (FP32 intermediates for BF16 inputs);
    # unmasked calls may retain Flash SDPA. Never hide nonfinite gradients.
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    return dict(torch_version=torch.__version__,
                flash=torch.backends.cuda.flash_sdp_enabled(),
                math=torch.backends.cuda.math_sdp_enabled(),
                memory_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
                cudnn=torch.backends.cuda.cudnn_sdp_enabled())
