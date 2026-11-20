"""Fixed-checkpoint LIBERO evaluation with mean Beta gating in FP32."""
from __future__ import annotations
import torch
from . import server as training_server
from .policy import Stage3Policy

class EvaluationPolicy(Stage3Policy):
    def _sample_eta(self, features):
        if self._stage3_eta_override is not None:
            return super()._sample_eta(features)
        with torch.autocast(features.device.type, enabled=False):
            dist = self.gate_actor.distribution(features.float())
            z = dist.mean
            return dict(z=z, eta=2.0*z, old_logp=None,
                        alpha=dist.concentration1, beta=dist.concentration0)

class EvaluationEngine(training_server.Engine):
    def info(self):
        result = super().info()
        result.update(gate_mode="beta_mean", actor_precision="float32",
                      residual_precision="unchanged Stage3Policy BF16 path")
        return result

    def infer(self, request):
        result = super().infer(request)
        result["gate"].pop("features", None)
        eta = torch.tensor(result["gate"]["eta"], dtype=torch.float32)
        result["gate"]["eta_applied_bf16"] = float(eta.to(torch.bfloat16).float())
        return result

def main():
    # This process-local substitution leaves training modules on disk unchanged.
    training_server.Stage3Policy = EvaluationPolicy
    training_server.Engine = EvaluationEngine
    training_server.main()

if __name__ == "__main__":
    main()
