"""Evaluation-only entry: deterministic Beta-mean GateActor, frozen Student."""
from pathlib import Path
import torch
from . import server
from .gate_actor import GateActor


class MeanGateActor(GateActor):
    @torch.no_grad()
    def collect(self, features, seed=None):
        dist = self.distribution(features)
        z = dist.mean
        return dict(z=z.detach(), eta=(2 * z).detach(), old_logp=None,
                    alpha=dist.concentration1.detach(), beta=dist.concentration0.detach())


class EvaluationEngine(server.Engine):
    def info(self):
        return super().info() | {'gate_mode': 'beta_mean', 'evaluation_only': True}

    def load(self, checkpoint, actor_checkpoint):
        payload = torch.load(Path(actor_checkpoint), map_location='cpu', weights_only=False)
        if payload.get('format') != 'stage3_gate_actor_v1' or payload.get('iteration') != 262:
            raise ValueError('evaluation requires the completed round-one update262 actor')
        del payload
        return super().load(checkpoint, actor_checkpoint)

    def infer(self, request):
        result = super().infer(request)
        # No RL feature cache is needed for inference; keep only gate statistics.
        result['gate'].pop('features', None)
        result['gate_mode'] = 'beta_mean'
        return result


def main():
    # Local bindings affect this new evaluation server process only.
    server.GateActor = MeanGateActor
    server.Engine = EvaluationEngine
    server.main()


if __name__ == '__main__':
    main()
