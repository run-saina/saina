"""Inference-only routed residual experts for shared-pass Helm checkpoints."""
import torch
from torch import nn
import torch.nn.functional as F


class RoutedResidualHead(nn.Module):
    # Parameter names match the training module so saved expert weights load strictly.
    def __init__(self, shared, experts, bottleneck, progress):
        super().__init__()
        self.shared = shared
        h, c, device = shared.in_features, shared.out_features, shared.weight.device
        self.norm = nn.LayerNorm(h, device=device, dtype=torch.float32)
        self.router = nn.Linear(h, experts, bias=False, device=device, dtype=torch.float32)
        self.down = nn.Parameter(torch.empty(experts, bottleneck, h, device=device))
        self.up = nn.Parameter(torch.empty(experts, c, bottleneck, device=device))
        self.progress = progress

    @property
    def weight(self):
        return self.shared.weight

    def routes(self, h):
        # The saved training schedule, evaluated without routing noise. Every expert
        # residual is still computed; routing only weights them.
        with torch.autocast(h.device.type, enabled=False):
            fraction = min(max(self.progress / .1, 0.), 1.)
            p = (self.router(self.norm(h.float())) / (2 - fraction)).softmax(-1)
        alpha = min(max((self.progress - .1) / .1, 0.), 1.)
        if alpha == 0:
            return p
        v, i = p.topk(2, dim=-1)
        sparse = torch.zeros_like(p).scatter(-1, i, v / v.sum(-1, keepdim=True))
        return p * (1 - alpha) + sparse * alpha

    def forward(self, h):
        # No feature retention: a serving process must not keep request activations.
        p = self.routes(h)
        hidden = F.gelu(torch.einsum('nh,ebh->neb', h, self.down.to(h.dtype)))
        residual = torch.einsum('neb,ecb->nec', hidden, self.up.to(h.dtype))
        return self.shared(h).float() + (p[:, :, None] * residual.float()).sum(1)
