# temporal.py
from __future__ import annotations
from typing import Optional, Tuple
import math
import torch
from torch import nn

__all__ = [
    "Time2Vec",
    "TemporalPositionalEncoding",
    "TemporalAwareEncoder",
    "temporal_decay_weights",
]


class Time2Vec(nn.Module):
    """
    Kazemi et al. (2020) Time2Vec.
    Maps a scalar time t to [linear(t), sin(w1 t + b1), ..., sin(wk t + bk)].
    Expect t as a 1D tensor of shape (B,) in some numeric time space.
    """

    def __init__(self, k: int = 4):
        super().__init__()
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.randn(k))
        self.b = nn.Parameter(torch.zeros(k))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) or (B,1)
        if t.dim() == 2 and t.size(-1) == 1:
            t = t.squeeze(-1)
        elif t.dim() != 1:
            t = t.view(-1)
        lin = self.w0 * t + self.b0  # (B,)
        sin = torch.sin(self.w * t.unsqueeze(-1) + self.b)  # (B, k)
        return torch.cat([lin.unsqueeze(-1), sin], dim=-1)  # (B, 1+k)


class TemporalPositionalEncoding(nn.Module):
    """
    Continuous-time positional encoding (Fourier features) + linear projection.
    Produces a fixed-dim feature from a scalar timestamp.
    """

    def __init__(self, dim: int, max_freq: float = 10.0, n_freqs: int = 8):
        super().__init__()
        self.dim = dim
        freqs = torch.logspace(0, math.log10(max_freq), n_freqs)
        self.register_buffer("freqs", freqs, persistent=False)
        self.proj = nn.Linear(2 * n_freqs, dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) or (B,1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (B,1)
        x = torch.cat(
            [torch.sin(self.freqs * t), torch.cos(self.freqs * t)], dim=-1
        )  # (B, 2*n_freqs)
        return self.proj(x)  # (B, dim)


class TemporalAwareEncoder(nn.Module):
    """
    Fuse base embeddings with time features via a gated residual.
    Args:
        base_dim: dimension of the base embedding
        time_feat_dim: hidden size for time features before fusion
        mode: 't2v' or 'pe'
        k: number of periodic terms for Time2Vec (ignored if mode='pe')
        drop: dropout on time features
    Shapes:
        x: (B, D), t: (B,) or (B,1)  -> out: (B, D)
    """

    def __init__(
        self,
        base_dim: int,
        time_feat_dim: int = 32,
        mode: str = "t2v",
        k: int = 4,
        drop: float = 0.1,
    ):
        super().__init__()
        assert mode in {"t2v", "pe"}
        self.mode = mode
        if mode == "t2v":
            self.time_enc = Time2Vec(k=k)
            in_dim = 1 + k
        else:
            self.time_enc = TemporalPositionalEncoding(dim=time_feat_dim)
            in_dim = time_feat_dim

        self.proj = nn.Linear(in_dim, time_feat_dim)
        # Project time features to base dimension for gating
        self.time_proj = nn.Linear(time_feat_dim, base_dim, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(base_dim + time_feat_dim, base_dim),
            nn.GELU(),
            nn.Linear(base_dim, base_dim),
            nn.Sigmoid(),
        )
        self.fuse = nn.Linear(base_dim + time_feat_dim, base_dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Normalize t shape
        if t.dim() == 0:
            t = t.expand(x.size(0))
        if t.dim() == 2 and t.size(-1) == 1:
            t = t.squeeze(-1)
        elif t.dim() != 1:
            t = t.view(-1)
        # Encode time
        tf = self.time_enc(t)  # (B, F)
        tf = self.proj(tf)  # (B, time_feat_dim)
        tf = self.dropout(tf)
        g = self.gate(torch.cat([x, tf], dim=-1))  # (B, D)
        # Project time features to base dimension for gating
        tf_proj = self.time_proj(tf)  # (B, D)
        out = self.fuse(torch.cat([x, g * tf_proj], dim=-1))
        return out


def temporal_decay_weights(
    anchor_t: torch.Tensor,
    pos_t: torch.Tensor,
    neg_t: Optional[torch.Tensor] = None,
    tau: float = 24.0,
    unit: str = "hours",
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute time-aware weights:
      w_pos = exp(-|Δt_ap| / tau)   -> recent positives weigh more
      w_neg = 1 - exp(-|Δt_an| / tau) -> down-weight near-time negatives
    Assumes raw timestamps are in seconds; adjust scaling if needed.

    anchor_t: (B,)    epoch seconds
    pos_t:    (B,)    epoch seconds
    neg_t:    (B,K)   epoch seconds (optional)
    """
    if unit not in {"seconds", "minutes", "hours", "days"}:
        raise ValueError("unit must be one of: seconds|minutes|hours|days")

    to_hours = {
        "seconds": 1.0 / 3600.0,
        "minutes": 1.0 / 60.0,
        "hours": 1.0,
        "days": 24.0,
    }[unit]

    dt_pos = (anchor_t - pos_t).abs() * to_hours
    w_pos = torch.exp(-dt_pos / max(tau, 1e-6))

    w_neg = None
    if neg_t is not None:
        dt_neg = (anchor_t.unsqueeze(1) - neg_t).abs() * to_hours
        w_neg = 1.0 - torch.exp(-dt_neg / max(tau, 1e-6))

    return w_pos, w_neg
