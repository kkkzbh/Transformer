from __future__ import annotations

import torch
from torch import Tensor


def make_causal_mask(length: int, *, device: torch.device | None = None) -> Tensor:
    """Return a bool mask where True means "this target position is hidden"."""
    return torch.ones((length, length), dtype=torch.bool, device=device).triu(diagonal=1)
