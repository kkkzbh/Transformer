from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def make_causal_mask(length: int, *, device: torch.device | None = None) -> Tensor:
    """Return a bool mask where True means "this target position is hidden"."""
    return torch.ones((length, length), dtype=torch.bool, device=device).triu(diagonal=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.d_model = d_model                 # 模型表示维度。
        self.num_heads = num_heads             # 并行注意力头数。
        self.head_dim = d_model // num_heads   # 单头表示维度。

        self.q_proj = nn.Linear(d_model, d_model)    # 查询投影。
        self.k_proj = nn.Linear(d_model, d_model)    # 键投影。
        self.v_proj = nn.Linear(d_model, d_model)    # 值投影。
        self.out_proj = nn.Linear(d_model, d_model)  # 输出投影。
        self.dropout = nn.Dropout(dropout)           # 注意力权重 dropout。

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size, query_len, _ = query.shape
        key_len = key.size(1)

        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask[None, None, :, :], float("-inf"))
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        context = weights @ v
        context = context.transpose(1, 2).contiguous().view(batch_size, query_len, self.d_model)

        if context.size(1) != query_len or key_len != key.size(1):
            raise RuntimeError("Unexpected attention shape change.")
        return self.out_proj(context)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)
