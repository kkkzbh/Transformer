from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from transformer.modeling.attention import MultiHeadAttention


class SinusoidalPositionalEncoding(nn.Module):
    encoding: Tensor  # 预计算正弦位置编码。

    def __init__(self, d_model: int, max_positions: int) -> None:
        super().__init__()
        positions = torch.arange(max_positions, dtype=torch.float).unsqueeze(1)
        div_terms = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10_000.0) / d_model)
        )
        encoding = torch.zeros(max_positions, d_model)
        encoding[:, 0::2] = torch.sin(positions * div_terms)
        encoding[:, 1::2] = torch.cos(positions * div_terms)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        if seq_len > self.encoding.size(1):
            raise ValueError(f"sequence length {seq_len} exceeds max_positions.")
        return x + self.encoding[:, :seq_len, :]


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(  # 逐位置前馈网络。
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    """Pre-LN 编码器层：先归一化，再进入子层并做残差连接。"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)  # 源序列自注意力。
        self.ff = FeedForward(d_model, d_ff, dropout)                     # 前馈子层。
        self.norm1 = nn.LayerNorm(d_model)                                # 自注意力前归一化。
        self.norm2 = nn.LayerNorm(d_model)                                # 前馈层前归一化。
        self.dropout1 = nn.Dropout(dropout)                               # 自注意力残差 dropout。
        self.dropout2 = nn.Dropout(dropout)                               # 前馈残差 dropout。

    def forward(self, x: Tensor, *, src_padding_mask: Tensor | None = None) -> Tensor:
        norm_x = self.norm1(x)
        x = x + self.dropout1(
            self.self_attn(
                norm_x,
                norm_x,
                norm_x,
                key_padding_mask=src_padding_mask,
            )
        )
        x = x + self.dropout2(self.ff(self.norm2(x)))
        return x


class DecoderLayer(nn.Module):
    """Pre-LN 解码器层：自注意力、交叉注意力、前馈层各自做残差连接。"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)   # 目标序列自注意力。
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)  # 编码器交叉注意力。
        self.ff = FeedForward(d_model, d_ff, dropout)                      # 前馈子层。
        self.norm1 = nn.LayerNorm(d_model)                                 # 自注意力前归一化。
        self.norm2 = nn.LayerNorm(d_model)                                 # 交叉注意力前归一化。
        self.norm3 = nn.LayerNorm(d_model)                                 # 前馈层前归一化。
        self.dropout1 = nn.Dropout(dropout)                                # 自注意力残差 dropout。
        self.dropout2 = nn.Dropout(dropout)                                # 交叉残差 dropout。
        self.dropout3 = nn.Dropout(dropout)                                # 前馈残差 dropout。

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        *,
        tgt_causal_mask: Tensor,
        tgt_padding_mask: Tensor | None = None,
        src_padding_mask: Tensor | None = None,
    ) -> Tensor:
        norm_x = self.norm1(x)
        x = x + self.dropout1(
            self.self_attn(
                norm_x,
                norm_x,
                norm_x,
                attn_mask=tgt_causal_mask,
                key_padding_mask=tgt_padding_mask,
            )
        )
        x = x + self.dropout2(
            self.cross_attn(
                self.norm2(x),
                memory,
                memory,
                key_padding_mask=src_padding_mask,
            )
        )
        x = x + self.dropout3(self.ff(self.norm3(x)))
        return x
