from __future__ import annotations

import math

from torch import Tensor, nn

from transformer.config import ModelConfig
from transformer.modeling.attention import make_causal_mask
from transformer.modeling.layers import DecoderLayer, EncoderLayer, SinusoidalPositionalEncoding


class Seq2SeqTransformer(nn.Module):
    def __init__(self, *, vocab_size: int, pad_id: int, config: ModelConfig) -> None:
        super().__init__()
        self.config = config  # 模型超参数。
        self.pad_id = pad_id  # 填充词元 id。

        self.src_embedding = nn.Embedding(  # 源 token 嵌入。
            vocab_size, config.d_model, padding_idx=pad_id
        )
        self.tgt_embedding = nn.Embedding(  # 目标 token 嵌入。
            vocab_size, config.d_model, padding_idx=pad_id
        )
        self.position = SinusoidalPositionalEncoding(  # 共享位置编码。
            config.d_model, config.max_positions
        )
        self.dropout = nn.Dropout(config.dropout)  # 嵌入层 dropout。

        self.encoder_layers = nn.ModuleList(  # 编码器层栈。
            [
                EncoderLayer(config.d_model, config.num_heads, config.d_ff, config.dropout)
                for _ in range(config.num_encoder_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(  # 解码器层栈。
            [
                DecoderLayer(config.d_model, config.num_heads, config.d_ff, config.dropout)
                for _ in range(config.num_decoder_layers)
            ]
        )
        self.encoder_norm = nn.LayerNorm(config.d_model)             # 编码器最终归一化。
        self.decoder_norm = nn.LayerNorm(config.d_model)             # 解码器最终归一化。
        self.output_projection = nn.Linear(config.d_model, vocab_size)  # 词元 logits。

        self._reset_parameters()

    def forward(self, src: Tensor, tgt_in: Tensor) -> Tensor:
        src_padding_mask = src.eq(self.pad_id)
        tgt_padding_mask = tgt_in.eq(self.pad_id)
        memory = self.encode(src, src_padding_mask=src_padding_mask)
        return self.decode(
            tgt_in,
            memory,
            tgt_padding_mask=tgt_padding_mask,
            src_padding_mask=src_padding_mask,
        )

    def encode(self, src: Tensor, *, src_padding_mask: Tensor | None = None) -> Tensor:
        x = self.dropout(self.position(self.src_embedding(src) * math.sqrt(self.config.d_model)))
        for layer in self.encoder_layers:
            x = layer(x, src_padding_mask=src_padding_mask)
        return self.encoder_norm(x)

    def decode(
        self,
        tgt_in: Tensor,
        memory: Tensor,
        *,
        tgt_padding_mask: Tensor | None = None,
        src_padding_mask: Tensor | None = None,
    ) -> Tensor:
        causal_mask = make_causal_mask(tgt_in.size(1), device=tgt_in.device)
        x = self.dropout(self.position(self.tgt_embedding(tgt_in) * math.sqrt(self.config.d_model)))
        for layer in self.decoder_layers:
            x = layer(
                x,
                memory,
                tgt_causal_mask=causal_mask,
                tgt_padding_mask=tgt_padding_mask,
                src_padding_mask=src_padding_mask,
            )
        x = self.decoder_norm(x)
        return self.output_projection(x)

    def _reset_parameters(self) -> None:
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
