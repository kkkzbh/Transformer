from __future__ import annotations

from dataclasses import dataclass, field

from torch import Tensor


@dataclass(slots=True)
class KeyValueCache:
    key: Tensor | None = None    # 已缓存 key，形状 [B, H, T, Dh]。
    value: Tensor | None = None  # 已缓存 value，形状 [B, H, T, Dh]。


@dataclass(slots=True)
class DecoderLayerCache:
    self_attn: KeyValueCache = field(default_factory=KeyValueCache)   # 自注意力 K/V 缓存。
    cross_attn: KeyValueCache = field(default_factory=KeyValueCache)  # 交叉注意力 K/V 缓存。


@dataclass(slots=True)
class PastKeyValues:
    layers: list[DecoderLayerCache]  # 每层解码器的 K/V 缓存。

    @classmethod
    def empty(cls, num_layers: int) -> PastKeyValues:
        return cls(layers=[DecoderLayerCache() for _ in range(num_layers)])


@dataclass(frozen=True, slots=True)
class EncoderOutput:
    memory: Tensor                    # 编码器输出，形状 [B, S, D]。
    padding_mask: Tensor | None       # 源序列 padding mask，形状 [B, S]。


@dataclass(frozen=True, slots=True)
class Seq2SeqLMOutput:
    logits: Tensor                              # 词表 logits，形状 [B, T, V]。
    encoder_outputs: EncoderOutput             # 编码器输出缓存。
    past_key_values: PastKeyValues | None = None  # 推理 K/V 缓存。
