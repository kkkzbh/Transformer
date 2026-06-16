from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from transformer.modeling.cache import EncoderOutput, PastKeyValues
from transformer.modeling.transformer import Seq2SeqTransformer
from transformer.training.checkpointing import load_model_for_inference
from transformer.utils.device import resolve_device


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    device: str                              # 实际运行设备。
    input: str                               # 原始输入 token 文本。
    without_cache_output: str                # 无 KV cache 生成结果。
    with_cache_output: str                   # KV cache 生成结果。
    without_cache_output_ids: list[int]      # 无 KV cache 原始输出 id。
    with_cache_output_ids: list[int]         # KV cache 原始输出 id。
    outputs_match: bool                      # 两条路径输出是否一致。
    load_seconds: float                      # checkpoint 加载耗时。
    encode_seconds_mean: float               # encoder 单次平均耗时。
    without_cache_decode_seconds_mean: float # 无 KV cache decoder 平均耗时。
    with_cache_decode_seconds_mean: float    # KV cache decoder 平均耗时。
    speedup: float                           # 无 cache / 有 cache 的 decode 加速比。
    tokens_per_second: float                 # KV cache 路径每秒生成 token 数。
    warmup: int                              # 预热次数。
    repeat: int                              # 正式计时次数。
    max_len: int                             # 最大生成长度。


def benchmark_inference(
    checkpoint: str | Path,
    *,
    input_text: str,
    max_len: int,
    warmup: int,
    repeat: int,
    device_name: str = "auto",
) -> BenchmarkResult:
    if max_len <= 0:
        raise ValueError("max_len must be positive.")
    if warmup < 0:
        raise ValueError("warmup must be non-negative.")
    if repeat <= 0:
        raise ValueError("repeat must be positive.")

    device = resolve_device(device_name)
    load_started = time.perf_counter()
    model, task, _ = load_model_for_inference(checkpoint, device=device)
    _synchronize(device)
    load_seconds = time.perf_counter() - load_started

    src_ids = task.encode_source(input_text)
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_padding_mask = src.eq(model.pad_id)

    with torch.inference_mode():
        _, encode_times = _run_repeated(
            lambda: model.encode(src, src_padding_mask=src_padding_mask),
            warmup=warmup,
            repeat=repeat,
            device=device,
        )
        encoder_outputs = model.encode(src, src_padding_mask=src_padding_mask)

        without_cache_ids, without_cache_times = _run_repeated(
            lambda: _decode_without_cache(
                model,
                encoder_outputs,
                bos_id=task.vocab.bos_id,
                eos_id=task.vocab.eos_id,
                max_len=max_len,
            ),
            warmup=warmup,
            repeat=repeat,
            device=device,
        )
        with_cache_ids, with_cache_times = _run_repeated(
            lambda: _decode_with_cache(
                model,
                encoder_outputs,
                bos_id=task.vocab.bos_id,
                eos_id=task.vocab.eos_id,
                max_len=max_len,
            ),
            warmup=warmup,
            repeat=repeat,
            device=device,
        )

    without_cache_mean = _mean(without_cache_times)
    with_cache_mean = _mean(with_cache_times)
    generated_tokens = len(with_cache_ids)
    return BenchmarkResult(
        device=str(device),
        input=input_text,
        without_cache_output=task.decode_target(without_cache_ids),
        with_cache_output=task.decode_target(with_cache_ids),
        without_cache_output_ids=without_cache_ids,
        with_cache_output_ids=with_cache_ids,
        outputs_match=without_cache_ids == with_cache_ids,
        load_seconds=load_seconds,
        encode_seconds_mean=_mean(encode_times),
        without_cache_decode_seconds_mean=without_cache_mean,
        with_cache_decode_seconds_mean=with_cache_mean,
        speedup=without_cache_mean / with_cache_mean if with_cache_mean > 0 else float("inf"),
        tokens_per_second=(
            generated_tokens / with_cache_mean if with_cache_mean > 0 else float("inf")
        ),
        warmup=warmup,
        repeat=repeat,
        max_len=max_len,
    )


def format_benchmark_report(result: BenchmarkResult) -> str:
    lines = [
        "KV cache benchmark",
        f"device: {result.device}",
        f"input: {result.input}",
        f"without_cache_output: {result.without_cache_output}",
        f"with_cache_output: {result.with_cache_output}",
        f"outputs_match: {result.outputs_match}",
        f"warmup: {result.warmup}",
        f"repeat: {result.repeat}",
        f"max_len: {result.max_len}",
        "",
        "Timing",
        f"load_seconds: {_format_float(result.load_seconds)}",
        f"encode_seconds_mean: {_format_float(result.encode_seconds_mean)}",
        "without_cache_decode_seconds_mean: "
        f"{_format_float(result.without_cache_decode_seconds_mean)}",
        f"with_cache_decode_seconds_mean: {_format_float(result.with_cache_decode_seconds_mean)}",
        f"speedup: {_format_float(result.speedup)}x",
        f"tokens_per_second: {_format_float(result.tokens_per_second)}",
    ]
    return "\n".join(lines)


def write_benchmark_json(result: BenchmarkResult, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(result), handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    return path


def _decode_without_cache(
    model: Seq2SeqTransformer,
    encoder_outputs: EncoderOutput,
    *,
    bos_id: int,
    eos_id: int,
    max_len: int,
) -> list[int]:
    generated = torch.tensor(
        [[bos_id]],
        dtype=torch.long,
        device=encoder_outputs.memory.device,
    )
    output_ids: list[int] = []
    for _ in range(max_len):
        logits = model(None, generated, encoder_outputs=encoder_outputs).logits
        next_id = logits[:, -1, :].argmax(dim=-1)
        output_ids.append(int(next_id.item()))
        if next_id.item() == eos_id:
            break
        generated = torch.cat([generated, next_id[:, None]], dim=1)
    return output_ids


def _decode_with_cache(
    model: Seq2SeqTransformer,
    encoder_outputs: EncoderOutput,
    *,
    bos_id: int,
    eos_id: int,
    max_len: int,
) -> list[int]:
    current_token = torch.tensor(
        [[bos_id]],
        dtype=torch.long,
        device=encoder_outputs.memory.device,
    )
    past_key_values: PastKeyValues | None = None
    output_ids: list[int] = []
    for position in range(max_len):
        output = model(
            None,
            current_token,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=position,
        )
        past_key_values = output.past_key_values
        next_id = output.logits[:, -1, :].argmax(dim=-1)
        output_ids.append(int(next_id.item()))
        if next_id.item() == eos_id:
            break
        current_token = next_id[:, None]
    return output_ids


def _run_repeated[T](
    fn: Callable[[], T],
    *,
    warmup: int,
    repeat: int,
    device: torch.device,
) -> tuple[T, list[float]]:
    last_result: T | None = None
    for _ in range(warmup):
        last_result = fn()
    times: list[float] = []
    for _ in range(repeat):
        _synchronize(device)
        started = time.perf_counter()
        last_result = fn()
        _synchronize(device)
        times.append(time.perf_counter() - started)
    if last_result is None:
        raise RuntimeError("benchmark produced no result.")
    return last_result, times


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty list.")
    return sum(values) / len(values)


def _format_float(value: float) -> str:
    return f"{value:.6f}"
