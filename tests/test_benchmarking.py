from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import torch
from torch import Tensor

from transformer import cli
from transformer.benchmarking.inference import (
    BenchmarkResult,
    _decode_with_cache,
    _decode_without_cache,
    _run_repeated,
    format_benchmark_report,
)
from transformer.config import ModelConfig
from transformer.modeling.cache import EncoderOutput, PastKeyValues, Seq2SeqLMOutput
from transformer.modeling.transformer import Seq2SeqTransformer


class _OneStepEosModel:
    def __init__(self, *, eos_id: int, vocab_size: int) -> None:
        self.eos_id = eos_id          # 每次调用都生成 eos。
        self.vocab_size = vocab_size  # 模拟词表大小。
        self.past_seen: list[bool] = []  # 每轮首步是否收到历史 cache。

    def __call__(
        self,
        src: Tensor | None,
        tgt_in: Tensor,
        *,
        encoder_outputs: EncoderOutput | None = None,
        past_key_values: PastKeyValues | None = None,
        use_cache: bool = False,
        cache_position: int | None = None,
    ) -> Seq2SeqLMOutput:
        del src, cache_position
        self.past_seen.append(past_key_values is not None)
        logits = torch.full((tgt_in.size(0), tgt_in.size(1), self.vocab_size), -1e9)
        logits[:, -1, self.eos_id] = 0.0
        return Seq2SeqLMOutput(
            logits=logits,
            encoder_outputs=cast(EncoderOutput, encoder_outputs),
            past_key_values=past_key_values or PastKeyValues.empty(0) if use_cache else None,
        )


def test_cached_and_uncached_benchmark_decoders_match() -> None:
    model = _tiny_model()
    model.eval()
    src = torch.tensor([[3, 5, 7]], dtype=torch.long)
    encoder_outputs = model.encode(src, src_padding_mask=src.eq(model.pad_id))

    without_cache_ids = _decode_without_cache(
        model,
        encoder_outputs,
        bos_id=1,
        eos_id=2,
        max_len=6,
    )
    with_cache_ids = _decode_with_cache(
        model,
        encoder_outputs,
        bos_id=1,
        eos_id=2,
        max_len=6,
    )

    assert with_cache_ids == without_cache_ids


def test_repeated_cached_decode_does_not_reuse_past_key_values() -> None:
    eos_id = 2
    model = _OneStepEosModel(eos_id=eos_id, vocab_size=6)
    encoder_outputs = EncoderOutput(memory=torch.zeros(1, 3, 4), padding_mask=None)

    output_ids, times = _run_repeated(
        lambda: _decode_with_cache(
            cast(Seq2SeqTransformer, model),
            encoder_outputs,
            bos_id=1,
            eos_id=eos_id,
            max_len=4,
        ),
        warmup=1,
        repeat=2,
        device=torch.device("cpu"),
    )

    assert output_ids == [eos_id]
    assert len(times) == 2
    assert model.past_seen == [False, False, False]


def test_format_benchmark_report_includes_core_metrics() -> None:
    report = format_benchmark_report(
        BenchmarkResult(
            device="cpu",
            input="2 3 4 5",
            without_cache_output="5 4 3 2",
            with_cache_output="5 4 3 2",
            without_cache_output_ids=[5, 4, 3, 2],
            with_cache_output_ids=[5, 4, 3, 2],
            outputs_match=True,
            load_seconds=0.1,
            encode_seconds_mean=0.01,
            without_cache_decode_seconds_mean=0.04,
            with_cache_decode_seconds_mean=0.02,
            speedup=2.0,
            tokens_per_second=200.0,
            warmup=5,
            repeat=50,
            max_len=32,
        )
    )

    assert "KV cache benchmark" in report
    assert "outputs_match: True" in report
    assert "without_cache_decode_seconds_mean: 0.040000" in report
    assert "with_cache_decode_seconds_mean: 0.020000" in report
    assert "speedup: 2.000000x" in report


def test_benchmark_infer_cli_parses_arguments(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_benchmark_inference(
        checkpoint: Path,
        *,
        input_text: str,
        max_len: int,
        warmup: int,
        repeat: int,
        device_name: str,
    ) -> BenchmarkResult:
        captured.update(
            {
                "checkpoint": checkpoint,
                "input_text": input_text,
                "max_len": max_len,
                "warmup": warmup,
                "repeat": repeat,
                "device_name": device_name,
            }
        )
        return BenchmarkResult(
            device="cpu",
            input=input_text,
            without_cache_output="",
            with_cache_output="",
            without_cache_output_ids=[],
            with_cache_output_ids=[],
            outputs_match=True,
            load_seconds=0.0,
            encode_seconds_mean=0.0,
            without_cache_decode_seconds_mean=1.0,
            with_cache_decode_seconds_mean=1.0,
            speedup=1.0,
            tokens_per_second=0.0,
            warmup=warmup,
            repeat=repeat,
            max_len=max_len,
        )

    written: dict[str, Path] = {}
    monkeypatch.setattr(cli, "benchmark_inference", fake_benchmark_inference)
    monkeypatch.setattr(cli, "format_benchmark_report", lambda result: "REPORT")
    monkeypatch.setattr(cli, "write_benchmark_json", lambda result, path: written.update(path=path))

    json_output = tmp_path / "benchmark.json"
    cli.main(
        [
            "benchmark-infer",
            "--checkpoint",
            "model.pt",
            "--input",
            "2 3 4 5",
            "--max-len",
            "12",
            "--warmup",
            "2",
            "--repeat",
            "3",
            "--device",
            "cpu",
            "--json-output",
            str(json_output),
        ]
    )

    assert captured == {
        "checkpoint": Path("model.pt"),
        "input_text": "2 3 4 5",
        "max_len": 12,
        "warmup": 2,
        "repeat": 3,
        "device_name": "cpu",
    }
    assert written == {"path": json_output}
    assert capsys.readouterr().out == "REPORT\n"


def _tiny_model() -> Seq2SeqTransformer:
    torch.manual_seed(0)
    return Seq2SeqTransformer(
        vocab_size=12,
        pad_id=0,
        config=ModelConfig(
            d_model=16,
            num_heads=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            d_ff=32,
            dropout=0.0,
            max_positions=16,
        ),
    )
