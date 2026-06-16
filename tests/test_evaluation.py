from __future__ import annotations

import json
from pathlib import Path

import torch

from transformer.config import ExperimentConfig, ModelConfig, TaskConfig, TrainConfig
from transformer.evaluation.report import (
    GenerationRecord,
    compute_generation_diagnostics,
    evaluate_checkpoint,
    format_evaluation_report,
)
from transformer.modeling.transformer import Seq2SeqTransformer
from transformer.training.checkpointing import save_checkpoint


def test_evaluate_checkpoint_writes_report_and_figures(tmp_path: Path) -> None:
    config = ExperimentConfig(
        task=TaskConfig(
            dataset_size=20,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            min_len=3,
            max_len=4,
        ),
        model=ModelConfig(
            d_model=16,
            num_heads=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            d_ff=32,
            dropout=0.0,
            max_positions=16,
        ),
        train=TrainConfig(batch_size=2, run_dir=tmp_path / "runs"),
    )
    vocab_size = 14
    pad_id = 0
    model = Seq2SeqTransformer(vocab_size=vocab_size, pad_id=pad_id, config=config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint = tmp_path / "runs" / "reverse" / "demo" / "checkpoints" / "best.pt"
    metrics = checkpoint.parent.parent / "metrics.jsonl"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text(
        "\n".join(
            [
                json.dumps({"step": 1, "train_loss": 2.0}),
                json.dumps({"step": 1, "val_loss": 2.1}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        config=config,
        step=1,
        best_val_loss=2.1,
    )

    report_dir = evaluate_checkpoint(
        checkpoint,
        output_dir=tmp_path / "eval",
        max_samples=2,
        device_name="cpu",
    )

    report_path = report_dir / "evaluation.json"
    predictions_path = report_dir / "predictions.jsonl"
    assert report_path.exists()
    assert predictions_path.exists()
    assert (report_dir / "figures" / "loss_curves.png").exists()
    assert (report_dir / "figures" / "exact_match_by_length.png").exists()
    assert (report_dir / "figures" / "generation_length_by_source_length.png").exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["split"] == "test"
    assert report["dataset_size"] == 2
    assert "perplexity" in report["teacher_forced"]
    assert report["greedy"]["examples"] == 2
    assert "length_match_rate" in report["greedy"]
    assert "avg_length_error" in report["greedy"]
    assert "generation_diagnostics" in report

    rendered = format_evaluation_report(report_dir)
    assert "Teacher forcing" in rendered
    assert "Greedy generation" in rendered
    assert "Generation diagnostics" in rendered
    assert "generation_length_by_source_length" in rendered


def test_compute_generation_diagnostics_uses_raw_generated_ids() -> None:
    diagnostics, samples = compute_generation_diagnostics(
        [
            GenerationRecord(output_ids=[5, 6, 2], target_length=2),
            GenerationRecord(output_ids=[3, 7, 8, 9], target_length=3),
            GenerationRecord(output_ids=[5, 2, 3], target_length=2),
        ],
        eos_id=2,
        unk_id=3,
        max_len=4,
    )

    assert diagnostics.eos_rate == 2 / 3
    assert diagnostics.missing_eos_rate == 1 / 3
    assert diagnostics.eos_at_expected_position_rate == 1 / 3
    assert diagnostics.max_len_truncation_rate == 1 / 3
    assert diagnostics.samples_with_unk_rate == 2 / 3
    assert diagnostics.unk_token_rate == 2 / 10
    assert diagnostics.avg_generated_length == 10 / 3
    assert samples[0].eos_index == 2
    assert samples[0].eos_at_expected_position
    assert samples[1].truncated_by_max_len
    assert samples[1].unk_count == 1
    assert samples[2].eos_index == 1
    assert not samples[2].eos_at_expected_position
