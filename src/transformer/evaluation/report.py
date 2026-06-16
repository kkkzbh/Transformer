from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sized
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

import matplotlib
import torch
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from transformer.data.batch import Seq2SeqBatch, Seq2SeqSample
from transformer.data.tasks.base import Seq2SeqTask
from transformer.data.vocab import Vocab
from transformer.inference.decoding import greedy_decode
from transformer.modeling.transformer import Seq2SeqTransformer
from transformer.training.checkpointing import load_model_for_inference
from transformer.training.losses import seq2seq_cross_entropy
from transformer.utils.device import resolve_device
from transformer.utils.run_names import descending_timestamp_name

DatasetSplit = Literal["train", "val", "test"]


@dataclass(frozen=True, slots=True)
class TeacherForcedMetrics:
    loss: float
    perplexity: float
    token_accuracy: float
    examples: int
    tokens: int


@dataclass(frozen=True, slots=True)
class GreedyMetrics:
    exact_match: float
    token_accuracy: float
    examples: int
    tokens: int
    avg_decode_seconds: float
    length_match_rate: float
    avg_length_error: float
    by_length: dict[int, float]
    avg_target_length_by_source_length: dict[int, float]
    avg_prediction_length_by_source_length: dict[int, float]


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    output_ids: list[int]
    target_length: int


@dataclass(frozen=True, slots=True)
class GenerationSampleDiagnostics:
    generated_ids: list[int]
    eos_index: int | None
    has_eos: bool
    eos_at_expected_position: bool
    truncated_by_max_len: bool
    unk_count: int
    has_unk: bool


@dataclass(frozen=True, slots=True)
class GenerationDiagnostics:
    eos_rate: float
    missing_eos_rate: float
    eos_at_expected_position_rate: float
    avg_generated_length: float
    max_len_truncation_rate: float
    unk_token_rate: float
    samples_with_unk_rate: float


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    split: DatasetSplit = "test",
    output_dir: str | Path | None = None,
    batch_size: int | None = None,
    max_samples: int = 512,
    max_len: int | None = None,
    device_name: str = "auto",
) -> Path:
    checkpoint = Path(checkpoint_path)
    device = resolve_device(device_name)
    model, task, config = load_model_for_inference(checkpoint, device=device)
    dataset = _dataset_for_split(task, split)
    collate_fn = task.make_collate_fn()
    report_dir = Path(output_dir) if output_dir is not None else _default_output_dir(checkpoint)
    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    effective_batch_size = batch_size or config.train.batch_size
    effective_max_len = max_len or config.model.max_positions
    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    teacher_metrics = _evaluate_teacher_forced(
        model,
        loader,
        device=device,
        pad_id=task.vocab.pad_id,
    )
    greedy_metrics, generation_diagnostics, prediction_rows = _evaluate_greedy(
        model,
        dataset,
        task=task,
        device=device,
        max_samples=max_samples,
        max_len=effective_max_len,
    )

    predictions_path = report_dir / "predictions.jsonl"
    _write_jsonl(predictions_path, prediction_rows)

    loss_figure = _plot_training_losses(_metrics_path_for_checkpoint(checkpoint), figures_dir)
    length_figure = _plot_exact_match_by_length(greedy_metrics.by_length, figures_dir)
    generation_length_figure = _plot_generation_lengths(
        greedy_metrics.avg_target_length_by_source_length,
        greedy_metrics.avg_prediction_length_by_source_length,
        figures_dir,
    )

    report_path = report_dir / "evaluation.json"
    report = {
        "checkpoint": str(checkpoint),
        "split": split,
        "dataset_size": _dataset_size(dataset),
        "runtime": {
            "device": str(device),
            "batch_size": effective_batch_size,
            "max_len": effective_max_len,
            "max_samples": max_samples,
        },
        "teacher_forced": asdict(teacher_metrics),
        "greedy": asdict(greedy_metrics),
        "generation_diagnostics": asdict(generation_diagnostics),
        "artifacts": {
            "evaluation_json": str(report_path),
            "predictions": str(predictions_path),
            "figures": {
                "loss_curves": str(loss_figure) if loss_figure is not None else None,
                "exact_match_by_length": str(length_figure),
                "generation_length_by_source_length": str(generation_length_figure),
            },
        },
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=True, indent=2)
        handle.write("\n")

    return report_dir


@torch.no_grad()
def _evaluate_teacher_forced(
    model: Seq2SeqTransformer,
    loader: Iterable[Seq2SeqBatch],
    *,
    device: torch.device,
    pad_id: int,
) -> TeacherForcedMetrics:
    model.eval()
    total_loss = 0.0
    batches = 0
    examples = 0
    correct_tokens = 0
    total_tokens = 0

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.src, batch.tgt_in)
        loss = seq2seq_cross_entropy(logits, batch.tgt_out, pad_id=pad_id)
        predictions = logits.argmax(dim=-1)
        token_mask = batch.tgt_out.ne(pad_id)

        total_loss += loss.item()
        batches += 1
        examples += batch.src.size(0)
        correct_tokens += int(predictions.eq(batch.tgt_out).logical_and(token_mask).sum().item())
        total_tokens += int(token_mask.sum().item())

    if batches == 0 or total_tokens == 0:
        raise ValueError("evaluation dataset produced no tokens.")

    mean_loss = total_loss / batches
    return TeacherForcedMetrics(
        loss=mean_loss,
        perplexity=math.exp(mean_loss),
        token_accuracy=correct_tokens / total_tokens,
        examples=examples,
        tokens=total_tokens,
    )


@torch.no_grad()
def _evaluate_greedy(
    model: Seq2SeqTransformer,
    dataset: Dataset[Seq2SeqSample],
    *,
    task: Seq2SeqTask,
    device: torch.device,
    max_samples: int,
    max_len: int,
) -> tuple[GreedyMetrics, GenerationDiagnostics, list[dict[str, object]]]:
    if max_samples <= 0:
        raise ValueError("max_samples must be positive.")

    sample_count = min(max_samples, _dataset_size(dataset))
    exact_matches = 0
    correct_tokens = 0
    total_tokens = 0
    total_decode_seconds = 0.0
    length_matches = 0
    total_abs_length_error = 0
    length_totals: dict[int, int] = defaultdict(int)
    length_exact: dict[int, int] = defaultdict(int)
    target_length_sums: dict[int, int] = defaultdict(int)
    prediction_length_sums: dict[int, int] = defaultdict(int)
    generation_records: list[GenerationRecord] = []
    rows: list[dict[str, object]] = []

    if sample_count == 0:
        raise ValueError("greedy evaluation dataset produced no samples.")

    for index in range(sample_count):
        sample = dataset[index]
        src = torch.tensor([sample.src], dtype=torch.long, device=device)
        started_at = perf_counter()
        output_ids = greedy_decode(
            model,
            src,
            bos_id=task.vocab.bos_id,
            eos_id=task.vocab.eos_id,
            max_len=max_len,
        )
        total_decode_seconds += perf_counter() - started_at

        source_tokens = _decode_tokens(task.vocab, sample.src)
        target_tokens = _decode_tokens(task.vocab, sample.tgt_out)
        predicted_tokens = _decode_tokens(task.vocab, output_ids)
        exact = predicted_tokens == target_tokens
        source_length = len(source_tokens)
        target_length = len(target_tokens)
        prediction_length = len(predicted_tokens)
        length_error = prediction_length - target_length
        generation_records.append(
            GenerationRecord(output_ids=output_ids, target_length=target_length)
        )

        exact_matches += int(exact)
        length_matches += int(prediction_length == target_length)
        total_abs_length_error += abs(length_error)
        length_totals[source_length] += 1
        length_exact[source_length] += int(exact)
        target_length_sums[source_length] += target_length
        prediction_length_sums[source_length] += prediction_length
        matched_tokens = sum(
            position < len(predicted_tokens) and predicted_tokens[position] == token
            for position, token in enumerate(target_tokens)
        )
        correct_tokens += matched_tokens
        total_tokens += target_length

        rows.append(
            {
                "index": index,
                "source": source_tokens,
                "target": target_tokens,
                "prediction": predicted_tokens,
                "exact": exact,
                "source_length": source_length,
                "target_length": target_length,
                "prediction_length": prediction_length,
                "length_error": length_error,
                "length_match": prediction_length == target_length,
            }
        )

    if total_tokens == 0:
        raise ValueError("greedy evaluation produced no target tokens.")

    by_length = {
        length: length_exact[length] / total
        for length, total in sorted(length_totals.items())
    }
    avg_target_length_by_source_length = {
        length: target_length_sums[length] / total
        for length, total in sorted(length_totals.items())
    }
    avg_prediction_length_by_source_length = {
        length: prediction_length_sums[length] / total
        for length, total in sorted(length_totals.items())
    }
    generation_diagnostics, sample_diagnostics = compute_generation_diagnostics(
        generation_records,
        eos_id=task.vocab.eos_id,
        unk_id=task.vocab.unk_id,
        max_len=max_len,
    )
    rows = [
        row | asdict(sample_diagnostic)
        for row, sample_diagnostic in zip(rows, sample_diagnostics, strict=True)
    ]
    return (
        GreedyMetrics(
            exact_match=exact_matches / sample_count,
            token_accuracy=correct_tokens / total_tokens,
            examples=sample_count,
            tokens=total_tokens,
            avg_decode_seconds=total_decode_seconds / sample_count,
            length_match_rate=length_matches / sample_count,
            avg_length_error=total_abs_length_error / sample_count,
            by_length=by_length,
            avg_target_length_by_source_length=avg_target_length_by_source_length,
            avg_prediction_length_by_source_length=avg_prediction_length_by_source_length,
        ),
        generation_diagnostics,
        rows,
    )


def compute_generation_diagnostics(
    records: Iterable[GenerationRecord],
    *,
    eos_id: int,
    unk_id: int,
    max_len: int,
) -> tuple[GenerationDiagnostics, list[GenerationSampleDiagnostics]]:
    record_list = list(records)
    if not record_list:
        raise ValueError("generation diagnostics require at least one record.")

    eos_count = 0
    eos_at_expected_position_count = 0
    truncation_count = 0
    total_generated_tokens = 0
    total_unk_tokens = 0
    samples_with_unk = 0
    sample_diagnostics: list[GenerationSampleDiagnostics] = []

    for record in record_list:
        eos_index = _first_index(record.output_ids, eos_id)
        has_eos = eos_index is not None
        eos_at_expected_position = eos_index == record.target_length
        truncated_by_max_len = not has_eos and len(record.output_ids) >= max_len
        unk_count = record.output_ids.count(unk_id)
        has_unk = unk_count > 0

        eos_count += int(has_eos)
        eos_at_expected_position_count += int(eos_at_expected_position)
        truncation_count += int(truncated_by_max_len)
        total_generated_tokens += len(record.output_ids)
        total_unk_tokens += unk_count
        samples_with_unk += int(has_unk)
        sample_diagnostics.append(
            GenerationSampleDiagnostics(
                generated_ids=record.output_ids,
                eos_index=eos_index,
                has_eos=has_eos,
                eos_at_expected_position=eos_at_expected_position,
                truncated_by_max_len=truncated_by_max_len,
                unk_count=unk_count,
                has_unk=has_unk,
            )
        )

    sample_count = len(record_list)
    return (
        GenerationDiagnostics(
            eos_rate=eos_count / sample_count,
            missing_eos_rate=(sample_count - eos_count) / sample_count,
            eos_at_expected_position_rate=eos_at_expected_position_count / sample_count,
            avg_generated_length=total_generated_tokens / sample_count,
            max_len_truncation_rate=truncation_count / sample_count,
            unk_token_rate=_safe_rate(total_unk_tokens, total_generated_tokens),
            samples_with_unk_rate=samples_with_unk / sample_count,
        ),
        sample_diagnostics,
    )


def format_evaluation_report(report_dir: str | Path) -> str:
    report = load_evaluation_report(report_dir)
    teacher = _require_mapping(report, "teacher_forced")
    greedy = _require_mapping(report, "greedy")
    generation_diagnostics = _require_mapping(report, "generation_diagnostics")
    runtime = _require_mapping(report, "runtime")
    artifacts = _require_mapping(report, "artifacts")
    figures = _require_mapping(artifacts, "figures")
    eos_rate = _format_percent(_require_float(generation_diagnostics, "eos_rate"))
    missing_eos_rate = _format_percent(
        _require_float(generation_diagnostics, "missing_eos_rate")
    )
    expected_eos_rate = _format_percent(
        _require_float(generation_diagnostics, "eos_at_expected_position_rate")
    )
    truncation_rate = _format_percent(
        _require_float(generation_diagnostics, "max_len_truncation_rate")
    )
    unk_token_rate = _format_percent(_require_float(generation_diagnostics, "unk_token_rate"))
    samples_with_unk_rate = _format_percent(
        _require_float(generation_diagnostics, "samples_with_unk_rate")
    )
    avg_generated_length = _format_float(
        _require_float(generation_diagnostics, "avg_generated_length")
    )

    lines = [
        "Evaluation",
        f"checkpoint: {_require_str(report, 'checkpoint')}",
        f"split: {_require_str(report, 'split')}",
        f"dataset_size: {_require_int(report, 'dataset_size')}",
        f"output_dir: {Path(report_dir)}",
        f"device: {_require_str(runtime, 'device')}",
        f"batch_size: {_require_int(runtime, 'batch_size')}",
        f"max_len: {_require_int(runtime, 'max_len')}",
        f"greedy_max_samples: {_require_int(runtime, 'max_samples')}",
        "",
        "Teacher forcing",
        f"loss: {_format_float(_require_float(teacher, 'loss'))}",
        f"perplexity: {_format_float(_require_float(teacher, 'perplexity'))}",
        f"token_accuracy: {_format_percent(_require_float(teacher, 'token_accuracy'))}",
        f"examples: {_require_int(teacher, 'examples')}",
        f"tokens: {_require_int(teacher, 'tokens')}",
        "",
        "Greedy generation",
        f"exact_match: {_format_percent(_require_float(greedy, 'exact_match'))}",
        f"token_accuracy: {_format_percent(_require_float(greedy, 'token_accuracy'))}",
        f"examples: {_require_int(greedy, 'examples')}",
        f"tokens: {_require_int(greedy, 'tokens')}",
        f"avg_decode_seconds: {_format_float(_require_float(greedy, 'avg_decode_seconds'))}",
        f"length_match_rate: {_format_percent(_require_float(greedy, 'length_match_rate'))}",
        f"avg_length_error: {_format_float(_require_float(greedy, 'avg_length_error'))}",
        "",
        "Generation diagnostics",
        f"eos_rate: {eos_rate}",
        f"missing_eos_rate: {missing_eos_rate}",
        f"eos_at_expected_position_rate: {expected_eos_rate}",
        f"max_len_truncation_rate: {truncation_rate}",
        f"unk_token_rate: {unk_token_rate}",
        f"samples_with_unk_rate: {samples_with_unk_rate}",
        f"avg_generated_length: {avg_generated_length}",
        "",
        "Exact match by source length",
    ]
    lines.extend(
        f"length {length}: {_format_percent(value)}"
        for length, value in _sorted_numeric_mapping(_require_mapping(greedy, "by_length"))
    )
    lines.extend(
        [
            "",
            "Artifacts",
            f"evaluation_json: {_require_str(artifacts, 'evaluation_json')}",
            f"predictions_jsonl: {_require_str(artifacts, 'predictions')}",
            "figures:",
            f"  loss_curves: {_optional_str(figures, 'loss_curves')}",
            f"  exact_match_by_length: {_require_str(figures, 'exact_match_by_length')}",
            "  generation_length_by_source_length: "
            f"{_require_str(figures, 'generation_length_by_source_length')}",
        ]
    )
    return "\n".join(lines)


def load_evaluation_report(report_dir: str | Path) -> Mapping[str, object]:
    report_path = Path(report_dir) / "evaluation.json"
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict):
        raise ValueError(f"{report_path} must contain a JSON object.")
    return report


def _dataset_for_split(task: Seq2SeqTask, split: DatasetSplit) -> Dataset[Seq2SeqSample]:
    match split:
        case "train":
            return task.make_train_dataset()
        case "val":
            return task.make_eval_dataset()
        case "test":
            return task.make_test_dataset()


def _decode_tokens(vocab: Vocab, ids: Iterable[int]) -> list[str]:
    return vocab.decode(ids, stop_at_eos=True)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _first_index(values: Iterable[int], needle: int) -> int | None:
    for index, value in enumerate(values):
        if value == needle:
            return index
    return None


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _require_mapping(mapping: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"evaluation report field {key!r} must be an object.")
    return value


def _require_str(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise ValueError(f"evaluation report field {key!r} must be a string.")
    return value


def _optional_str(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if value is None:
        return "none"
    if not isinstance(value, str):
        raise ValueError(f"evaluation report field {key!r} must be a string or null.")
    return value


def _require_int(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int):
        raise ValueError(f"evaluation report field {key!r} must be an integer.")
    return value


def _require_float(mapping: Mapping[str, object], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, int | float):
        raise ValueError(f"evaluation report field {key!r} must be numeric.")
    return float(value)


def _sorted_numeric_mapping(mapping: Mapping[str, object]) -> list[tuple[int, float]]:
    values: list[tuple[int, float]] = []
    for key, value in mapping.items():
        if not isinstance(key, str) or not isinstance(value, int | float):
            continue
        values.append((int(key), float(value)))
    return sorted(values)


def _format_float(value: float) -> str:
    return f"{value:.4f}"


def _format_percent(value: float) -> str:
    return f"{value:.2%}"


def _dataset_size(dataset: Dataset[Seq2SeqSample]) -> int:
    if not isinstance(dataset, Sized):
        raise TypeError("evaluation dataset must be sized.")
    return len(dataset)


def _checkpoint_run_dir(checkpoint: Path) -> Path:
    if checkpoint.parent.name == "checkpoints":
        return checkpoint.parent.parent
    return checkpoint.parent


def _default_output_dir(checkpoint: Path) -> Path:
    return _checkpoint_run_dir(checkpoint) / "evaluations" / descending_timestamp_name()


def _metrics_path_for_checkpoint(checkpoint: Path) -> Path:
    return _checkpoint_run_dir(checkpoint) / "metrics.jsonl"


def _plot_training_losses(metrics_path: Path, figures_dir: Path) -> Path | None:
    if not metrics_path.exists():
        return None

    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            step = row.get("step")
            if not isinstance(step, int):
                continue
            for key in ("train_loss", "val_loss", "test_loss"):
                value = row.get(key)
                if isinstance(value, int | float):
                    series[key].append((step, float(value)))

    if not any(series.values()):
        return None

    figure_path = figures_dir / "loss_curves.png"
    plt.figure(figsize=(8, 5))
    for key, values in series.items():
        if values:
            steps, losses = zip(*values, strict=True)
            plt.plot(steps, losses, marker="o", label=key)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Training and evaluation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=160)
    plt.close()
    return figure_path


def _plot_exact_match_by_length(by_length: Mapping[int, float], figures_dir: Path) -> Path:
    figure_path = figures_dir / "exact_match_by_length.png"
    lengths = list(by_length.keys())
    accuracies = [by_length[length] for length in lengths]

    plt.figure(figsize=(8, 5))
    plt.bar([str(length) for length in lengths], accuracies)
    plt.ylim(0.0, 1.0)
    plt.xlabel("source length")
    plt.ylabel("exact match")
    plt.title("Greedy exact match by source length")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=160)
    plt.close()
    return figure_path


def _plot_generation_lengths(
    avg_target_length_by_source_length: Mapping[int, float],
    avg_prediction_length_by_source_length: Mapping[int, float],
    figures_dir: Path,
) -> Path:
    figure_path = figures_dir / "generation_length_by_source_length.png"
    lengths = sorted(
        set(avg_target_length_by_source_length) | set(avg_prediction_length_by_source_length)
    )
    target_lengths = [avg_target_length_by_source_length[length] for length in lengths]
    prediction_lengths = [avg_prediction_length_by_source_length[length] for length in lengths]

    plt.figure(figsize=(8, 5))
    plt.plot(lengths, target_lengths, marker="o", label="target length")
    plt.plot(lengths, prediction_lengths, marker="o", label="prediction length")
    plt.xlabel("source length")
    plt.ylabel("average decoded length")
    plt.title("Generation length by source length")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=160)
    plt.close()
    return figure_path
