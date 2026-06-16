from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TaskConfig:
    name: str = "reverse"
    digit_count: int = 10
    min_len: int = 3
    max_len: int = 12
    train_size: int = 12_000
    val_size: int = 512
    seed: int = 42
    source_eos: bool = True
    samples_to_export: int = 32
    data_dir: Path = Path("data/generated")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    d_model: int = 64
    num_heads: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.1
    max_positions: int = 64


@dataclass(frozen=True, slots=True)
class TrainConfig:
    batch_size: int = 64
    max_steps: int = 1_000
    eval_every: int = 100
    log_every: int = 20
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    device: str = "auto"
    run_dir: Path = Path("runs")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    task: TaskConfig = TaskConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    return ExperimentConfig(
        task=_task_config(raw.get("task", {})),
        model=_model_config(raw.get("model", {})),
        train=_train_config(raw.get("train", {})),
    )


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    return _to_builtin(asdict(config))


def config_from_dict(raw: Mapping[str, Any]) -> ExperimentConfig:
    return ExperimentConfig(
        task=_task_config(raw.get("task", {})),
        model=_model_config(raw.get("model", {})),
        train=_train_config(raw.get("train", {})),
    )


def _task_config(raw: Mapping[str, Any]) -> TaskConfig:
    defaults = _field_defaults(TaskConfig)
    return TaskConfig(
        name=str(raw.get("name", defaults["name"])),
        digit_count=int(raw.get("digit_count", defaults["digit_count"])),
        min_len=int(raw.get("min_len", defaults["min_len"])),
        max_len=int(raw.get("max_len", defaults["max_len"])),
        train_size=int(raw.get("train_size", defaults["train_size"])),
        val_size=int(raw.get("val_size", defaults["val_size"])),
        seed=int(raw.get("seed", defaults["seed"])),
        source_eos=bool(raw.get("source_eos", defaults["source_eos"])),
        samples_to_export=int(raw.get("samples_to_export", defaults["samples_to_export"])),
        data_dir=Path(raw.get("data_dir", defaults["data_dir"])),
    )


def _model_config(raw: Mapping[str, Any]) -> ModelConfig:
    defaults = _field_defaults(ModelConfig)
    return ModelConfig(
        d_model=int(raw.get("d_model", defaults["d_model"])),
        num_heads=int(raw.get("num_heads", defaults["num_heads"])),
        num_encoder_layers=int(raw.get("num_encoder_layers", defaults["num_encoder_layers"])),
        num_decoder_layers=int(raw.get("num_decoder_layers", defaults["num_decoder_layers"])),
        d_ff=int(raw.get("d_ff", defaults["d_ff"])),
        dropout=float(raw.get("dropout", defaults["dropout"])),
        max_positions=int(raw.get("max_positions", defaults["max_positions"])),
    )


def _train_config(raw: Mapping[str, Any]) -> TrainConfig:
    defaults = _field_defaults(TrainConfig)
    return TrainConfig(
        batch_size=int(raw.get("batch_size", defaults["batch_size"])),
        max_steps=int(raw.get("max_steps", defaults["max_steps"])),
        eval_every=int(raw.get("eval_every", defaults["eval_every"])),
        log_every=int(raw.get("log_every", defaults["log_every"])),
        learning_rate=float(raw.get("learning_rate", defaults["learning_rate"])),
        weight_decay=float(raw.get("weight_decay", defaults["weight_decay"])),
        grad_clip=float(raw.get("grad_clip", defaults["grad_clip"])),
        device=str(raw.get("device", defaults["device"])),
        run_dir=Path(raw.get("run_dir", defaults["run_dir"])),
    )


def _field_defaults(cls: type[Any]) -> dict[str, Any]:
    return {field.name: field.default for field in fields(cls)}


def _to_builtin(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_builtin(item) for item in value]
    return value
