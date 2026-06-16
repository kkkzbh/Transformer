from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from transformer.config import ExperimentConfig, ModelConfig, config_from_dict, config_to_dict
from transformer.data.registry import create_task
from transformer.modeling.transformer import Seq2SeqTransformer


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    config: ExperimentConfig,
    step: int,
    best_val_loss: float,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config_to_dict(config),
            "step": step,
            "best_val_loss": best_val_loss,
        },
        checkpoint_path,
    )


def load_model_for_inference(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> tuple[Seq2SeqTransformer, Any, ExperimentConfig]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = config_from_dict(checkpoint["config"])
    task = create_task(config.task)
    model = Seq2SeqTransformer(
        vocab_size=len(task.vocab),
        pad_id=task.vocab.pad_id,
        config=ModelConfig(**config_to_dict(config)["model"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, task, config
