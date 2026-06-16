from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformer.config import ExperimentConfig, config_to_dict
from transformer.data.batch import Seq2SeqBatch
from transformer.data.registry import create_task
from transformer.modeling.transformer import Seq2SeqTransformer
from transformer.training.checkpointing import save_checkpoint
from transformer.training.losses import seq2seq_cross_entropy
from transformer.utils.device import resolve_device
from transformer.utils.run_names import descending_timestamp_name
from transformer.utils.seed import set_seed


def train(
    config: ExperimentConfig,
    *,
    config_path: str | Path | None = None,
    max_steps: int | None = None,
) -> Path:
    if max_steps is not None:
        config = replace(config, train=replace(config.train, max_steps=max_steps))

    set_seed(config.task.seed)
    device = resolve_device(config.train.device)
    task = create_task(config.task)
    task.prepare()

    run_dir = _make_run_dir(config)
    if config_path is not None:
        shutil.copy2(config_path, run_dir / "config.toml")
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config_to_dict(config), handle, ensure_ascii=True, indent=2)
        handle.write("\n")

    train_loader = DataLoader(
        task.make_train_dataset(),
        batch_size=config.train.batch_size,
        shuffle=True,
        collate_fn=task.make_collate_fn(),
    )
    eval_loader = DataLoader(
        task.make_eval_dataset(),
        batch_size=config.train.batch_size,
        shuffle=False,
        collate_fn=task.make_collate_fn(),
    )
    test_loader = DataLoader(
        task.make_test_dataset(),
        batch_size=config.train.batch_size,
        shuffle=False,
        collate_fn=task.make_collate_fn(),
    )

    model = Seq2SeqTransformer(
        vocab_size=len(task.vocab),
        pad_id=task.vocab.pad_id,
        config=config.model,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )

    metrics_path = run_dir / "metrics.jsonl"
    best_val_loss = float("inf")
    best_model_state: dict[str, torch.Tensor] | None = None
    step = 0
    progress = tqdm(total=config.train.max_steps, desc="train", dynamic_ncols=True)

    while step < config.train.max_steps:
        for batch in train_loader:
            step += 1
            model.train()
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.src, batch.tgt_in)
            loss = seq2seq_cross_entropy(logits, batch.tgt_out, pad_id=task.vocab.pad_id)
            loss.backward()
            if config.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            optimizer.step()

            progress.update(1)
            progress.set_postfix(loss=f"{loss.item():.4f}")

            if step % config.train.log_every == 0:
                _append_metric(metrics_path, {"step": step, "train_loss": loss.item()})

            if step % config.train.eval_every == 0 or step == config.train.max_steps:
                val_loss = evaluate(model, eval_loader, device=device, pad_id=task.vocab.pad_id)
                _append_metric(metrics_path, {"step": step, "val_loss": val_loss})
                save_checkpoint(
                    run_dir / "checkpoints" / "last.pt",
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    step=step,
                    best_val_loss=min(best_val_loss, val_loss),
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in model.state_dict().items()
                    }
                    save_checkpoint(
                        run_dir / "checkpoints" / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        step=step,
                        best_val_loss=best_val_loss,
                    )

            if step >= config.train.max_steps:
                break

    progress.close()
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)
    test_loss = evaluate(model, test_loader, device=device, pad_id=task.vocab.pad_id)
    _append_metric(metrics_path, {"step": step, "test_loss": test_loss})
    return run_dir


@torch.no_grad()
def evaluate(
    model: Seq2SeqTransformer,
    loader: Iterable[Seq2SeqBatch],
    *,
    device: torch.device,
    pad_id: int,
) -> float:
    model.eval()
    total_loss = 0.0
    batches = 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.src, batch.tgt_in)
        loss = seq2seq_cross_entropy(logits, batch.tgt_out, pad_id=pad_id)
        total_loss += loss.item()
        batches += 1
    if batches == 0:
        raise ValueError("evaluation loader produced no batches.")
    return total_loss / batches


def _make_run_dir(config: ExperimentConfig) -> Path:
    run_dir = config.train.run_dir / config.task.name / descending_timestamp_name()
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
    return run_dir


def _append_metric(path: Path, metric: dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metric, ensure_ascii=True) + "\n")
