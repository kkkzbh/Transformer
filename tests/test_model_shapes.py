from __future__ import annotations

import torch

from transformer.config import ModelConfig, TaskConfig
from transformer.data.tasks.reverse import ReverseTask
from transformer.modeling.transformer import Seq2SeqTransformer
from transformer.training.losses import seq2seq_cross_entropy


def test_model_forward_shape_and_backward() -> None:
    task = ReverseTask(TaskConfig(train_size=8, val_size=8, min_len=3, max_len=5))
    dataset = task.make_train_dataset()
    collate = task.make_collate_fn()
    batch = collate([dataset[0], dataset[1]])
    model = Seq2SeqTransformer(
        vocab_size=len(task.vocab),
        pad_id=task.vocab.pad_id,
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

    logits = model(batch.src, batch.tgt_in)
    loss = seq2seq_cross_entropy(logits, batch.tgt_out, pad_id=task.vocab.pad_id)
    loss.backward()

    assert logits.shape == (*batch.tgt_in.shape, len(task.vocab))
    assert torch.isfinite(loss)
    assert any(param.grad is not None for param in model.parameters())
