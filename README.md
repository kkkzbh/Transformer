# Transformer

A small, extensible PyTorch project for implementing Transformer sequence-to-sequence
models from first principles.

The first task is `reverse`: given a sequence of digit tokens, generate the reversed
sequence. The project is structured so later tasks can be added under
`src/transformer/data/tasks/` without changing the model or training loop.

## Quick Start

```bash
uv sync
uv run transformer prepare-data --config configs/reverse.toml
uv run transformer train --config configs/reverse.toml --max-steps 200
uv run transformer infer --checkpoint runs/reverse/<run-id>/checkpoints/best.pt --input "3 7 1 9"
```

## Layout

```text
configs/                  Experiment configs
data/generated/           Generated vocabularies and inspectable samples
runs/                     Training outputs and checkpoints
src/transformer/data/     Vocab, batch, and task code
src/transformer/modeling/ Transformer implementation
src/transformer/training/ Training loop and checkpointing
src/transformer/inference/Decoding utilities
tests/                    Unit tests
```
