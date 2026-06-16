# Transformer

A small, extensible PyTorch project for implementing Transformer sequence-to-sequence
models from first principles.

The first task is `reverse`: given a sequence of digit tokens, generate the reversed
sequence. The project is structured so later tasks can be added under
`src/transformer/data/tasks/` without changing the model or training loop.
The synthetic reverse data is generated deterministically and split into
train/validation/test partitions by ratio from `configs/reverse.toml`.

## Quick Start

```bash
uv sync
uv run transformer prepare-data --config configs/reverse.toml
uv run transformer train --config configs/reverse.toml --max-steps 200
uv run transformer infer --checkpoint runs/reverse/<run-id>/checkpoints/best.pt --input "3 7 1 9"
uv run transformer evaluate --checkpoint runs/reverse/<run-id>/checkpoints/best.pt
```

`evaluate` prints a terminal summary for teacher-forcing metrics and greedy
generation metrics, then writes `evaluation.json`, `predictions.jsonl`, and
figures under the run's `evaluations/<timestamp>/` directory.

## Layout

```text
configs/                  Experiment configs
data/generated/           Generated vocabularies and inspectable samples
runs/                     Training outputs and checkpoints
src/transformer/data/     Vocab, batch, and task code
src/transformer/modeling/ Transformer implementation
src/transformer/training/ Training loop and checkpointing
src/transformer/inference/Decoding utilities
src/transformer/evaluation/Evaluation reports and figures
tests/                    Unit tests
```

New run and evaluation directories use names like
`0-8235748797-20260616-180642`: the leading reverse-time key makes normal
ascending filename sort show the newest directories first, while the suffix keeps
the creation time readable.
