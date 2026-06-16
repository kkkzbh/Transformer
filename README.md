# Transformer

A small, extensible PyTorch project for implementing Transformer sequence-to-sequence
models from first principles.

The first task is `reverse`: given a sequence of digit tokens, generate the reversed
sequence. The project is structured so later tasks can be added under
`src/transformer/data/tasks/` without changing the model or training loop.
The synthetic reverse data is generated deterministically and split into
train/validation/test partitions by ratio from `configs/reverse.toml`.

## What I implemented

- A configurable encoder-decoder Transformer in PyTorch, including embeddings,
  sinusoidal positional encoding, multi-head attention, pre-LN encoder/decoder
  layers, causal masks, and padding masks.
- A synthetic `reverse` sequence-to-sequence task with deterministic data
  generation, vocabulary handling, train/validation/test splits, and exported
  inspectable samples.
- A CLI workflow for preparing data, training, checkpoint loading, greedy
  inference, evaluation, and KV cache benchmarking.
- Model-owned KV cache inference in an HF-style boundary: `forward()` accepts
  `encoder_outputs`, `past_key_values`, `use_cache`, and `cache_position`, while
  greedy decoding treats the cache as an opaque model state.
- Evaluation reports with teacher-forced metrics, greedy generation metrics,
  EOS/UNK generation diagnostics, prediction JSONL output, and figures.
- A benchmark command that compares cached and uncached decoder inference inside
  one Python process, separating checkpoint loading, encoder time, and decode
  time.
- Codex App Actions for prepare-data, 3000-step training, inference, evaluation,
  and KV cache benchmarking.

## Quick Start

```bash
uv sync
uv run transformer prepare-data --config configs/reverse.toml
uv run transformer train --config configs/reverse.toml --max-steps 200
uv run transformer infer --checkpoint runs/reverse/<run-id>/checkpoints/best.pt --input "3 7 1 9"
uv run transformer evaluate --checkpoint runs/reverse/<run-id>/checkpoints/best.pt
uv run transformer benchmark-infer --checkpoint runs/reverse/<run-id>/checkpoints/best.pt --input "2 3 4 5"
```

`evaluate` prints a terminal summary for teacher-forcing metrics and greedy
generation metrics, then writes `evaluation.json`, `predictions.jsonl`, and
figures under the run's `evaluations/<timestamp>/` directory.

`benchmark-infer` compares full-prefix decoder inference against KV-cache
inference for the same checkpoint and input. It reports whether both paths
produce the same output, then prints decode timing and speedup numbers.

## Layout

```text
configs/                  Experiment configs
data/generated/           Generated vocabularies and inspectable samples
runs/                     Training outputs and checkpoints
src/transformer/data/     Vocab, batch, and task code
src/transformer/modeling/ Transformer implementation
src/transformer/training/ Training loop and checkpointing
src/transformer/inference/Decoding utilities
src/transformer/benchmarking/Inference benchmarks
src/transformer/evaluation/Evaluation reports and figures
tests/                    Unit tests
```

New run and evaluation directories use names like
`0-8235748797-20260616-180642`: the leading reverse-time key makes normal
ascending filename sort show the newest directories first, while the suffix keeps
the creation time readable.
