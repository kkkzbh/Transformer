# Transformer

[English](README.md) | 简体中文

一个小型、可扩展的 PyTorch 项目，用来从零实现 Transformer sequence-to-sequence 模型。

第一个任务是 `reverse`：给定一串数字 token，生成反转后的序列。项目结构支持之后在 `src/transformer/data/tasks/` 下添加新任务，而不需要改模型或训练循环。合成 reverse 数据会确定性生成，并按 `configs/reverse.toml` 中的比例划分 train/validation/test。

## 已实现内容

- 可配置的 encoder-decoder Transformer，包含 embedding、sinusoidal positional encoding、multi-head attention、pre-LN encoder/decoder layer、causal mask 和 padding mask。
- 合成 `reverse` sequence-to-sequence 任务，包含确定性数据生成、词表处理、train/validation/test 划分和可检查样本导出。
- CLI 工作流：准备数据、训练、加载 checkpoint、greedy inference、evaluation 和 KV cache benchmark。
- 模型拥有的 KV cache inference 边界，风格接近 HF：`forward()` 接受 `encoder_outputs`、`past_key_values`、`use_cache` 和 `cache_position`，而 greedy decoding 把 cache 当作不透明模型状态。
- evaluation 报告，包含 teacher-forced metrics、greedy generation metrics、EOS/UNK 生成诊断、prediction JSONL 输出和图表。
- benchmark 命令在同一个 Python 进程内比较 cached 和 uncached decoder inference，区分 checkpoint loading、encoder time 和 decode time。
- Codex App Actions：prepare-data、3000-step training、inference、evaluation 和 KV cache benchmarking。

## 快速开始

```bash
uv sync
uv run transformer prepare-data --config configs/reverse.toml
uv run transformer train --config configs/reverse.toml --max-steps 200
uv run transformer infer --checkpoint runs/reverse/<run-id>/checkpoints/best.pt --input "3 7 1 9"
uv run transformer evaluate --checkpoint runs/reverse/<run-id>/checkpoints/best.pt
uv run transformer benchmark-infer --checkpoint runs/reverse/<run-id>/checkpoints/best.pt --input "2 3 4 5"
```

`evaluate` 会在终端打印 teacher-forcing metrics 和 greedy generation metrics 的摘要，并把 `evaluation.json`、`predictions.jsonl` 和图表写入该 run 的 `evaluations/<timestamp>/` 目录。

`benchmark-infer` 会用同一个 checkpoint 和输入比较 full-prefix decoder inference 与 KV-cache inference。它会报告两条路径是否生成相同输出，然后打印 decode timing 和 speedup。

## 目录结构

```text
configs/                  实验配置
data/generated/           生成的词表和可检查样本
runs/                     训练输出和 checkpoints
src/transformer/data/     Vocab、batch 和 task 代码
src/transformer/modeling/ Transformer 实现
src/transformer/training/ 训练循环和 checkpoint
src/transformer/inference/Decoding 工具
src/transformer/benchmarking/Inference benchmark
src/transformer/evaluation/Evaluation 报告和图表
tests/                    单元测试
```

新的 run 和 evaluation 目录名类似 `0-8235748797-20260616-180642`：前面的 reverse-time key 让普通升序文件名排序显示最新目录，后缀保留可读创建时间。
