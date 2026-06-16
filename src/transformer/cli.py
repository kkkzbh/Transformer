from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch

from transformer.config import load_config
from transformer.data.registry import create_task
from transformer.evaluation.report import evaluate_checkpoint, format_evaluation_report
from transformer.inference.decoding import greedy_decode
from transformer.training.checkpointing import load_model_for_inference
from transformer.training.loop import train
from transformer.utils.device import resolve_device


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    match args.command:
        case "prepare-data":
            config = load_config(args.config)
            task = create_task(config.task)
            task_dir = task.prepare()
            print(f"prepared {task.name} data at {task_dir}")
        case "train":
            config = load_config(args.config)
            run_dir = train(config, config_path=args.config, max_steps=args.max_steps)
            print(f"run_dir={run_dir}")
        case "infer":
            device = resolve_device(args.device)
            model, task, _ = load_model_for_inference(args.checkpoint, device=device)
            src_ids = task.encode_source(args.input)
            src = torch.tensor([src_ids], dtype=torch.long, device=device)
            output_ids = greedy_decode(
                model,
                src,
                bos_id=task.vocab.bos_id,
                eos_id=task.vocab.eos_id,
                max_len=args.max_len,
            )
            print(task.decode_target(output_ids))
        case "evaluate":
            report_dir = evaluate_checkpoint(
                args.checkpoint,
                split=args.split,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                max_samples=args.max_samples,
                max_len=args.max_len,
                device_name=args.device,
            )
            print(format_evaluation_report(report_dir))
        case _:
            parser.error("missing command")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="transformer")
    subparsers = parser.add_subparsers(dest="command")

    prepare = subparsers.add_parser("prepare-data")
    prepare.add_argument("--config", type=Path, required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", type=Path, required=True)
    train_parser.add_argument("--max-steps", type=int, default=None)

    infer = subparsers.add_parser("infer")
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--input", required=True)
    infer.add_argument("--device", default="auto")
    infer.add_argument("--max-len", type=int, default=32)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--split", choices=["train", "val", "test"], default="test")
    evaluate.add_argument("--output-dir", type=Path, default=None)
    evaluate.add_argument("--batch-size", type=int, default=None)
    evaluate.add_argument("--max-samples", type=int, default=512)
    evaluate.add_argument("--max-len", type=int, default=None)
    evaluate.add_argument("--device", default="auto")

    return parser


if __name__ == "__main__":
    main()
