#!/usr/bin/env python3
"""
Tiny public demo training script.

This script demonstrates a minimal GPT-style training flow on a tiny dataset.
It does not include the full chained-training engine, schedule recovery,
state tracking, or checkpoint orchestration used in the full toolkit.

Example:
    python demo/demo_train_tiny.py \
        --data demo/sample_data/demo_wikipedia.jsonl \
        --tokenizer demo/tokenizer \
        --output demo/tiny_checkpoint \
        --max-steps 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
)


class TinyJsonlDataset(Dataset):
    def __init__(self, path: Path, tokenizer: GPT2TokenizerFast, block_size: int) -> None:
        self.samples = []

        with path.open("r", encoding="utf-8") as file:
            for line in file:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text = str(obj.get("text", "")).strip()
                if not text:
                    continue

                tokens = tokenizer(
                    text,
                    max_length=block_size,
                    padding="max_length",
                    truncation=True,
                    return_tensors=None,
                )

                self.samples.append(
                    {
                        "input_ids": torch.tensor(tokens["input_ids"], dtype=torch.long),
                        "attention_mask": torch.tensor(tokens["attention_mask"], dtype=torch.long),
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tiny GPT-style demo training.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("demo/tiny_checkpoint"))
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Dataset not found: {args.data}")

    if not args.tokenizer.exists():
        raise FileNotFoundError(f"Tokenizer not found: {args.tokenizer}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")

    tokenizer = GPT2TokenizerFast.from_pretrained(str(args.tokenizer))
    tokenizer.pad_token = tokenizer.eos_token

    dataset = TinyJsonlDataset(args.data, tokenizer, args.block_size)

    config = GPT2Config(
        vocab_size=len(tokenizer),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        n_ctx=args.block_size,
    )

    model = GPT2LMHeadModel(config)
    model.to(device)

    training_args = TrainingArguments(
        output_dir=str(args.output),
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        max_steps=args.max_steps,
        save_steps=args.max_steps,
        save_total_limit=1,
        logging_steps=max(1, args.max_steps // 5),
        report_to="none",
        fp16=False,
        dataloader_num_workers=0,
        learning_rate=5e-5,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))

    print(f"Tiny demo checkpoint saved to: {args.output}")


if __name__ == "__main__":
    main()
