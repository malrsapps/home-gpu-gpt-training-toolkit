#!/usr/bin/env python3
"""
Tiny public demo inference script.

Example:
    python demo/demo_inference.py \
        --model demo/tiny_checkpoint \
        --tokenizer demo/tokenizer \
        --prompt "Merhaba" \
        --max-new-tokens 40
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tiny demo inference.")
    parser.add_argument("--model", type=Path, default=Path("demo/tiny_checkpoint"))
    parser.add_argument("--tokenizer", type=Path, default=Path("demo/tokenizer"))
    parser.add_argument("--prompt", type=str, default="Merhaba")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--no-sampling", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.model.exists():
        raise FileNotFoundError(
            f"Model directory not found: {args.model}. "
            "Run demo/demo_train_tiny.py first."
        )

    if not args.tokenizer.exists():
        raise FileNotFoundError(f"Tokenizer directory not found: {args.tokenizer}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")

    tokenizer = GPT2TokenizerFast.from_pretrained(str(args.tokenizer))
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(str(args.model))
    model.to(device)
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=not args.no_sampling,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n--- Prompt ---")
    print(args.prompt)

    print("\n--- Output ---")
    print(text)


if __name__ == "__main__":
    main()
