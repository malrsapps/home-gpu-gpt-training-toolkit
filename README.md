# Home GPU GPT Training Toolkit

Train small GPT-style language models on consumer hardware.

This repository demonstrates practical low-resource language model experimentation workflows designed for local environments and older GPUs.

---

## Project Overview

This project explores practical GPT-style training workflows built for:

- low-VRAM GPUs
- local experimentation
- small-model iteration
- reproducible workflows
- tokenizer experimentation
- lightweight inference testing

The workflow was developed and tested primarily on:

- NVIDIA GTX1050
- 24GB RAM
- Pop!_OS 22.04

The goal is not to compete with large-scale cloud training systems.

The goal is to make small-model experimentation more accessible on normal hardware.

---

## Why This Project Exists

Most modern LLM discussions assume access to:
- cloud infrastructure
- large GPU clusters
- expensive hardware

This project explores a different question:

> Can useful GPT-style experimentation workflows run on older consumer hardware?

During development, multiple:
- tokenizer strategies
- dataset structures
- training schedules
- low-VRAM optimizations

were tested.

Several important observations emerged:

- structured datasets significantly affect model behavior
- small datasets may intentionally overfit for specialization
- tokenizer quality strongly affects generation quality
- workflow organization matters as much as the model itself
- low-resource experimentation requires practical engineering tradeoffs

---

## What This Repository Demonstrates

### Data Preparation

- Wikipedia extraction workflows
- JSONL dataset preparation
- corpus cleaning
- Turkish-oriented normalization
- tokenizer corpus export

### Tokenizer Workflow

- Byte-Level BPE tokenizer training
- GPT2TokenizerFast export
- tokenizer experimentation

### Dataset Preparation

- token-safe dataset splitting
- small dataset chunking
- experimental workflow preparation

### Experimental Training

- tiny GPT-style demo training
- low-resource experimentation
- local checkpoint generation
- local inference testing

---

## Documentation

- [Quick Start](docs/01_quick_start.md)
- [Public Demo Scope](docs/02_public_demo_scope.md)
- [Design Notes](docs/03_design_notes.md)
- [Full Training Workflow](docs/05_full_training_workflow.md)
- [Troubleshooting](docs/06_troubleshooting.md)
- [Inference Workflow](docs/07_inference_workflow.md)

---

## Training Pipeline Overview

```text
Raw Wikipedia Dump
        ↓
JSON Extraction
        ↓
JSONL Corpus Preparation
        ↓
Corpus Cleaning
        ↓
Turkish Text Normalization
        ↓
Tokenizer Corpus Export
        ↓
Custom GPT Tokenizer Training
        ↓
Token-Safe Dataset Splitting
        ↓
Experimental GPT Training
```

---

## Public Demo Scope

This public repository contains:

- tiny demo datasets
- tiny GPT-style training examples
- local inference examples
- tokenizer workflows
- documentation and experimentation notes

The repository is intended for:
- educational use
- experimentation
- workflow exploration
- local testing

Advanced orchestration workflows are maintained separately.

---

## Hardware Philosophy

This repository intentionally focuses on constrained environments.

The workflow was designed around:

- limited VRAM
- long-running local experimentation
- iterative testing
- small-model workflows
- offline/local development

The focus is practical experimentation rather than large-scale production infrastructure.

---

## Current Limitations

This project is experimental.

Current limitations include:

- small-scale model capacity
- unstable generations
- numerical reasoning limitations
- long training durations on low-end GPUs
- manual workflow stages

Some workflows may require:
- Linux familiarity
- Python familiarity
- basic CUDA/PyTorch troubleshooting

---

## Important Notes

- Large datasets are not distributed with this repository
- Third-party datasets may require separate licensing review
- This project is intended for research and educational purposes
- Generated outputs may contain hallucinations or incorrect information

---

## License

Released under the Apache 2.0 License.

---

## Related Article

**Training GPT-Style Models on a GTX1050: What I Learned**

https://medium.com/@malrsapps/training-gpt-style-models-on-a-gtx1050-what-i-learned-3a1534d75cc5

---

**Created by MA**