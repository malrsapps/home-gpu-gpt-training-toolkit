# Design Notes

This document describes some of the practical decisions behind the toolkit.

The workflow evolved through repeated experimentation on low-VRAM consumer hardware rather than large-scale infrastructure.

The goal was not to build a frontier-scale model, but to explore reproducible GPT-style training workflows on a home computer.

---

## Why chained training?

Early experiments showed that training large datasets in a single uninterrupted run was difficult on low-end hardware.

Long runs created several problems:

- slow iteration cycles
- difficult crash recovery
- unstable experimentation
- inefficient checkpoint management

The chained training structure was introduced to make training sessions smaller and easier to manage.

Instead of treating the dataset as a single monolithic corpus, the workflow trains sequential dataset parts while preserving checkpoint continuity.

---

## Why split into many parts?

The number of dataset parts evolved gradually through experimentation.

Early experiments used:
- single-part datasets
- 6-part schedules
- 12-part schedules

These approaches still produced long training sessions and slow iteration cycles.

Increasing the number of parts significantly improved:
- restart flexibility
- checkpoint safety
- runtime management
- experimentation speed

---

## Why 120 parts?

The final workflow used approximately 120 dataset parts.

On GTX1050-class hardware, this reduced individual training sessions from very long runs into shorter and more manageable iterations.

Smaller chained sessions also improved:
- monitoring
- debugging
- schedule recovery
- experimentation speed

This value is not universal.

Different hardware and datasets may require different split strategies.

---

## Why token-safe splitting?

Simple line-based splitting can produce highly inconsistent token lengths.

Large token variance creates unstable training batches and unpredictable runtime behavior.

The token-safe splitting workflow attempts to:
- keep samples within a token range
- avoid oversized sequences
- preserve more stable training behavior

Sentence-aware splitting was introduced after earlier experiments with naive chunking approaches.

---

## Why schedule-based training?

The training process evolved into a schedule-driven workflow because manual part tracking became difficult over time.

The schedule system allows:
- sequential chained training
- resumable workflows
- automatic checkpoint progression
- runtime tracking
- controlled experimentation

The toolkit stores runtime information in files such as:

```text
schedule.json
process.json
summary.json
state.json
```

This makes long-running experiments easier to monitor and resume.

## Why max_steps instead of only epochs?

Early experiments relied mostly on epoch-based training.

Later experiments showed that step-based scheduling gave more predictable control over:

- runtime duration
- checkpoint frequency
- iteration speed
- GPU utilization

The toolkit therefore calculates estimated `max_steps` values from:

- token counts
- block size
- gradient accumulation
- batch size

This is still an experimental heuristic rather than a strict scientific formula.

---

## Why custom tokenizer?

The tokenizer workflow evolved through repeated experimentation with Turkish-oriented corpora.

Several tokenizer configurations were tested before arriving at the current setup.

The goal was to:

- improve token consistency
- reduce noisy segmentation
- preserve Turkish character behavior
- maintain compatibility with GPT-style workflows

The final tokenizer settings are practical experimental choices rather than universal recommendations.

---

## Why low-VRAM settings?

The toolkit was designed around consumer-grade hardware constraints.

Many settings were adjusted specifically for low-VRAM environments, including:

- small batch sizes
- gradient accumulation
- chained checkpoints
- segmented schedules
- lightweight runtime tracking

The original experiments were performed on GTX1050-class hardware under Linux-based environments.

---

## What this toolkit is not

This project is not intended to be:

- a frontier-scale training framework
- a production-grade distributed training system
- a replacement for large modern LLM infrastructure

The focus is educational experimentation, reproducibility, and practical low-resource workflows.

---

## Notes

The workflow changed many times during experimentation.

Some earlier experiments involved:

- alternative corpora
- different tokenizer approaches
- different schedule structures
- different dataset splitting strategies

The public repository intentionally focuses on the final reproducible workflow rather than every historical experiment.