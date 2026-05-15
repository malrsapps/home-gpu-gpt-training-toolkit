# Full Training Workflow

This document explains the main chained training workflow used by the toolkit.

The workflow is designed around:
- low-VRAM environments
- resumable training
- chained checkpoint progression
- schedule-driven experimentation
- reproducible small-scale GPT training

The toolkit intentionally separates training into multiple stages instead of treating the entire corpus as a single uninterrupted run.

---

# High-Level Workflow

```text
Wikipedia Dump
    ↓
JSON Extraction
    ↓
Corpus Cleaning
    ↓
Normalization
    ↓
Tokenizer Corpus Export
    ↓
Tokenizer Training
    ↓
Dataset Splitting
    ↓
Schedule Generation
    ↓
Max Steps Calculation
    ↓
Chained Training Loop
    ↓
Checkpoint Continuity
```

---

# Core Workflow Philosophy

The workflow was designed to make experimentation manageable on consumer hardware.

Instead of:
- extremely long uninterrupted runs
- large monolithic datasets
- fragile manual checkpoint handling

the toolkit uses:
- smaller dataset parts
- chained checkpoints
- runtime state tracking
- resumable schedules

This improves:
- experimentation speed
- crash recovery
- workflow reproducibility
- checkpoint organization

---

# Training Lifecycle

The training lifecycle typically follows these stages:

## 1. Dataset Preparation

Raw Wikipedia dumps are extracted and converted into JSONL format.

Relevant scripts:

```text
scripts/data_preparation/
```

---

## 2. Corpus Cleaning

The corpus is cleaned to reduce:
- malformed text
- extremely noisy samples
- invalid lines
- problematic token sequences

---

## 3. Corpus Normalization

Turkish-oriented normalization is applied to improve tokenizer consistency.

This stage evolved through multiple experiments with:
- punctuation normalization
- whitespace normalization
- unicode cleanup
- Turkish character handling

---

## 4. Tokenizer Training

A Byte-Level BPE tokenizer is trained from the processed corpus.

Relevant scripts:

```text
scripts/tokenizer/
```

The tokenizer is later exported into Hugging Face compatible GPT2TokenizerFast files.

---

# Dataset Splitting

After preprocessing, the corpus is divided into smaller token-safe dataset parts.

Relevant script:

```text
scripts/dataset_split/08_split_token_safe_parts.py
```

The split strategy attempts to:
- avoid oversized token sequences
- maintain more stable runtime behavior
- improve checkpoint progression
- reduce training interruption costs

The final workflow used many smaller dataset parts instead of a few large ones.

---

# Schedule Generation

Training is driven by a schedule file.

Relevant scripts:

```text
scripts/schedule/
```

The schedule system stores:
- dataset order
- training status
- checkpoint relationships
- gradient accumulation values
- max_steps values
- evaluation dataset paths

Example:

```text
schedule.json
```

---

# Max Steps Calculation

The toolkit estimates `max_steps` values using:
- token counts
- block size
- batch size
- gradient accumulation
- safety multipliers

This provides more predictable runtime control compared to relying only on epochs.

The resulting values are experimental heuristics rather than strict scientific recommendations.

---

# Chained Training Loop

Training is controlled through:

```text
scripts/training/run_training_loop.py
```

This script:
- reads the schedule
- launches training parts sequentially
- manages retries
- tracks runtime state
- verifies checkpoints
- verifies logs
- updates schedule progression

The loop system was designed to reduce manual intervention during long experiments.

---

# Training Engine

Actual training is performed by:

```text
scripts/training/train_one_part_engine.py
```

The engine:
- loads tokenizer files
- loads training datasets
- loads previous checkpoints
- trains the next dataset part
- saves checkpoints
- updates runtime state
- writes logs and summaries

The engine is normally launched through the training loop rather than directly.

---

# Checkpoint Continuity

The workflow uses chained checkpoint progression.

Example:

```text
part_000 → checkpoint
    ↓
part_001 loads previous checkpoint
    ↓
part_002 loads previous checkpoint
```

This creates a sequential training chain across many dataset parts.

---

# Runtime Tracking Files

The toolkit maintains several runtime state files.

---

## schedule.json

Stores:
- dataset steps
- current index
- training status
- max_steps
- gradient accumulation values

---

## process.json

Stores:
- current runtime status
- retries
- current training part
- latest errors
- verification state

---

## summary.json

Stores:
- completed training summaries
- durations
- verification results
- final losses

---

## state.json

Stores:
- last trained index
- last completed checkpoint
- loop continuation state

---

# Failure Recovery

The workflow includes several recovery-oriented mechanisms:

- retry handling
- resumable schedules
- checkpoint verification
- runtime logging
- continuation tracking

This makes experimentation more manageable on unstable or resource-constrained environments.

---

# Why Smaller Parts Matter

During experimentation, smaller dataset parts significantly improved:

- recovery speed
- experimentation speed
- debugging
- schedule management
- interruption handling

This became especially important on GTX1050-class hardware.

---

# GPU and Runtime Philosophy

The toolkit was intentionally developed around:
- consumer GPUs
- limited VRAM
- slower experimentation cycles
- practical iteration speed

The focus is:
- reproducibility
- experimentation
- workflow management

rather than large-scale distributed infrastructure.

---

# Important

This toolkit is experimental.

The workflow may evolve over time as additional:
- training strategies
- tokenizer experiments
- dataset structures
- scheduling approaches

are explored.

The toolkit reflects a practical experimental workflow rather than a universal training standard.