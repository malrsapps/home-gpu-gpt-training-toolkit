# Troubleshooting Guide

This document lists common issues encountered during experimentation with the toolkit.

The workflow was developed primarily on Linux-based environments with low-VRAM consumer GPUs.

Because Python, CUDA, PyTorch, and tokenizer dependencies evolve rapidly, some environments may require additional adjustments.

---

# CUDA / PyTorch Version Mismatch

## Symptoms

Examples:

```text
CUDA initialization failed
Torch not compiled with CUDA enabled
RuntimeError: CUDA error
```

or:

```text
torch.cuda.is_available() == False
```

---

## Possible Causes

- incompatible CUDA version
- incompatible PyTorch build
- driver mismatch
- broken virtual environment
- conflicting package installations

---

## Suggested Checks

Check NVIDIA driver:

```bash
nvidia-smi
```

Check Python:

```bash
python --version
```

Check Torch:

```bash
python -c "import torch; print(torch.__version__)"
```

Check CUDA visibility:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Notes

The toolkit was tested primarily using:

- NVIDIA GTX1050
- Pop!_OS 22.04
- CUDA 12.x
- PyTorch 2.6.x

Different systems may require different dependency combinations.

---

# Tokenizer Path Errors

## Symptoms

Examples:

```text
OSError: tokenizer is not a local folder
```

or:

```text
Can't load tokenizer
```

---

## Possible Causes

- wrong tokenizer path
- tokenizer export not completed
- missing tokenizer files
- incorrect working directory

---

## Suggested Checks

Verify tokenizer directory:

```text
tokenizer/
```

or:

```text
demo/tokenizer/
```

Verify required files exist:

```text
tokenizer.json
tokenizer_config.json
vocab.json
merges.txt
```

---

# Out Of Memory (OOM) Errors

## Symptoms

Examples:

```text
CUDA out of memory
```

or sudden training crashes during batching.

---

## Suggested Fixes

Reduce one or more of the following:

- batch size
- block size
- gradient accumulation
- dataset part size

Smaller dataset parts may also improve runtime stability.

---

# Slow Training Speed

Training on low-end GPUs may be significantly slower than cloud environments.

Long runtimes are expected on consumer hardware.

---

## Possible Improvements

- reduce model size
- reduce block size
- reduce tokenizer vocabulary size
- reduce max_steps
- reduce dataset size

---

# Hardware Temperature Notes

Long-running training sessions may increase GPU and system temperatures, especially on older consumer GPUs, laptops, or systems with limited airflow.

During longer experiments:

- monitor GPU temperature
- ensure proper cooling and ventilation
- avoid blocking laptop air vents
- consider shorter training runs for testing
- stop training if the system becomes unstable

This toolkit is intended for experimentation, and users are responsible for monitoring their own hardware.

---

# schedule.json and state.json Mismatch

## Symptoms

Examples:

```text
loop resumed incorrectly
```

or:

```text
checkpoint progression mismatch
```

---

## Possible Causes

- interrupted training
- manually edited schedule files
- missing checkpoints
- stale runtime state files

---

## Suggested Fixes

Verify consistency between:

```text
schedule.json
state.json
process.json
summary.json
```

Check that expected checkpoints exist.

If necessary, recreate the schedule or reset runtime state files carefully.

---

# Missing Checkpoints

## Symptoms

Examples:

```text
checkpoint not found
```

or:

```text
cannot load previous checkpoint
```

---

## Suggested Checks

Verify checkpoint directories:

```text
checkpoints/
```

or:

```text
demo/checkpoints/
```

Ensure previous training parts completed successfully.

---

# Demo Outputs Generated In Wrong Location

## Symptoms

Examples:

```text
demo files created in repository root
```

---

## Possible Causes

Runtime output paths were not explicitly provided.

---

## Suggested Fix

Use explicit runtime arguments:

```bash
--checkpoints-dir
--tmp-dir
--logs-dir
--process-path
--summary-path
--state-path
```

when running demo workflows.

---

# WikiExtractor Issues

## Symptoms

Examples:

```text
ModuleNotFoundError
```

or extraction failures.

---

## Suggested Checks

Verify dependencies:

```bash
pip install -r requirements-core.txt
```

Ensure Wikipedia dump paths are correct.

---

# Linux Dependency Problems

Some Linux environments may require additional system packages.

Examples:

- build-essential
- python3-dev
- CUDA libraries
- NVIDIA drivers

Package requirements vary between distributions.

---

# Virtual Environment Problems

## Symptoms

Examples:

```text
package conflicts
```

or inconsistent runtime behavior.

---

## Suggested Fixes

Create a fresh virtual environment:

```bash
python -m venv .venv
```

Activate:

```bash
source .venv/bin/activate
```

Reinstall dependencies:

```bash
pip install -r requirements-core.txt
```

---

# Important Notes

This toolkit is experimental.

Different:
- GPUs
- CUDA versions
- Linux distributions
- Python versions
- PyTorch builds

may behave differently.

The workflow intentionally prioritizes experimentation and reproducibility over strict environment standardization.