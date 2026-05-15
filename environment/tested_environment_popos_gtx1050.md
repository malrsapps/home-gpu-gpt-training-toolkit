# Tested Environment: Pop!_OS + GTX1050

This file documents the reference environment where the original chained-training workflow was tested successfully.

This is **not a required setup**. Your CUDA, PyTorch, and driver versions may differ depending on your operating system and GPU.

## System

- OS: Pop!_OS 22.04
- Python: 3.10.12
- RAM: 24GB
- GPU: NVIDIA GeForce GTX 1050

## NVIDIA / CUDA

- NVIDIA Driver: 565.77
- CUDA reported by `nvidia-smi`: 12.7

## Python ML Stack

- PyTorch: 2.6.0+cu124
- Torch CUDA: 12.4
- CUDA available in PyTorch: True
- Transformers: 4.50.2
- Tokenizers: 0.21.1

## Notes

This environment worked for the original low-VRAM chained-training workflow.

PyTorch and CUDA compatibility can be sensitive. If you use a different GPU, operating system, or driver version, install PyTorch according to your own CUDA/CPU setup.

Recommended reference:

```bash
python --version
nvidia-smi
python -c "import torch; print('torch:', torch.__version__); print('torch cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
python -c "import transformers, tokenizers; print('transformers:', transformers.__version__); print('tokenizers:', tokenizers.__version__)"