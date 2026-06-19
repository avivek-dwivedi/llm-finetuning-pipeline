from __future__ import annotations
import importlib
import subprocess
import sys

def package_version(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
        return str(getattr(module, '__version__', 'installed'))
    except Exception as exc:
        return f'NOT INSTALLED ({exc})'


print(f'python version: {sys.version.split()[0]}')

try:
    import torch
    print(f'torch version: {torch.__version__}')
    print(f'cuda available: {torch.cuda.is_available()}')
    print(f'gpu count: {torch.cuda.device_count()}')
except Exception as exc:
    print(f'torch version: NOT INSTALLED ({exc})')
    print('cuda available: unknown')
    print('gpu count: unknown')

print(f'transformers version: {package_version("transformers")}')
print(f'datasets version: {package_version("datasets")}')
print(f'accelerate version: {package_version("accelerate")}')
print(f'peft version: {package_version("peft")}')
print(f'trl version: {package_version("trl")}')
print(f'bitsandbytes version: {package_version("bitsandbytes")}')
print(f'ray version: {package_version("ray")}')
print(f'wandb version: {package_version("wandb")}')
print(f'mlflow version: {package_version("mlflow")}')

print('\n=== nvidia-smi ===')
try:
    print(subprocess.check_output(['nvidia-smi'], text=True))
except Exception as exc:
    print(f'nvidia-smi failed: {exc}')
