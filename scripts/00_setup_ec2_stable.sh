#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "========== EC2 Fine-Tuning V1 | Setup =========="

if [[ ! -f requirements-v1-stable.txt ]]; then
  echo "Missing requirements-v1-stable.txt. Run this script from the repo checkout."
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. Use the AWS Deep Learning AMI GPU PyTorch image or install NVIDIA drivers before running this repo."
  exit 1
fi

echo "========== Verifying NVIDIA Driver =========="
nvidia-smi

sudo apt update -y
sudo apt install -y git git-lfs tmux htop unzip build-essential awscli nvtop

git lfs install || true

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda not found. Use AWS Deep Learning AMI GPU PyTorch or install Miniconda."
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n ec2ftv1 python=3.10 -y
conda activate ec2ftv1
python -m pip install --upgrade pip setuptools wheel

echo "========== Installing PyTorch CUDA Wheels =========="
# PyTorch CUDA install. On AWS DLAMI, torch may already exist; reinstalling keeps the class environment predictable.
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

echo "========== Installing Pinned Python Dependencies =========="
pip install -r requirements-v1-stable.txt

mkdir -p data outputs adapters logs mlruns ray_results
cp -n .env.example .env || true

echo "========== Verifying Environment =========="
python scripts/00_verify_versions.py
python scripts/00_smoke_test_imports.py

echo "Setup complete. Run: conda activate ec2ftv1"
