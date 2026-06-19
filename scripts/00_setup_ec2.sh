#!/usr/bin/env bash
set -euo pipefail

sudo apt update -y
sudo apt install -y git git-lfs tmux htop unzip build-essential awscli

git lfs install || true

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda not found. Use AWS Deep Learning AMI GPU PyTorch or install Miniconda."
  exit 1
fi

conda create -n ec2ftv1 python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ec2ftv1
python -m pip install --upgrade pip
pip install -r requirements.txt

mkdir -p data outputs adapters logs mlruns
cp -n .env.example .env || true

echo "Setup complete. Run: conda activate ec2ftv1"
