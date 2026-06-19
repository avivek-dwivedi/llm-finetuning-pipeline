#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "========== Stage 4 | Train VLM Adapter =========="

mkdir -p outputs/vlm logs data/vlm/images

if [[ ! -f configs/vlm_lora.yaml ]]; then
	echo "Missing config: configs/vlm_lora.yaml"
	exit 1
fi

if [[ ! -f data/vlm/train.jsonl ]]; then
	echo "Missing data/vlm/train.jsonl. Run bash scripts/01_pull_all_data.sh first."
	exit 1
fi

python train/train_vlm_lora.py --config configs/vlm_lora.yaml
