#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "========== Stage 3 | Train Embedding Model =========="

mkdir -p outputs/embedding_model logs

if [[ ! -f configs/embedding_finetune.yaml ]]; then
	echo "Missing config: configs/embedding_finetune.yaml"
	exit 1
fi

if [[ ! -f data/embedding/pairs.jsonl ]]; then
	echo "Missing data/embedding/pairs.jsonl. Run bash scripts/01_pull_all_data.sh first."
	exit 1
fi

python train/train_embedding_model.py --config configs/embedding_finetune.yaml
