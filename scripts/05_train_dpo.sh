#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "========== Stage 5 | Train DPO Adapter =========="

mkdir -p outputs/dpo logs adapters

if [[ ! -f configs/dpo_qlora.yaml ]]; then
	echo "Missing config: configs/dpo_qlora.yaml"
	exit 1
fi

if [[ ! -f data/dpo/train.jsonl ]]; then
	echo "Missing data/dpo/train.jsonl. Run bash scripts/01_pull_all_data.sh first."
	exit 1
fi

python train/train_dpo_qlora.py --config configs/dpo_qlora.yaml
