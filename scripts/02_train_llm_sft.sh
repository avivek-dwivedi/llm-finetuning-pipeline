#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "========== Stage 2 | Train LLM SFT Adapter =========="

mkdir -p outputs/llm_sft logs adapters

if [[ ! -f configs/llm_sft_qlora.yaml ]]; then
	echo "Missing config: configs/llm_sft_qlora.yaml"
	exit 1
fi

if [[ ! -f data/llm_sft/train.jsonl ]]; then
	echo "Missing data/llm_sft/train.jsonl. Run bash scripts/01_pull_all_data.sh first."
	exit 1
fi

python train/train_llm_qlora_sft.py --config configs/llm_sft_qlora.yaml
