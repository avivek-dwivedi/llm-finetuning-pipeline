#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "========== Stage 10 | Ray Multi-GPU LLM Training =========="

mkdir -p outputs/ray_llm_sft logs ray_results

if [[ ! -f configs/ray_llm_sft_lora.yaml ]]; then
	echo "Missing config: configs/ray_llm_sft_lora.yaml"
	exit 1
fi

if [[ ! -f data/llm_sft/train.jsonl ]]; then
	echo "Missing data/llm_sft/train.jsonl. Run bash scripts/01_pull_all_data.sh first."
	exit 1
fi

python ray/train_ray_multigpu_llm_sft.py --config configs/ray_llm_sft_lora.yaml
