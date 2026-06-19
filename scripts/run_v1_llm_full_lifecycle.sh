#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "========== Full LLM Lifecycle =========="

mkdir -p outputs/llm_sft logs adapters mlruns data/llm_sft

if [[ ! -f configs/llm_sft_qlora.yaml ]]; then
	echo "Missing config: configs/llm_sft_qlora.yaml"
	exit 1
fi

echo "[1/3] Pulling LLM SFT data"
python pull/pull_llm_sft_data.py --config configs/llm_sft_qlora.yaml
echo "[2/3] Training adapter and merging into base model inline"
python train/train_llm_qlora_sft.py --config configs/llm_sft_qlora.yaml
echo "[3/3] Inference on merged checkpoint (preferred)"
python infer/infer_llm.py --mode merged --model-path ./outputs/llm_sft/merged --prompt "Explain LLM serving in 5 bullet points."
