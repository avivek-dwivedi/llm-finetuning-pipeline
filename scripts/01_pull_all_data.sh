#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "========== Stage 1 | Pull All Data =========="

mkdir -p data/llm_sft data/embedding data/dpo data/vlm data/vlm/images logs outputs adapters mlruns

for config_path in configs/llm_sft_qlora.yaml configs/embedding_finetune.yaml configs/dpo_qlora.yaml configs/vlm_lora.yaml; do
	if [[ ! -f "$config_path" ]]; then
		echo "Missing config: $config_path"
		exit 1
	fi
done

echo "[1/4] Pulling LLM SFT data"
python pull/pull_llm_sft_data.py --config configs/llm_sft_qlora.yaml
echo "[2/4] Pulling embedding data"
python pull/pull_embedding_data.py --config configs/embedding_finetune.yaml
echo "[3/4] Pulling DPO data"
python pull/pull_dpo_data.py --config configs/dpo_qlora.yaml
echo "[4/4] Pulling VLM data"
python pull/pull_vlm_data.py --config configs/vlm_lora.yaml
