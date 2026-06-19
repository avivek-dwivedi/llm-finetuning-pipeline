#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "========== Stage 6 | Merge LLM SFT Adapter =========="

mkdir -p outputs/llm_sft/merged logs

if [[ ! -f outputs/llm_sft/adapter/adapter_config.json ]]; then
  echo "Missing outputs/llm_sft/adapter/adapter_config.json. Run bash scripts/02_train_llm_sft.sh first."
  exit 1
fi

python merge/merge_lora_to_base.py \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter-path ./outputs/llm_sft/adapter \
  --output-dir ./outputs/llm_sft/merged \
  --dtype bf16
