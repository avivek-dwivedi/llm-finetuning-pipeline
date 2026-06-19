#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "========== Stage 6b | Merge DPO Adapter =========="

mkdir -p outputs/dpo/merged logs

if [[ ! -f outputs/dpo/adapter/adapter_config.json ]]; then
  echo "Missing outputs/dpo/adapter/adapter_config.json. Run bash scripts/05_train_dpo.sh first."
  exit 1
fi

python merge/merge_lora_to_base.py \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter-path ./outputs/dpo/adapter \
  --output-dir ./outputs/dpo/merged \
  --dtype bf16
