#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "========== Stage 9 | Inference Checks =========="

if [[ ! -d outputs/llm_sft/adapter ]]; then
  echo "Missing outputs/llm_sft/adapter. Run the LLM lifecycle first."
  exit 1
fi

if [[ ! -d outputs/llm_sft/merged ]]; then
  echo "Missing outputs/llm_sft/merged. Run bash scripts/06_merge_llm_sft_adapter.sh first."
  exit 1
fi

python infer/infer_llm.py --mode auto --adapter-path ./outputs/llm_sft/adapter --model-path ./outputs/llm_sft/merged --prompt "Explain retry and fallback in LLM serving."
python infer/infer_llm_adapter.py --adapter-path ./outputs/llm_sft/adapter --prompt "Explain LoRA fine-tuning in simple words."
if [[ -d outputs/embedding_model && -f data/embedding/pairs.jsonl ]]; then
  python infer/infer_embedding_retrieval.py --query "What is fallback in LLM serving?"
else
  echo "Skipping embedding inference because outputs/embedding_model or data/embedding/pairs.jsonl is missing."
fi
if compgen -G "./data/vlm/images/*.jpg" > /dev/null; then
  IMG=$(ls ./data/vlm/images/*.jpg | head -n 1)
  python infer/infer_vlm_adapter.py --image-path "$IMG"
else
  echo "Skipping VLM inference because no JPG image was found under data/vlm/images/."
fi
