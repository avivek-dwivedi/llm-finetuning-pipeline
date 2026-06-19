#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "========== Stage 7 | Upload Artifacts to S3 =========="

mkdir -p outputs adapters configs logs mlruns

if [[ -f .env ]]; then
	set -a
	source .env
	set +a
fi

: "${S3_BUCKET:?Set S3_BUCKET in .env, e.g. s3://your-bucket/ec2-finetuning-v1}"

echo "Syncing outputs/"
aws s3 sync ./outputs "$S3_BUCKET/outputs" --only-show-errors
echo "Syncing adapters/"
aws s3 sync ./adapters "$S3_BUCKET/adapters" --only-show-errors
echo "Syncing configs/"
aws s3 sync ./configs "$S3_BUCKET/configs" --only-show-errors
echo "Syncing logs/"
aws s3 sync ./logs "$S3_BUCKET/logs" --only-show-errors
echo "Syncing mlruns/"
aws s3 sync ./mlruns "$S3_BUCKET/mlruns" --only-show-errors

echo "Uploaded outputs/, adapters/, configs/, logs/, and mlruns/ to $S3_BUCKET"
