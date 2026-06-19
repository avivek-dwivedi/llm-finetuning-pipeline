#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "========== Stage 8 | Download Artifacts From S3 =========="

mkdir -p outputs adapters configs logs mlruns

if [[ -f .env ]]; then
	set -a
	source .env
	set +a
fi

: "${S3_BUCKET:?Set S3_BUCKET in .env, e.g. s3://your-bucket/ec2-finetuning-v1}"

echo "Restoring outputs/"
aws s3 sync "$S3_BUCKET/outputs" ./outputs --only-show-errors
echo "Restoring adapters/"
aws s3 sync "$S3_BUCKET/adapters" ./adapters --only-show-errors
echo "Restoring configs/"
aws s3 sync "$S3_BUCKET/configs" ./configs --only-show-errors
echo "Restoring logs/"
aws s3 sync "$S3_BUCKET/logs" ./logs --only-show-errors
echo "Restoring mlruns/"
aws s3 sync "$S3_BUCKET/mlruns" ./mlruns --only-show-errors

echo "Downloaded outputs/, adapters/, configs/, logs/, and mlruns/ from $S3_BUCKET"
