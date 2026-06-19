# EC2 Fine-Tuning V1 — Step-by-Step Guide

This V1 course proves adaptability and completion, not final accuracy.

Goal:

```text
Hugging Face dataset
  ↓
pull/cache on EC2
  ↓
train adapter
  ↓
save adapter on EC2 disk
  ↓
merge adapter into base checkpoint
  ↓
save merged checkpoint on EC2 disk
  ↓
upload to S3
  ↓
download on another EC2
  ↓
reload and infer
```

---

## 1. Launch EC2

Recommended for V1:

```text
Single GPU: g5.xlarge or g6.xlarge
Multi-GPU Ray: g5.12xlarge / g6.12xlarge or above
AMI: AWS Deep Learning AMI GPU PyTorch
Disk: 200 GB gp3 EBS minimum
Security: SSH only; use SSH tunnel for MLflow/Ray dashboard
```

---

## 2. Connect

```bash
ssh -i your-key.pem ubuntu@EC2_PUBLIC_IP
```

Use tmux:

```bash
tmux new -s ftv1
```

---

## 3. Upload or clone project

```bash
unzip ec2-finetuning-v1-complete.zip
cd ec2-finetuning-v1-complete
```

---

## 4. Stable setup

```bash
bash scripts/00_setup_ec2_stable.sh
conda activate ec2ftv1
```

This runs:

```text
package install
version print
GPU check
import/API smoke test
```

For a quick environment check later:

```bash
python scripts/00_verify_versions.py
python scripts/00_smoke_test_imports.py
```

---

## 5. Login

```bash
huggingface-cli login
wandb login
aws configure
```

For MLflow local file tracking, no login is needed.

---

## 6. Pull all datasets

```bash
bash scripts/01_pull_all_data.sh
```

Outputs:

```text
data/llm_sft/train.jsonl
data/embedding/pairs.jsonl
data/vlm/train.jsonl
data/vlm/images/
data/dpo/train.jsonl
```

---

## 7. LLM SFT lifecycle

```bash
bash scripts/run_v1_llm_full_lifecycle.sh
```

This runs:

```text
pull LLM data
train QLoRA adapter
merge adapter into base model
infer from adapter
infer from merged checkpoint
```

Individual commands:

```bash
bash scripts/02_train_llm_sft.sh
bash scripts/06_merge_llm_sft_adapter.sh
python infer/infer_llm_adapter.py --base-model Qwen/Qwen2.5-0.5B-Instruct --adapter-path ./outputs/llm_sft/adapter
python infer/infer_llm_merged.py --model-path ./outputs/llm_sft/merged_model
```

---

## 8. Embedding model fine-tuning

```bash
bash scripts/03_train_embedding.sh
python infer/infer_embedding_retrieval.py --model-path ./outputs/embedding_model --data ./data/embedding/pairs.jsonl
```

Output:

```text
outputs/embedding_model/
```

---

## 9. VLM LoRA lifecycle

```bash
bash scripts/04_train_vlm.sh
python infer/infer_vlm_adapter.py --adapter-path ./outputs/vlm/adapter --image-path ./data/vlm/images/sample_0000.jpg
```

V1 uses BLIP image captioning because it is stable for teaching image-text adapter lifecycle. V2 can upgrade to Qwen2.5-VL or SmolVLM visual instruction tuning.

---

## 10. DPO lifecycle

```bash
bash scripts/05_train_dpo.sh
bash scripts/06b_merge_dpo_adapter.sh
```

Output:

```text
outputs/dpo/adapter
outputs/dpo/merged_model
```

---

## 11. S3 artifact lifecycle

Set bucket:

```bash
export S3_BUCKET=s3://your-bucket-name/ec2-finetuning-v1
```

Upload:

```bash
bash scripts/07_upload_outputs_to_s3.sh
```

Download on another EC2:

```bash
bash scripts/08_download_outputs_from_s3.sh
```

Infer again:

```bash
bash scripts/09_infer_all.sh
```

---

## 12. Ray multi-GPU SFT

Use only on multi-GPU EC2.

Edit:

```text
configs/ray_llm_sft_lora.yaml
```

Set workers equal to GPU count:

```yaml
ray:
  num_workers: 4
```

Run:

```bash
bash scripts/10_train_ray_multigpu_llm.sh
```

---

## 13. What counts as V1 success?

V1 success does not mean high accuracy. V1 success means:

```text
[ ] data pulled from Hugging Face
[ ] training runs complete
[ ] adapter saved
[ ] adapter can be reloaded
[ ] adapter can be merged
[ ] merged checkpoint can run inference
[ ] outputs can be uploaded to S3
[ ] outputs can be downloaded on another EC2
[ ] downloaded outputs can run inference
[ ] W&B/MLflow logs exist
```

Accuracy comes later from better data, more epochs, better eval, and hyperparameter tuning.
