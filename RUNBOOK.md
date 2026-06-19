# EC2 Fine-Tuning V1 — Runbook

Concepts, models, and lifecycle explanation live in [README.md](README.md).

This file is the operational guide.

It answers these questions first:

1. Where do NVIDIA drivers, `torch`, `ray`, and `mlflow` come from?
2. What exact steps do I follow on single-GPU EC2?
3. What exact steps do I follow on multi-GPU EC2?

---

## Install Truth Table

| Component | Where it comes from | How you verify it |
|---|---|---|
| NVIDIA driver + `nvidia-smi` | AWS Deep Learning AMI GPU PyTorch | `nvidia-smi` |
| Conda | AWS Deep Learning AMI GPU PyTorch | `conda --version` |
| `torch==2.6.0`, `torchvision`, `torchaudio` | `scripts/00_setup_ec2_stable.sh` | `python -c "import torch; print(torch.__version__)"` |
| `ray[train]==2.42.1` | `scripts/00_setup_ec2_stable.sh` via `requirements-v1-stable.txt` | `python -c "import ray; print(ray.__version__)"` |
| `mlflow==2.20.2` | `scripts/00_setup_ec2_stable.sh` via `requirements-v1-stable.txt` | `python -c "import mlflow; print(mlflow.__version__)"` |
| `transformers`, `datasets`, `accelerate`, `peft`, `trl`, `bitsandbytes`, `sentence-transformers` | `scripts/00_setup_ec2_stable.sh` via `requirements-v1-stable.txt` | `python scripts/00_smoke_test_imports.py` |

If `nvidia-smi` is missing, the EC2 image or driver layer is wrong.
If `torch`, `ray`, or `mlflow` are missing, the repo setup script did not complete successfully inside the `ec2ftv1` conda environment.

---

# Part A — Single-GPU Path

Use this for LLM SFT, DPO, VLM, and embedding runs on one GPU.

---

## A1 — Create IAM Role

AWS Console → IAM → Roles → Create role

1. Trusted entity type: `AWS service`
2. Service or use case: `EC2`
3. Permissions: attach `AmazonS3FullAccess`
4. Role name: `ec2-finetuning-s3-role`
5. Create role

If you want tighter access, replace `AmazonS3FullAccess` with a bucket-scoped custom policy.

---

## A2 — Create Security Group

AWS Console → EC2 → Security Groups → Create security group

Suggested values:

| Field | Value |
|---|---|
| Security group name | `ec2-finetuning-v1-sg` |
| Description | `EC2 Fine-Tuning V1 SSH-only access` |
| VPC | same VPC you will launch into |

Inbound rules:

| Type | Protocol | Port | Source |
|---|---|---|---|
| SSH | TCP | 22 | `My IP` |

Do not open `5000`, `6379`, or `8265` publicly. MLflow and Ray access should use SSH tunnels only.

---

## A3 — Launch Single-GPU EC2

AWS Console → EC2 → Launch Instance

| Field | Value |
|---|---|
| Name | `ec2-finetuning-v1` |
| AMI | search `Deep Learning AMI GPU PyTorch`, choose latest Ubuntu |
| Instance type | `g5.xlarge` or `g6.xlarge` |
| Key pair | create or choose an RSA `.pem` key |
| Storage | `200 GB gp3` |

Network settings:

| Field | Value |
|---|---|
| VPC | default VPC or your own |
| Subnet | a public subnet |
| Auto-assign public IP | `Enable` |
| Security group | `ec2-finetuning-v1-sg` |

Public subnet means the subnet route table has `0.0.0.0/0` going to an Internet Gateway. If you launch into a private subnet without NAT, package installs and Hugging Face downloads will fail.

Advanced details:

| Field | Value |
|---|---|
| IAM instance profile | `ec2-finetuning-s3-role` |

Launch the instance.

---

## A4 — SSH In and Verify the Base Machine

From your local machine:

```bash
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@<EC2-PUBLIC-IP>
```

Immediately verify the base image:

```bash
nvidia-smi
conda --version
python3 --version
```

Expected:

- `nvidia-smi` prints the GPU and NVIDIA driver
- `conda` exists
- Python exists

Start a persistent shell:

```bash
tmux new -s train
```

Detach: `Ctrl+B` then `D`

Reattach later:

```bash
tmux attach -t train
```

---

## A5 — Clone Repo and Install Everything

```bash
git clone <repo-url>
cd ec2-finetuning-v1-complete
git checkout v1.1
git rev-parse HEAD
```

Run the setup script:

```bash
bash scripts/00_setup_ec2_stable.sh
```

What this script does:

1. verifies `nvidia-smi`
2. installs `git`, `git-lfs`, `tmux`, `htop`, `nvtop`, `awscli`
3. creates conda env `ec2ftv1`
4. installs `torch==2.6.0`, `torchvision==0.21.0`, `torchaudio==2.6.0`
5. installs `ray[train]==2.42.1`
6. installs `mlflow==2.20.2`
7. installs `transformers`, `datasets`, `accelerate`, `peft`, `trl`, `bitsandbytes`, `sentence-transformers`, and the rest of the pinned stack
8. creates `data/`, `outputs/`, `adapters/`, `logs/`, `mlruns/`, `ray_results/`

Activate the environment after setup and after every future SSH session:

```bash
conda activate ec2ftv1
```

---

## A6 — Configure Credentials

```bash
cp .env.example .env
nano .env
```

Set:

```text
HF_TOKEN=hf_...
WANDB_API_KEY=...
WANDB_PROJECT=ec2-finetuning-v1
MLFLOW_TRACKING_URI=./mlruns
S3_BUCKET=s3://your-bucket/ec2-finetuning-v1
```

Authenticate:

```bash
huggingface-cli login
wandb login
```

If you did not attach the IAM role, also run:

```bash
aws configure
```

---

## A7 — Verify the Installed Stack

```bash
conda activate ec2ftv1
python scripts/00_verify_versions.py
python scripts/00_smoke_test_imports.py
```

Minimum expected signals:

- `cuda available: True`
- `gpu count: 1`
- `torch version: 2.6.0+cu124`
- `ray version: 2.42.1`
- `mlflow version: 2.20.2`
- smoke test prints all imports OK

If any import is missing, rerun:

```bash
pip install -r requirements-v1-stable.txt
```

inside the `ec2ftv1` environment.

---

## A8 — Pull Datasets

```bash
bash scripts/01_pull_all_data.sh
```

Verify the local artifacts:

```bash
wc -l data/llm_sft/train.jsonl
wc -l data/dpo/train.jsonl
ls data/vlm/images/ | head -5
```

Default configs use `max_samples: 20` for the smoke test.

---

## A9 — Run the Full LLM Lifecycle

```bash
bash scripts/run_v1_llm_full_lifecycle.sh
```

This runs:

1. pull LLM SFT data
2. train LoRA adapter
3. merge adapter inline into a standalone checkpoint
4. run inference on the merged checkpoint

Check outputs:

```bash
ls outputs/llm_sft/adapter/
ls outputs/llm_sft/merged/
cat outputs/llm_sft/training_summary.json
```

---

## A10 — Run Other Tracks

LLM SFT only:

```bash
bash scripts/02_train_llm_sft.sh
```

Embedding full fine-tuning:

```bash
bash scripts/03_train_embedding.sh
```

VLM LoRA:

```bash
bash scripts/04_train_vlm.sh
```

DPO:

```bash
bash scripts/05_train_dpo.sh
```

---

## A11 — Inference

Preferred LLM inference flow:

```bash
python infer/infer_llm.py \
  --mode auto \
  --model-path ./outputs/llm_sft/merged \
  --adapter-path ./outputs/llm_sft/adapter \
  --prompt "Explain LoRA fine-tuning in simple terms."
```

Adapter-only reload proof:

```bash
python infer/infer_llm_adapter.py \
  --adapter-path ./outputs/llm_sft/adapter \
  --prompt "What is QLoRA?"
```

Embedding retrieval:

```bash
python infer/infer_embedding_retrieval.py \
  --query "What is fallback in LLM serving?" \
  --model-path ./outputs/embedding_model \
  --data-path ./data/embedding/pairs.jsonl
```

VLM inference:

```bash
IMG=$(ls data/vlm/images/*.jpg | head -n 1)
python infer/infer_vlm_adapter.py --image-path "$IMG"
```

All checks:

```bash
bash scripts/09_infer_all.sh
```

---

## A12 — Start MLflow Correctly

`mlflow` is not a system command outside the conda env. Always do this first:

```bash
conda activate ec2ftv1
mlflow ui --host 0.0.0.0 --port 5000
```

From your local machine in a separate terminal:

```bash
ssh -i your-key.pem -L 5000:localhost:5000 ubuntu@<EC2-PUBLIC-IP>
```

Then open `http://localhost:5000` locally.

---

## A13 — Upload to S3 and Restore on Another EC2

Check S3 access first:

```bash
aws sts get-caller-identity
aws s3 ls $S3_BUCKET
```

Upload:

```bash
bash scripts/07_upload_outputs_to_s3.sh
```

On EC2-B, after you repeat A4 through A7:

```bash
bash scripts/08_download_outputs_from_s3.sh
bash scripts/09_infer_all.sh
```

---

## A14 — Scale Up After Smoke Test

Once the 20-sample flow works end to end, edit YAML configs and increase:

```yaml
data:
  max_samples: 500
```

or

```yaml
data:
  max_samples: 1000
```

If you hit OOM, reduce:

```yaml
training:
  max_seq_length: 256
  per_device_train_batch_size: 1
lora:
  r: 8
```

---

# Part B — Multi-GPU Path

Use this path for Ray training on a 4-GPU box.

---

## B1 — Create the Multi-GPU EC2 Instance

Create IAM role and security group exactly as in A1 and A2.

Launch Instance values:

| Field | Value |
|---|---|
| Name | `ec2-finetuning-v1-multigpu` |
| AMI | `Deep Learning AMI GPU PyTorch` Ubuntu |
| Instance type | `g5.12xlarge` or `g6.12xlarge` |
| Key pair | same `.pem` key is fine |
| Storage | `300 GB gp3` |

Network settings:

| Field | Value |
|---|---|
| VPC | same VPC as your other EC2s or default |
| Subnet | public subnet with Internet Gateway route |
| Auto-assign public IP | `Enable` |
| Security group | `ec2-finetuning-v1-sg` or another SSH-only group |

Advanced details:

| Field | Value |
|---|---|
| IAM instance profile | `ec2-finetuning-s3-role` |

Do not open Ray ports publicly. Use SSH tunnels.

---

## B2 — SSH In and Verify All GPUs

```bash
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@<MULTIGPU-EC2-IP>
```

Check hardware:

```bash
nvidia-smi
```

Expected: all GPUs should appear, for example GPU 0 through GPU 3.

Optional topology check:

```bash
nvidia-smi topo -m
```

Start tmux:

```bash
tmux new -s raytrain
```

---

## B3 — Install the Same Software Stack

Same commands as single-GPU:

```bash
git clone <repo-url>
cd ec2-finetuning-v1-complete
git checkout v1.1
bash scripts/00_setup_ec2_stable.sh
conda activate ec2ftv1
```

This installs `torch`, `ray`, and `mlflow` the same way as the single-GPU path.

---

## B4 — Configure Credentials and Verify the Environment

Same as A6 and A7:

```bash
cp .env.example .env
nano .env
huggingface-cli login
wandb login
python scripts/00_verify_versions.py
python scripts/00_smoke_test_imports.py
```

Important multi-GPU verification: `gpu count` should match your actual box, for example `4`.

---

## B5 — Pull Data Before Ray Training

```bash
bash scripts/01_pull_all_data.sh
```

Ray reuses the local JSONL dataset written by the pull step.

---

## B6 — Set One Worker per GPU

Edit the Ray config:

```bash
nano configs/ray_llm_sft_lora.yaml
```

For a 4-GPU machine, set:

```yaml
ray:
  num_workers: 4
  use_gpu: true
  cpus_per_worker: 4
```

Rule: `num_workers = number of GPUs you want to use`.

---

## B7 — Run Ray Multi-GPU Training

```bash
bash scripts/10_train_ray_multigpu_llm.sh
```

Important detail: you do not need to manually install or start a separate Ray service. `ray` is installed by the setup script, and [ray/train_ray_multigpu_llm_sft.py](ray/train_ray_multigpu_llm_sft.py) calls `ray.init()` locally on the EC2 instance.

That means the actual multi-GPU sequence is:

1. install `ray` in `ec2ftv1`
2. verify all GPUs with `nvidia-smi`
3. set `num_workers`
4. run `bash scripts/10_train_ray_multigpu_llm.sh`

Output path:

```text
outputs/ray_llm_sft/adapter/
```

---

## B8 — Open the Ray Dashboard

From your local machine:

```bash
ssh -i your-key.pem -L 8265:localhost:8265 ubuntu@<MULTIGPU-EC2-IP>
```

Then open `http://localhost:8265` locally.

---

## B9 — Run the Other Tracks on the Same Box if Needed

The multi-GPU instance can also run the standard tracks with the same commands:

```bash
bash scripts/run_v1_llm_full_lifecycle.sh
bash scripts/03_train_embedding.sh
bash scripts/04_train_vlm.sh
bash scripts/05_train_dpo.sh
```

Ray only runs when you explicitly invoke `scripts/10_train_ray_multigpu_llm.sh`.

---

## Common Fixes

| Problem | Fix |
|---|---|
| `nvidia-smi: command not found` | Wrong AMI or missing NVIDIA drivers. Use the AWS Deep Learning AMI GPU PyTorch image. |
| `conda: command not found` | Wrong AMI or broken shell init. Use the AWS Deep Learning AMI GPU PyTorch image. |
| `mlflow: command not found` | Run `conda activate ec2ftv1` first. `mlflow` is installed inside that env. |
| `ray: import error` | Re-run `bash scripts/00_setup_ec2_stable.sh` or `pip install -r requirements-v1-stable.txt` inside `ec2ftv1`. |
| `torch.cuda.is_available() = False` | Driver / CUDA mismatch or wrong EC2 AMI. Check `nvidia-smi` first. |
| `S3_BUCKET not set` | Fill `.env` with `S3_BUCKET=...`. |
| W&B login prompt during training | Run `wandb login` first or set `report_to: []` in the YAML. |
| OOM | Reduce `max_seq_length`, `per_device_train_batch_size`, or `lora.r`. |
| Ray workers not starting | Verify `nvidia-smi` shows all GPUs and `num_workers` matches that count. |