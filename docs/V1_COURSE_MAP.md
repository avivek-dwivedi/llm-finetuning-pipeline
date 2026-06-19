# V1 Course Map

## V1 Scope

V1 covers the practical 75%:

```text
LLM SFT + QLoRA
Embedding fine-tuning
VLM LoRA fine-tuning
DPO preference tuning
Tracking
S3 artifact lifecycle
Optional Ray multi-GPU
```

## V1 End-to-End Lifecycle

```text
HF Dataset
  ↓
Local JSONL cache on EC2
  ↓
Train
  ↓
Save adapter/model
  ↓
Merge adapter where applicable
  ↓
Save merged checkpoint to EC2 disk
  ↓
Upload outputs to S3
  ↓
Download outputs on new EC2
  ↓
Reload
  ↓
Infer
```

## V2 Remaining

```text
DAPT / continued pretraining
Full fine-tuning
Reranker training
Reward model training
PPO / GRPO / ORPO
FSDP / DeepSpeed
Multi-node Ray
Model registry and promotion workflows
```
