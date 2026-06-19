# EC2 Fine-Tuning V1

A class-ready fine-tuning pipeline that covers the full industry lifecycle on AWS EC2 — from pulling a Hugging Face dataset to inferring from a merged checkpoint on a second machine.

The goal is **lifecycle completion**, not benchmark accuracy. Every step is meant to be teachable, inspectable, and repeatable.

> **Want to run it right now?** → See [RUNBOOK.md](RUNBOOK.md)

---

## What This Covers

| Track | Technique | What it teaches |
|---|---|---|
| LLM SFT | QLoRA (4-bit base + LoRA adapter) | Instruction fine-tuning, adapter save, inline merge |
| DPO | QLoRA + DPOTrainer | Preference alignment, same adapter/merge lifecycle |
| VLM | LoRA on BLIP | Image-caption fine-tuning, multimodal adapter workflow |
| Embedding | Full model fine-tuning | SentenceTransformer training, retrieval demo |
| Ray multi-GPU | LoRA + bf16 distributed | Multi-worker training, one GPU per worker |

---

## Models and Data Sources

| Track | Model (Hugging Face Hub) | Dataset (Hugging Face Hub) |
|---|---|---|
| LLM SFT | `Qwen/Qwen2.5-0.5B-Instruct` | `yahma/alpaca-cleaned` |
| DPO | `Qwen/Qwen2.5-0.5B-Instruct` | `trl-lib/ultrafeedback_binarized` |
| Ray multi-GPU | `Qwen/Qwen2.5-0.5B-Instruct` | reuses LLM SFT data |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` | local JSONL pulled from HF |
| VLM | `Salesforce/blip-image-captioning-base` | `nlphuji/flickr30k` |

All models are deliberately small — chosen for classroom GPU budgets, not leaderboard performance.

---

## Lifecycle Flow

```
Hugging Face Dataset
        ↓
  pull/ scripts → JSONL files in data/
        ↓
  train/ scripts
    ├─ load base model (4-bit for QLoRA tracks)
    ├─ inject LoRA adapter via PEFT
    ├─ train adapter weights only
    ├─ save adapter → outputs/<track>/adapter/
    ├─ reload base in bf16 for merge
    ├─ merge_and_unload()
    └─ save merged checkpoint → outputs/<track>/merged/
        ↓
  infer/ scripts
    ├─ infer_llm.py --mode auto   ← prefers merged, falls back to adapter
    ├─ infer_llm_merged.py        ← merged checkpoint only
    ├─ infer_llm_adapter.py       ← base + adapter (proves PEFT reload)
    ├─ infer_embedding_retrieval.py
    └─ infer_vlm_adapter.py
        ↓
  scripts/07_upload_outputs_to_s3.sh
        ↓
       S3
        ↓
  scripts/08_download_outputs_from_s3.sh  (on EC2-B)
        ↓
  scripts/09_infer_all.sh  (on EC2-B) ← proves full portability
```

### Why merge is done inside the training script

QLoRA training uses a 4-bit quantized base model for GPU efficiency. A quantized model cannot be directly saved as a standalone usable checkpoint — the quantization is a training-time trick. After training, the script reloads the base model in bf16/fp16 (full precision, no quantization), loads the LoRA adapter on top, and calls `merge_and_unload()`. This produces a clean, self-contained checkpoint that can be loaded anywhere without knowing anything about PEFT or bitsandbytes.

### Embedding is different

The embedding track uses `SentenceTransformer.fit()` — this is standard full-weight fine-tuning, not LoRA. There is no adapter, no merge step. The output directory already contains the complete fine-tuned model.

---

## Repository Layout

```
configs/          YAML configs — one per training track
pull/             Data pull scripts (HF → JSONL)
train/            Training scripts (adapter + inline merge)
merge/            Standalone re-merge script (optional)
infer/            Inference scripts
ray/              Ray multi-GPU training
scripts/          Shell lifecycle scripts
utils/            Shared helpers (config loader, JSONL, tracking)
docs/             WATCHOUTS.md, QUANTIZATION_AND_MERGE.md
data/             Pulled datasets (not committed)
outputs/          Training outputs (not committed)
adapters/         Saved adapters (not committed)
logs/             Run logs
mlruns/           MLflow tracking store (not committed)
```

---

## Tracking

- **W&B** — project name defaults to `ec2-finetuning-v1`. Disable per-run with `report_to: []` in the YAML.
- **MLflow** — tracking URI defaults to `./mlruns`. View with `mlflow ui` over an SSH tunnel.

---

## Watchouts

See [docs/WATCHOUTS.md](docs/WATCHOUTS.md). Short version:

- Start with `max_samples: 20`. Scale up only after a full lifecycle passes.
- Do not commit models, adapters, or MLflow runs to Git. Use S3.
- Do not open Ray or MLflow ports publicly. Use SSH tunnels.
- Merge always uses bf16/fp16 base, not the 4-bit training load path.

