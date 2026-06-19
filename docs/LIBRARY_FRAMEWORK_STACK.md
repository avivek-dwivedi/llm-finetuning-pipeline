# Library and Framework Stack

## Core training stack

| Layer | Tool | Role |
|---|---|---|
| Compute | AWS EC2 G5/G6 | GPU training machine |
| Base framework | PyTorch | GPU tensor training |
| Model API | Hugging Face Transformers | Load LLM/VLM models, Trainer |
| Dataset API | Hugging Face Datasets | Pull and cache datasets from HF Hub |
| Adapter tuning | PEFT | LoRA/QLoRA adapter injection, save, merge |
| Quantization | bitsandbytes | 4-bit QLoRA loading |
| Alignment | TRL | DPO training |
| Embeddings | SentenceTransformers | Embedding model fine-tuning |
| Distributed | Ray Train | Multi-GPU orchestration |
| Tracking | W&B | Live training charts |
| Tracking | MLflow | Local experiment/artifact tracking |
| Storage | S3 + AWS CLI | Artifact upload/download |

## Target stable versions

See:

```text
requirements-v1-stable.txt
```

The original `requirements.txt` is flexible. For class delivery, prefer the stable file so the code does not break from API changes.

## Version testing policy

Before teaching, run:

```bash
python scripts/00_verify_versions.py
python scripts/00_smoke_test_imports.py
```

Then run one tiny lifecycle test:

```bash
# Edit configs/llm_sft_qlora.yaml and set max_samples: 20 first
bash scripts/run_v1_llm_full_lifecycle.sh
```

If the tiny lifecycle passes, increase `max_samples` to 500–1000 for class.
