from __future__ import annotations
import argparse
import random
from utils.config import load_config
from utils.jsonl import write_jsonl


def build_synthetic_pairs(max_samples: int):
    # Teaching dataset: small but complete lifecycle.
    # Replace with your real RAG query-positive pairs later.
    base_pairs = [
        ('What is LLM serving?', 'LLM serving means exposing a language model through a controlled API or gateway.'),
        ('Why do we need fallback in LLM serving?', 'Fallback means switching to another provider or model when the primary call fails.'),
        ('What is retry logic?', 'Retry logic repeats temporary failed requests such as timeout or 5xx errors.'),
        ('What is a vector database?', 'A vector database stores embeddings and retrieves semantically similar records.'),
        ('What is hybrid RAG?', 'Hybrid RAG combines dense vector retrieval with sparse keyword retrieval such as BM25.'),
        ('What is LoRA?', 'LoRA fine-tunes small low-rank adapter weights instead of all base model weights.'),
        ('What is QLoRA?', 'QLoRA fine-tunes LoRA adapters while loading the base model in low-bit quantized form.'),
        ('What is DPO?', 'DPO uses chosen and rejected answers to directly optimize model preference behavior.'),
        ('What is VLM fine-tuning?', 'VLM fine-tuning adapts an image-text model using image and text supervision.'),
        ('Why use MLflow?', 'MLflow tracks parameters, metrics, artifacts, and model versions for experiments.'),
    ]
    rows = []
    for i in range(max_samples):
        q, pos = base_pairs[i % len(base_pairs)]
        neg = random.choice([p for _, p in base_pairs if p != pos])
        rows.append({'query': q, 'positive': pos, 'negative': neg})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/embedding_finetune.yaml')
    args = parser.parse_args()
    cfg = load_config(args.config)
    max_samples = int(cfg['data'].get('max_samples', 1000))
    rows = build_synthetic_pairs(max_samples)
    write_jsonl(cfg['data']['output_jsonl'], rows)
    print(f"Saved {len(rows)} embedding pairs to {cfg['data']['output_jsonl']}")


if __name__ == '__main__':
    main()
