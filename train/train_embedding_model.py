from __future__ import annotations
import argparse
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from utils.config import load_config, ensure_dir
from utils.jsonl import read_jsonl
from utils.tracking import mlflow_log_basic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/embedding_finetune.yaml')
    args = parser.parse_args()
    cfg = load_config(args.config)

    rows = read_jsonl(cfg['data']['output_jsonl'])
    examples = [InputExample(texts=[r['query'], r['positive']]) for r in rows if r.get('query') and r.get('positive')]

    # Embedding fine-tuning is full model fine-tuning — no LoRA adapter, no merge step.
    # SentenceTransformer.fit() updates all weights directly and saves the complete model.
    model = SentenceTransformer(cfg['model']['name'])
    train_loader = DataLoader(examples, shuffle=True, batch_size=int(cfg['training']['batch_size']))
    train_loss = losses.MultipleNegativesRankingLoss(model)

    out_dir = ensure_dir(cfg['project']['output_dir'])
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=int(cfg['training']['epochs']),
        warmup_steps=int(cfg['training'].get('warmup_steps', 50)),
        output_path=str(out_dir),
        show_progress_bar=bool(cfg['training'].get('show_progress_bar', True)),
    )

    mlflow_log_basic(cfg, metrics={'num_pairs': len(examples)}, artifact_dir=str(out_dir))
    print(f'Embedding model saved to: {out_dir}')


if __name__ == '__main__':
    main()
