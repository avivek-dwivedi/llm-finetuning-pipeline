from __future__ import annotations
import argparse
from datasets import load_dataset
from utils.config import load_config
from utils.jsonl import write_jsonl


def to_text(x):
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        # Handle chat message list formats.
        parts = []
        for item in x:
            if isinstance(item, dict):
                role = item.get('role', '')
                content = item.get('content', '')
                parts.append(f"{role}: {content}" if role else str(content))
            else:
                parts.append(str(item))
        return '\n'.join(parts)
    return str(x)


def normalize_row(row):
    prompt = row.get('prompt') or row.get('instruction') or row.get('question') or ''
    chosen = row.get('chosen') or row.get('chosen_response') or row.get('response_chosen') or ''
    rejected = row.get('rejected') or row.get('rejected_response') or row.get('response_rejected') or ''
    return {'prompt': to_text(prompt), 'chosen': to_text(chosen), 'rejected': to_text(rejected)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/dpo_qlora.yaml')
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg['data']
    ds = load_dataset(data_cfg['dataset_name'], split=data_cfg.get('split', 'train'))
    max_samples = int(data_cfg.get('max_samples', 0) or 0)
    if max_samples:
        ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))
    rows = [normalize_row(row) for row in ds]
    rows = [r for r in rows if r['prompt'] and r['chosen'] and r['rejected']]
    write_jsonl(data_cfg['output_jsonl'], rows)
    print(f"Saved {len(rows)} DPO preference samples to {data_cfg['output_jsonl']}")


if __name__ == '__main__':
    main()
