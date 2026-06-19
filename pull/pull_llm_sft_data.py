from __future__ import annotations
import argparse
from datasets import load_dataset
from utils.config import load_config
from utils.jsonl import write_jsonl


def format_alpaca(row):
    instruction = str(row.get('instruction', '')).strip()
    inp = str(row.get('input', '') or '').strip()
    output = str(row.get('output', '')).strip()
    if inp:
        text = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n{output}"
    else:
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    return {'text': text}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/llm_sft_qlora.yaml')
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg['data']

    ds = load_dataset(data_cfg['dataset_name'], split=data_cfg.get('split', 'train'))
    max_samples = int(data_cfg.get('max_samples', 0) or 0)
    if max_samples:
        ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))

    rows = [format_alpaca(row) for row in ds]
    write_jsonl(data_cfg['output_jsonl'], rows)
    print(f"Saved {len(rows)} LLM SFT samples to {data_cfg['output_jsonl']}")


if __name__ == '__main__':
    main()
