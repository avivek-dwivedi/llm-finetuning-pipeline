from __future__ import annotations
import argparse
from pathlib import Path
from datasets import load_dataset
from utils.config import load_config
from utils.jsonl import write_jsonl


def find_image_key(row):
    for k, v in row.items():
        if hasattr(v, 'save'):
            return k
    for key in ['image', 'img', 'jpg']:
        if key in row:
            return key
    return None


def find_caption(row):
    for key in ['caption', 'captions', 'sentence', 'sentences', 'text']:
        if key in row:
            val = row[key]
            if isinstance(val, str):
                return val
            if isinstance(val, list) and val:
                first = val[0]
                if isinstance(first, str):
                    return first
                if isinstance(first, dict):
                    return first.get('raw') or first.get('text') or str(first)
            if isinstance(val, dict):
                return val.get('raw') or val.get('text') or str(val)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/vlm_lora.yaml')
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg['data']
    image_dir = Path(data_cfg['image_dir'])
    image_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(data_cfg['dataset_name'], split=data_cfg.get('split', 'train'))
    max_samples = int(data_cfg.get('max_samples', 100))
    ds = ds.select(range(min(max_samples, len(ds))))

    rows = []
    for i, row in enumerate(ds):
        image_key = find_image_key(row)
        caption = find_caption(row)
        if not image_key or not caption:
            continue
        img = row[image_key]
        if not hasattr(img, 'save'):
            continue
        img_path = image_dir / f'image_{i:05d}.jpg'
        img.convert('RGB').save(img_path, quality=90)
        rows.append({
            'image_path': str(img_path),
            'question': 'Describe this image.',
            'answer': str(caption)
        })
    write_jsonl(data_cfg['output_jsonl'], rows)
    print(f"Saved {len(rows)} VLM image-text samples to {data_cfg['output_jsonl']}")


if __name__ == '__main__':
    main()
