from __future__ import annotations
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def dtype_from_name(name: str):
    value = str(name).lower()
    if value == 'bf16':
        return torch.bfloat16
    if value == 'fp16':
        return torch.float16
    return torch.float32


def main():
    parser = argparse.ArgumentParser(description='Merge a LoRA/QLoRA adapter into its base causal LM and save standalone checkpoint.')
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--adapter-path', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--dtype', default='bf16', choices=['bf16', 'fp16', 'fp32'])
    args = parser.parse_args()

    dtype = dtype_from_name(args.dtype)
    print(f'Loading tokenizer from adapter path: {args.adapter_path}')
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True)
    print(f'Loading base model in {args.dtype} precision for merge: {args.base_model}')
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype, device_map='auto', trust_remote_code=True)
    print('Base model loaded.')
    model = PeftModel.from_pretrained(base, args.adapter_path)
    print('Adapter loaded.')
    merged = model.merge_and_unload()
    print('Adapter merged into base model.')

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out), safe_serialization=True)
    tokenizer.save_pretrained(str(out))
    print(f'Merged checkpoint saved to: {out}')


if __name__ == '__main__':
    main()
