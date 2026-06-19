from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def make_prompt(text: str) -> str:
    return f"### Instruction:\n{text}\n\n### Response:\n"


def preferred_dtype():
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def load_adapter_model(base_model: str, adapter_path: str):
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=preferred_dtype(),
        device_map='auto',
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path).eval()
    return tokenizer, model


def load_merged_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=preferred_dtype(),
        device_map='auto',
        trust_remote_code=True,
    ).eval()
    return tokenizer, model


def main():
    parser = argparse.ArgumentParser(description='Unified LLM inference. Auto mode prefers the merged checkpoint when it exists.')
    parser.add_argument('--mode', default='auto', choices=['auto', 'merged', 'adapter'])
    parser.add_argument('--base-model', default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--adapter-path', default='./outputs/llm_sft/adapter')
    parser.add_argument('--model-path', default='./outputs/llm_sft/merged')
    parser.add_argument('--prompt', default='Explain LLM serving in 5 bullet points.')
    parser.add_argument('--max-new-tokens', type=int, default=200)
    args = parser.parse_args()

    adapter_path = Path(args.adapter_path)
    model_path = Path(args.model_path)
    mode = args.mode

    if mode == 'auto':
        mode = 'merged' if (model_path / 'config.json').exists() else 'adapter'

    if mode == 'merged':
        if not (model_path / 'config.json').exists():
            raise FileNotFoundError(f'Merged checkpoint not found at {model_path}. Run merge/merge_lora_to_base.py first.')
        print(f'Using merged checkpoint: {model_path}')
        tokenizer, model = load_merged_model(str(model_path))
    else:
        if not (adapter_path / 'adapter_config.json').exists():
            raise FileNotFoundError(f'Adapter checkpoint not found at {adapter_path}. Run training first.')
        print(f'Using base model + adapter: base={args.base_model}, adapter={adapter_path}')
        tokenizer, model = load_adapter_model(args.base_model, str(adapter_path))

    inputs = tokenizer(make_prompt(args.prompt), return_tensors='pt').to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == '__main__':
    main()