from __future__ import annotations
import argparse

from infer.infer_llm import load_adapter_model, make_prompt
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--adapter-path', default='./outputs/llm_sft/adapter')
    parser.add_argument('--prompt', default='Explain LoRA fine-tuning in simple words.')
    parser.add_argument('--max-new-tokens', type=int, default=200)
    args = parser.parse_args()

    print('Adapter-mode inference selected. This proves reload of base model + adapter without merging.')
    tokenizer, model = load_adapter_model(args.base_model, args.adapter_path)
    inputs = tokenizer(make_prompt(args.prompt), return_tensors='pt').to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9)
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == '__main__':
    main()
