from __future__ import annotations
import argparse

from infer.infer_llm import load_merged_model, make_prompt
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='./outputs/llm_sft/merged')
    parser.add_argument('--prompt', default='Explain retry and fallback in LLM serving.')
    parser.add_argument('--max-new-tokens', type=int, default=200)
    args = parser.parse_args()

    print('Merged-checkpoint inference selected. This is the preferred deployment flow after merge.')
    tokenizer, model = load_merged_model(args.model_path)
    inputs = tokenizer(make_prompt(args.prompt), return_tensors='pt').to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9)
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == '__main__':
    main()
