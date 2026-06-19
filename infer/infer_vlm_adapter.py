from __future__ import annotations
import argparse
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', default='Salesforce/blip-image-captioning-base')
    parser.add_argument('--adapter-path', default='./outputs/vlm/adapter')
    parser.add_argument('--image-path', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=40)
    args = parser.parse_args()

    processor = BlipProcessor.from_pretrained(args.adapter_path)
    base = BlipForConditionalGeneration.from_pretrained(args.base_model, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    if torch.cuda.is_available():
        base = base.to('cuda')
    model = PeftModel.from_pretrained(base, args.adapter_path).eval()

    image = Image.open(args.image_path).convert('RGB')
    inputs = processor(images=image, return_tensors='pt')
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    print(processor.decode(output[0], skip_special_tokens=True))


if __name__ == '__main__':
    main()
