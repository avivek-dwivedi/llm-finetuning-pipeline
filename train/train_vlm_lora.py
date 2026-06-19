from __future__ import annotations
import argparse
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import BlipProcessor, BlipForConditionalGeneration, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from utils.config import load_config, ensure_dir
from utils.jsonl import read_jsonl
from utils.tracking import setup_wandb, mlflow_log_basic


class ImageCaptionDataset(Dataset):
    def __init__(self, rows, processor):
        self.rows = rows
        self.processor = processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = Image.open(row['image_path']).convert('RGB')
        caption = row['answer']
        enc = self.processor(images=image, text=caption, padding='max_length', truncation=True, max_length=64, return_tensors='pt')
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels'] = item['input_ids'].clone()
        return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/vlm_lora.yaml')
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_wandb(cfg)

    out_dir = ensure_dir(cfg['project']['output_dir'])
    adapter_dir = ensure_dir(Path(out_dir) / 'adapter')
    model_name = cfg['model']['name']

    processor = BlipProcessor.from_pretrained(model_name)
    dtype = torch.bfloat16 if cfg['model'].get('torch_dtype', 'bf16') == 'bf16' else torch.float16
    model = BlipForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype if torch.cuda.is_available() else torch.float32)
    if torch.cuda.is_available():
        model = model.to('cuda')

    lcfg = cfg['lora']
    # BLIP is seq2seq/image-captioning, PEFT task type SEQ_2_SEQ_LM is the closest match.
    peft_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=int(lcfg['r']),
        lora_alpha=int(lcfg['alpha']),
        lora_dropout=float(lcfg['dropout']),
        target_modules=lcfg['target_modules'],
        bias='none',
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    rows = read_jsonl(cfg['data']['output_jsonl'])
    train_ds = ImageCaptionDataset(rows, processor)
    tcfg = cfg['training']
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        run_name=cfg['project']['run_name'],
        num_train_epochs=float(tcfg['epochs']),
        per_device_train_batch_size=int(tcfg['per_device_train_batch_size']),
        gradient_accumulation_steps=int(tcfg['gradient_accumulation_steps']),
        learning_rate=float(tcfg['learning_rate']),
        logging_steps=int(tcfg['logging_steps']),
        save_steps=int(tcfg['save_steps']),
        bf16=bool(tcfg.get('bf16', True)),
        fp16=bool(tcfg.get('fp16', False)),
        report_to=tcfg.get('report_to', []),
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds)
    result = trainer.train()
    # --- Step 1: Save adapter ---
    trainer.save_model(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    print(f'VLM adapter saved to: {adapter_dir}')

    # --- Step 2: Merge adapter into base model inline ---
    # Reload base in bf16/fp32 (not via the quantized path) so merge_and_unload() works correctly.
    merge_dtype = dtype if torch.cuda.is_available() else torch.float32
    merged_dir = ensure_dir(Path(out_dir) / 'merged')
    print(f'Reloading BLIP base model in {merge_dtype} for merge.')
    base_for_merge = BlipForConditionalGeneration.from_pretrained(model_name, torch_dtype=merge_dtype)
    if torch.cuda.is_available():
        base_for_merge = base_for_merge.to('cuda')
    peft_model = PeftModel.from_pretrained(base_for_merge, str(adapter_dir))
    print('Merging adapter into BLIP base model.')
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
    processor.save_pretrained(str(merged_dir))
    print(f'VLM merged checkpoint saved to: {merged_dir}')

    mlflow_log_basic(cfg, metrics=result.metrics if hasattr(result, 'metrics') else {}, artifact_dir=str(merged_dir))


if __name__ == '__main__':
    main()
