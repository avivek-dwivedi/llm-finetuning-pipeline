from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, TaskType
from utils.config import load_config, ensure_dir
from utils.tracking import setup_wandb, mlflow_log_basic


def dtype_from_name(name: str):
    name = str(name).lower()
    if name in {'bf16', 'bfloat16'}:
        return torch.bfloat16
    if name in {'fp16', 'float16'}:
        return torch.float16
    return torch.float32


def write_training_summary(summary_path: Path, cfg: dict, sample_count: int, metrics: dict, merged_dir: str) -> None:
    payload = {
        'summary': 'Adapter trained and merged into base model in a single run.',
        'base_model': cfg['model']['name'],
        'dataset_source': cfg['data'].get('dataset_name', cfg['data'].get('output_jsonl')),
        'sample_count': sample_count,
        'use_4bit_base_model': bool(cfg['model'].get('use_4bit', True)),
        'adapter_output_dir': str(Path(cfg['project']['output_dir']) / 'adapter'),
        'merged_output_dir': merged_dir,
        'metrics': {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/llm_sft_qlora.yaml')
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_wandb(cfg)

    out_dir = ensure_dir(cfg['project']['output_dir'])
    adapter_dir = ensure_dir(Path(out_dir) / 'adapter')
    used_cfg = Path(out_dir) / 'used_config.yaml'
    used_cfg.write_text(Path(args.config).read_text(), encoding='utf-8')
    log_summary_path = Path(out_dir) / 'training_summary.json'

    # This script:
    #   1. Loads the base model in 4-bit with bitsandbytes (when use_4bit: true)
    #   2. Injects a PEFT LoRA adapter
    #   3. Trains adapter weights only
    #   4. Saves the adapter
    #   5. Reloads the base model in bf16/fp16 and merges the adapter inline
    #   6. Saves the merged standalone checkpoint

    model_name = cfg['model']['name']
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=cfg['model'].get('trust_remote_code', True))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = dtype_from_name(cfg['model'].get('torch_dtype', 'bf16'))
    quant_config = None
    model_kwargs = {'trust_remote_code': cfg['model'].get('trust_remote_code', True), 'torch_dtype': torch_dtype}
    if cfg['model'].get('use_4bit', True):
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch_dtype, bnb_4bit_use_double_quant=True)
        model_kwargs['quantization_config'] = quant_config
        model_kwargs['device_map'] = 'auto'
        print('Loading base model in 4-bit NF4 for adapter training.')
    else:
        print('Loading base model without 4-bit quantization for adapter training.')

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.config.use_cache = False
    if cfg['model'].get('use_4bit', True):
        model = prepare_model_for_kbit_training(model)

    lora_cfg = cfg['lora']
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_cfg['r']),
        lora_alpha=int(lora_cfg['alpha']),
        lora_dropout=float(lora_cfg['dropout']),
        target_modules=lora_cfg['target_modules'],
        bias='none',
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    print('LoRA adapter injected with PEFT. Training will update adapter weights only.')

    data_path = cfg['data']['output_jsonl']
    ds = load_dataset('json', data_files=data_path, split='train')
    max_len = int(cfg['training']['max_seq_length'])

    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True, max_length=max_len, padding=False)

    tokenized = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    tcfg = cfg['training']
    args_train = TrainingArguments(
        output_dir=str(out_dir),
        run_name=cfg['project']['run_name'],
        num_train_epochs=float(tcfg['epochs']),
        per_device_train_batch_size=int(tcfg['per_device_train_batch_size']),
        gradient_accumulation_steps=int(tcfg['gradient_accumulation_steps']),
        learning_rate=float(tcfg['learning_rate']),
        warmup_ratio=float(tcfg.get('warmup_ratio', 0.03)),
        logging_steps=int(tcfg['logging_steps']),
        save_steps=int(tcfg['save_steps']),
        save_total_limit=int(tcfg.get('save_total_limit', 2)),
        bf16=bool(tcfg.get('bf16', True)),
        fp16=bool(tcfg.get('fp16', False)),
        gradient_checkpointing=bool(tcfg.get('gradient_checkpointing', True)),
        report_to=tcfg.get('report_to', []),
        remove_unused_columns=False,
        logging_first_step=True,
    )
    trainer = Trainer(model=model, args=args_train, train_dataset=tokenized, data_collator=collator, tokenizer=tokenizer)
    result = trainer.train()

    # --- Step 1: Save adapter ---
    print('Saving adapter artifacts.')
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f'Adapter saved to: {adapter_dir}')

    # --- Step 2: Merge adapter into base model inline ---
    # Reload base in bf16/fp16 (not 4-bit) so merge_and_unload() works correctly.
    merge_dtype = torch.bfloat16 if str(cfg['model'].get('torch_dtype', 'bf16')).lower() in {'bf16', 'bfloat16'} else torch.float16
    merged_dir = ensure_dir(Path(out_dir) / 'merged')
    print(f'Reloading base model in {merge_dtype} for merge (not 4-bit).')
    base_for_merge = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=merge_dtype,
        device_map='auto',
        trust_remote_code=cfg['model'].get('trust_remote_code', True),
    )
    print('Loading adapter for merge.')
    peft_model = PeftModel.from_pretrained(base_for_merge, str(adapter_dir))
    print('Merging adapter into base model.')
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))
    print(f'Merged checkpoint saved to: {merged_dir}')

    # --- Step 3: Log metrics and summary ---
    metrics = result.metrics or {}
    write_training_summary(log_summary_path, cfg, sample_count=len(ds), metrics=metrics, merged_dir=str(merged_dir))
    mlflow_log_basic(cfg, metrics=metrics, artifact_dir=str(merged_dir))
    print(f'Config snapshot: {used_cfg}')
    print(f'Training summary: {log_summary_path}')


if __name__ == '__main__':
    main()
