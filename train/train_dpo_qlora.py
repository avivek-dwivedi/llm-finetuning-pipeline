from __future__ import annotations
import argparse
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel, TaskType
from trl import DPOTrainer, DPOConfig
from utils.config import load_config, ensure_dir
from utils.tracking import setup_wandb, mlflow_log_basic


def dtype_from_name(name: str):
    name = str(name).lower()
    if name in {'bf16', 'bfloat16'}:
        return torch.bfloat16
    if name in {'fp16', 'float16'}:
        return torch.float16
    return torch.float32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/dpo_qlora.yaml')
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_wandb(cfg)

    out_dir = ensure_dir(cfg['project']['output_dir'])
    adapter_dir = ensure_dir(Path(out_dir) / 'adapter')

    model_name = cfg['model']['name']
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=cfg['model'].get('trust_remote_code', True))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = dtype_from_name(cfg['model'].get('torch_dtype', 'bf16'))
    model_kwargs = {'trust_remote_code': cfg['model'].get('trust_remote_code', True), 'torch_dtype': torch_dtype}
    if cfg['model'].get('use_4bit', True):
        model_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch_dtype, bnb_4bit_use_double_quant=True)
        model_kwargs['device_map'] = 'auto'

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.config.use_cache = False
    if cfg['model'].get('use_4bit', True):
        model = prepare_model_for_kbit_training(model)

    lcfg = cfg['lora']
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lcfg['r']),
        lora_alpha=int(lcfg['alpha']),
        lora_dropout=float(lcfg['dropout']),
        target_modules=lcfg['target_modules'],
        bias='none',
    )

    ds = load_dataset('json', data_files=cfg['data']['output_jsonl'], split='train')
    tcfg = cfg['training']
    dpo_args = DPOConfig(
        output_dir=str(out_dir),
        run_name=cfg['project']['run_name'],
        num_train_epochs=float(tcfg['epochs']),
        per_device_train_batch_size=int(tcfg['per_device_train_batch_size']),
        gradient_accumulation_steps=int(tcfg['gradient_accumulation_steps']),
        learning_rate=float(tcfg['learning_rate']),
        logging_steps=int(tcfg['logging_steps']),
        save_steps=int(tcfg['save_steps']),
        beta=float(tcfg.get('beta', 0.1)),
        max_length=int(tcfg.get('max_length', 512)),
        max_prompt_length=int(tcfg.get('max_prompt_length', 256)),
        bf16=bool(tcfg.get('bf16', True)),
        fp16=bool(tcfg.get('fp16', False)),
        report_to=tcfg.get('report_to', []),
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_cfg,
    )
    result = trainer.train()

    # --- Step 1: Save adapter ---
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f'DPO adapter saved to: {adapter_dir}')

    # --- Step 2: Merge adapter into base model inline ---
    # Reload base in bf16/fp16 (not 4-bit) so merge_and_unload() works correctly.
    merge_dtype = torch_dtype
    merged_dir = ensure_dir(Path(out_dir) / 'merged')
    print(f'Reloading base model in {merge_dtype} for DPO merge.')
    base_for_merge = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=merge_dtype,
        device_map='auto',
        trust_remote_code=cfg['model'].get('trust_remote_code', True),
    )
    peft_model = PeftModel.from_pretrained(base_for_merge, str(adapter_dir))
    print('Merging DPO adapter into base model.')
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))
    print(f'DPO merged checkpoint saved to: {merged_dir}')

    mlflow_log_basic(cfg, metrics=result.metrics if hasattr(result, 'metrics') else {}, artifact_dir=str(merged_dir))


if __name__ == '__main__':
    main()
