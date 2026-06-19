from __future__ import annotations
import argparse
from pathlib import Path
import ray
from ray.train import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer
from ray.train.huggingface.transformers import RayTrainReportCallback, prepare_trainer
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from utils.config import load_config, ensure_dir
from utils.tracking import setup_wandb, mlflow_log_basic


def dtype_from_name(name: str):
    return torch.bfloat16 if str(name).lower() in {'bf16', 'bfloat16'} else torch.float16


def train_loop(cfg):
    setup_wandb(cfg)
    print('Ray worker starting. Hugging Face Trainer will run the training loop and PEFT will inject the LoRA adapter.')
    model_name = cfg['model']['name']
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=cfg['model'].get('trust_remote_code', True))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype_from_name(cfg['model'].get('torch_dtype', 'bf16')), trust_remote_code=cfg['model'].get('trust_remote_code', True))
    model.config.use_cache = False
    lcfg = cfg['lora']
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=lcfg['r'], lora_alpha=lcfg['alpha'], lora_dropout=lcfg['dropout'], target_modules=lcfg['target_modules'], bias='none'))
    model.print_trainable_parameters()

    ds = load_dataset('json', data_files=cfg['data']['output_jsonl'], split='train')
    max_len = int(cfg['training']['max_seq_length'])
    def tok(batch):
        return tokenizer(batch['text'], truncation=True, max_length=max_len, padding=False)
    tokenized = ds.map(tok, batched=True, remove_columns=ds.column_names)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    out_dir = ensure_dir(cfg['project']['output_dir'])
    tcfg = cfg['training']
    args = TrainingArguments(
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
        gradient_checkpointing=bool(tcfg.get('gradient_checkpointing', True)),
        report_to=tcfg.get('report_to', []),
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=tokenized, data_collator=collator, tokenizer=tokenizer, callbacks=[RayTrainReportCallback()])
    trainer = prepare_trainer(trainer)
    result = trainer.train()
    final = Path(out_dir) / 'adapter'
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    return result.metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/ray_llm_sft_lora.yaml')
    parser.add_argument('--ray-address', default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    addr = args.ray_address if args.ray_address is not None else cfg['ray'].get('address')
    ray.init(address=addr) if addr else ray.init()

    ray_cfg = cfg['ray']
    print('Ray is the orchestration and distributed worker layer. Hugging Face Trainer performs training. PEFT performs adapter injection.')
    scaling = ScalingConfig(num_workers=int(ray_cfg['num_workers']), use_gpu=bool(ray_cfg.get('use_gpu', True)), resources_per_worker={'CPU': int(ray_cfg.get('cpus_per_worker', 4)), 'GPU': 1})
    run_config = RunConfig(name=cfg['project']['run_name'], storage_path=str(Path(cfg['project'].get('ray_storage_path', './ray_results')).resolve()))
    trainer = TorchTrainer(train_loop, train_loop_config=cfg, scaling_config=scaling, run_config=run_config)
    result = trainer.fit()
    print(result)
    mlflow_log_basic(cfg, metrics=result.metrics if hasattr(result, 'metrics') else {}, artifact_dir=str(Path(cfg['project']['output_dir']) / 'adapter'))




if __name__ == '__main__':
    main()
