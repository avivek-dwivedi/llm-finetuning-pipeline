# Quantization And Merge

## Training

The LLM SFT V1 path trains with a 4-bit base model plus a LoRA adapter:

- bitsandbytes loads the base model in 4-bit NF4
- PEFT injects the LoRA adapter
- training updates adapter weights only

This keeps GPU memory lower during training, which is why it is useful on EC2 for small-class demos.

## Save

Training saves the adapter only, not a merged standalone model.

Expected adapter artifacts include:

- `adapter_model.safetensors`
- `adapter_config.json`
- tokenizer files
- `used_config.yaml`
- `training_summary.json`

## Merge

The merge step is separate:

1. Reload the base model in `bf16` or `fp16`
2. Load the adapter with `PeftModel.from_pretrained(...)`
3. Call `merge_and_unload()`
4. Save the merged checkpoint with `save_pretrained(..., safe_serialization=True)`

This is why the merge script should not default to a 4-bit training load path.

## Inference

You have two valid inference modes:

1. Base model + adapter
2. Merged standalone checkpoint

Both are useful to demonstrate in class:

- base + adapter shows the PEFT deployment path
- merged checkpoint shows the standalone export path