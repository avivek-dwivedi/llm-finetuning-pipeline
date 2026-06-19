# Watchouts

- Do not run the full dataset first. Start with the 20-sample smoke-test configs and increase only after the lifecycle works.
- Do not commit models, adapters, or MLflow artifacts to Git.
- Use S3 for artifacts that need to move between EC2 instances.
- Use an SSH tunnel for MLflow and Ray dashboards instead of exposing ports publicly.
- Do not expose Ray ports publicly in the EC2 security group.
- QLoRA training quantization is not the same as deployment quantization.
- Merge adapters by reloading the base model in bf16 or fp16, not 4-bit, before calling `merge_and_unload()`.
- For Ray multi-GPU runs, start with LoRA + bf16 before attempting more complex quantized distributed setups.
- Log the Git commit hash for every class run so students can reproduce the exact environment.