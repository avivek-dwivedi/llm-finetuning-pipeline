from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any


def setup_wandb(config: Dict[str, Any]) -> None:
    project = config.get('project', {})
    os.environ.setdefault('WANDB_PROJECT', os.getenv('WANDB_PROJECT', project.get('name', 'ec2-finetuning-v1')))
    os.environ.setdefault('WANDB_RUN_GROUP', project.get('run_name', 'training-run'))


def mlflow_log_basic(config: Dict[str, Any], metrics: Dict[str, float] | None = None, artifact_dir: str | None = None):
    try:
        import mlflow
    except Exception as e:
        print(f'MLflow not available: {e}')
        return

    tracking_uri = os.getenv('MLFLOW_TRACKING_URI', './mlruns')
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.get('project', {}).get('name', 'ec2-finetuning-v1'))

    with mlflow.start_run(run_name=config.get('project', {}).get('run_name', 'run')):
        def flatten(prefix, obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    yield from flatten(f'{prefix}.{k}' if prefix else str(k), v)
            elif isinstance(obj, (str, int, float, bool)) or obj is None:
                yield prefix, obj

        for k, v in flatten('', config):
            mlflow.log_param(k[:250], v)
        if metrics:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, float(v))
        if artifact_dir and Path(artifact_dir).exists():
            mlflow.log_artifacts(str(artifact_dir), artifact_path='artifacts')
