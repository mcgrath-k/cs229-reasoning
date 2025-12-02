import pydantic

import torch
from torch import nn

from typing import Optional, Any, Sequence, List

from dataclasses import dataclass

class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig

    # Dataset
    dataset_path: str # Can be huggingface repo_id or local folder
    dataset_is_remote: bool # Is the dataset hosted on huggingface?
    scoring_func: str # String that denotes which dataset scoring func to run.

    # Hyperparams
    batch_size: int
    epochs: int

    curriculum_threshold: float
    mix_curriculum_ratio: float # Mix (1 - mix_curriculum_ratio) * regular data + (mix_curriculum_ratio) curriculum data


    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    checkpoint_every_epoch: bool = False
    eval_interval: Optional[int] = None
    min_eval_interval: Optional[int] = 0 # when to start eval
    eval_save_outputs: List[str] = []

    seed: int = 0
    save_interval: int # Save every n steps.

    ema: bool = False # use Exponential-Moving-Average
    ema_rate: float = 0.999 # EMA-rate
    freeze_weights: bool = False # If True, freeze weights and only learn the embeddings

@dataclass
class TrainState:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    optimizer_lr: float
    carry: Any
    step: int
    total_steps: int
