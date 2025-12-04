import os
import math
import torch
import coolname
import numpy as np
from adam_atan2 import AdamATan2
from omegaconf import DictConfig

from my_dataclasses import TrainState, PretrainConfig

from models.losses import LossHead
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from dataset.build_addition_dataset import ArithmeticTokenizer
from constants import IGNORE_LABEL_ID


def create_model(config: PretrainConfig, vocab_size):
    """Create Model

    Args:
        config (PretrainConfig): Configuration object containing model specification
        vocab_size (int): length of vocab
        seq_len (int): length of sequence
    """

    # **config.arch.__pydantic_extra__ is double splatting all of the settings contained in
    # config/arch/(selected_model).yaml. Whatever model is picked will be concatenated with this dict.

    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.batch_size,
        vocab_size=vocab_size,
        max_reasoning_steps=config.arch.max_reasoning_steps,
        causal=True # Enable autoregressive modeling [working]
    )

    model = TinyRecursiveReasoningModel_ACTV1(model_cfg)
    model = LossHead(model, loss_type='stablemax_cross_entropy') #type: ignore

    # Optimizers and lr
    optimizer = AdamATan2(
            model.parameters(),
            lr=0,  # Needs to be set by scheduler
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2))
    
    optimizer_lr = config.lr

    return model, optimizer, optimizer_lr

def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )

def save_checkpoint(train_state, config, filename="checkpoint.pt"):
    """Save model, optimizer, and metadata to disk."""
    os.makedirs(config.checkpoint_path, exist_ok=True)
    path = os.path.join(config.checkpoint_path, filename)

    torch.save({
        "step": train_state.step,
        "model_state_dict": train_state.model.state_dict(),
        "optimizer_state_dict": train_state.optimizer.state_dict(),
        "config": config.model_dump(),
    }, path)

    print(f"Saved checkpoint to {path}")
    return path


def quick_validate(output_tensor: torch.Tensor, label_tensor: torch.Tensor) -> None:
    """
    Decode and print model outputs and labels for quick debugging.

    Args:
        output_tensor (torch.Tensor): Predicted token IDs.
        label_tensor (torch.Tensor): True token IDs (with IGNORE_LABEL_ID positions removed).
    """

    tokenizer = ArithmeticTokenizer()

    # Slice top of tensor for samples.
    output_samples = output_tensor.tolist()

    mask = np.array((label_tensor != IGNORE_LABEL_ID).tolist())
    label_samples = np.array(label_tensor.tolist())

    label_masked = label_samples[mask]

    print(f"output: {tokenizer.decode(output_samples)}")
    print(f"labels: {tokenizer.decode(label_masked)}")


def load_config(hydra_config: DictConfig) -> PretrainConfig:
    config = PretrainConfig(**hydra_config) #type: ignore

    # Set project name, run name, and checkpoint path if not provided
    if config.project_name is None:
        config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-ACT-torch"
    if config.run_name is None:
        config.run_name = f"{config.arch.name.split('@')[-1]}_{coolname.generate_slug(2)}"
    if config.checkpoint_path is None:
        config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

    if config.curriculum_type:
        if not (0 < config.mix_curriculum_ratio <= 1):
            raise ValueError(f"Mixing ratio: {config.mix_curriculum_ratio} not acceptable."
                              "Choose from range (0,1].")

    if config.checkpoint_every_epoch:
        print("!!! Saving a checkpoint every Epoch - Ensure sufficient space is available !!!")

    return config

def load_checkpoint(path, tokenizer, optimizer=None, device="cuda"):
    """Load model and optimizer from checkpoint."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    checkpoint = torch.load(path, map_location=device)

    config = load_config(checkpoint['config'])

    model, _, _= create_model(config, vocab_size=len(tokenizer))
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Loaded checkpoint from {path} (step={checkpoint.get('step', 'N/A')})")

    return model
