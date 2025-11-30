import os
import torch
import hydra
import wandb
from typing import Optional, Any, Sequence, List
import math


from omegaconf import DictConfig

## Dataset imports
from torch.utils.data import DataLoader
from datasets import load_from_disk

## Project imports
import train_utils
from dataset.build_addition_dataset import (
    ArithmeticTorchDataset, ArithmeticTokenizer, dynamic_padding_collate_fn, DifficultySampler
)
from my_dataclasses import PretrainConfig, TrainState

from evaluate import evaluate_model


DATASET_PATH = "data/addition_dataset_big_scored_max_len"


def init_train_state(config: PretrainConfig, vocab_size, total_steps: int):
    # Model
    model, optimizer, optimizer_lr = train_utils.create_model(config, vocab_size)

    return TrainState(
        step=0,
        total_steps=total_steps,
        model=model,
        optimizer=optimizer,
        optimizer_lr=optimizer_lr,
        carry=None
    )

def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return train_utils.cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    # Assemble hydra dict into dataclass
    config = train_utils.load_config(hydra_config)

    torch.random.manual_seed(config.seed)

    ### Will move this out of pretrain to keep it generic

    # 1. Load dataset
    dataset_dict = load_from_disk(DATASET_PATH)
    tokenizer = ArithmeticTokenizer()

    # Wrap splits as PyTorch Datasets
    train_ds = ArithmeticTorchDataset(dataset_dict["train"], tokenizer)
    val_ds = ArithmeticTorchDataset(dataset_dict["val"], tokenizer)
    test_ds = ArithmeticTorchDataset(dataset_dict["test"], tokenizer)


    curr_difficulty = 1
    # Setup curriculum learning with difficulty sampler
    train_sampler = DifficultySampler(train_ds, starting_difficulty=curr_difficulty, name="Train")
    val_sampler = DifficultySampler(val_ds, starting_difficulty=curr_difficulty, name="Val")

    # Create DataLoaders with dynamic padding
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        sampler=train_sampler, # Curriculum sampler for training
        # shuffle=True, # Cannot be used with sampler, shuffle inside sampler.
        collate_fn=dynamic_padding_collate_fn
    )
    val_loader = DataLoader(val_ds,
                            batch_size=config.batch_size,
                            sampler=val_sampler,
                            collate_fn=dynamic_padding_collate_fn)
    
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, collate_fn=dynamic_padding_collate_fn)

    vocab_size = len(tokenizer)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * config.epochs

    # Encapsulate model and iterative data in a single state dataclass
    train_state = init_train_state(config, vocab_size=vocab_size, total_steps=total_steps)

    wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(), settings=wandb.Settings(_disable_stats=True))  # type: ignore
    wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)

    # move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_state.model.to(device)

    examples_seen = 0

    # Train
    print(f"Starting training with {len(train_ds)} training samples.")

    for epoch in range(config.epochs):
        train_state.model.train()
        epoch_metrics = {
            "loss": 0.0,
            "lm_loss": 0.0,
            "q_halt_loss": 0.0,
            "accuracy": 0.0,
            "exact_accuracy": 0.0,
            "steps": 0.0,
            "count": 0.0
        }

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Track current batch size and update examples_seen
            batch_size = batch['input_ids'].size(0)
            examples_seen += batch_size

            model_batch = {
                "inputs": batch["input_ids"],
                "labels": batch["labels"],
            }

            # for now, we re-initialize carry every batch
            # but persist within the batch

            # Initialize carry
            with torch.device(device):
                train_state.carry = train_state.model.initial_carry(model_batch)

            # forward
            new_carry, loss, metrics, output, halted = train_state.model(carry=train_state.carry, batch=model_batch, return_keys=["logits", "preds"])

            # backward
            train_state.optimizer.zero_grad()
            loss.backward()

            # step optimizer lr
            lr_this_step = compute_lr(train_state.optimizer_lr, config, train_state)
            for param_group in train_state.optimizer.param_groups:
                param_group["lr"] = lr_this_step
            train_state.optimizer.step()

            # training metrics
            epoch_metrics["loss"] += loss.item()
            for k in ["lm_loss", "q_halt_loss", "accuracy", "exact_accuracy", "steps", "count"]:
                if k in metrics:
                    epoch_metrics[k] += metrics[k].item() if isinstance(metrics[k], torch.Tensor) else metrics[k]

            # Report to terminal
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Step: {train_state.step} | LR: {lr_this_step:.8f} | Loss: {loss.item():.4f} | Difficulty: {train_sampler.curr_difficulty}")

            # Save checkpoint
            if train_state.step % config.save_interval == 0 and train_state.step > 0:
                train_utils.save_checkpoint(train_state, config, filename=f"step_{train_state.step}.pt")

            # Run validation
            if train_state.step % config.eval_interval == 0 and train_state.step > 0: # type: ignore
                print(f"Evaluating: Step {train_state.step} | Difficulty: {val_sampler.curr_difficulty}")
                total_metrics = evaluate_model(train_state.model, val_loader, device, num_batches=10)
                val_metrics = {f"val/{k}": v for k, v in total_metrics.items()}
                val_metrics['examples_seen'] = examples_seen
                wandb.log(val_metrics, step=train_state.step)

                # Curriculum learning
                if total_metrics['exact_accuracy_pct'] >= config.curriculum_threshold:
                    curr_difficulty += 1
                    train_sampler.update_difficulty(curr_difficulty)
                    val_sampler.update_difficulty(curr_difficulty)

            # Compute percentage metrics
            train_metrics = {f"train/{k}": v for k, v in metrics.items()}
            # Add percentage metrics if count > 0
            if metrics.get("count", 0) > 0:
                train_metrics["train/accuracy_pct"] = metrics["accuracy"] / metrics["count"]
                train_metrics["train/exact_accuracy_pct"] = metrics["exact_accuracy"] / metrics["count"]
            # Use examples seen as X-axis in WandB to make runs comparable.
            train_metrics['examples_seen'] = examples_seen
            wandb.log(train_metrics, step=train_state.step)
            train_state.step += 1
            wandb.log({"train/difficulty": train_sampler.curr_difficulty, "val/difficulty": val_sampler.curr_difficulty}, step=train_state.step)
    
        train_utils.save_checkpoint(train_state, config, filename=f"step_{train_state.step}.pt")
        # Final epoch metrics
        num_batches = len(train_loader)
        print(f"\n=== Epoch {epoch} Summary ===")
        print(f"Avg Loss: {epoch_metrics['loss'] / num_batches:.4f}")
        print(f"Avg LM Loss: {epoch_metrics['lm_loss'] / num_batches:.4f}")
        print(f"Avg Q Halt Loss: {epoch_metrics['q_halt_loss'] / num_batches:.4f}")
        if epoch_metrics['count'] > 0:
            print(f"Accuracy: {epoch_metrics['accuracy'] / epoch_metrics['count']:.4f}")
            print(f"Exact Accuracy: {epoch_metrics['exact_accuracy'] / epoch_metrics['count']:.4f}")
            print(f"Avg Steps: {epoch_metrics['steps'] / epoch_metrics['count']:.2f}")
        print("="*30 + "\n")

if __name__ == "__main__":
    launch()
