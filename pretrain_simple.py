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
from dataset.build_addition_dataset import ArithmeticTorchDataset, ArithmeticTokenizer, dynamic_padding_collate_fn
from my_dataclasses import PretrainConfig, TrainState

from evaluate import evaluate_model


DATASET_PATH = "data/addition_dataset"


def init_train_state(config: PretrainConfig, vocab_size):
    # Estimated total training steps
    total_steps = int(config.epochs / config.global_batch_size)

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

    # Create DataLoaders with dynamic padding
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=dynamic_padding_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=64, collate_fn=dynamic_padding_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=128, collate_fn=dynamic_padding_collate_fn)

    vocab_size = len(tokenizer)

    # Encapsulate model and iterative data in a single state dataclass
    train_state = init_train_state(config, vocab_size=vocab_size)

    wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(), settings=wandb.Settings(_disable_stats=True))  # type: ignore
    wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)

    # move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_state.model.to(device)

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
            model_batch = {
                "inputs": batch["input_ids"],
                "labels": batch["labels"],
                # make puzzle identifiers all zeros since we dont use them -- TODO: remove since not used
                "puzzle_identifiers": torch.zeros_like(batch["input_ids"], dtype=torch.long)
            }

            # for now, we re-initialize carry every batch
            # but persist within the batch

            # Initialize carry
            with torch.device(device):
                carry = train_state.model.initial_carry(model_batch)

            # ACT loop

            all_halted = False
            act_step = 0
            batch_metrics = None

            # loop until all sequences have halted or max steps reached
            while not all_halted and act_step < train_state.model.config.max_steps:

                # forward
                carry, loss, metrics, output, all_halted = train_state.model(carry=carry, batch=model_batch, return_keys=["logits", "preds"])

                # backward
                train_state.optimizer.zero_grad()
                loss.backward()

                # step optimizer lr
                lr_this_step = compute_lr(train_state.optimizer_lr, config, train_state)
                for param_group in train_state.optimizer.param_groups:
                    param_group["lr"] = lr_this_step
                train_state.optimizer.step()

                # accululate metrics for halted sequence
                if batch_metrics is None:
                    batch_metrics = {
                        k: (v.item() if torch.is_tensor(v) else v) 
                        for k, v in metrics.items()
                    }
                else:
                    for k, v in metrics.items():
                        batch_metrics[k] += v.item() if torch.is_tensor(v) else v

                train_state.step += 1
                act_step += 1

                ### ACT STEP LEVEL LOGGING  

                if train_state.step % config.save_interval == 0 and train_state.step > 0:
                    train_utils.save_checkpoint(train_state, config, filename=f"step_{train_state.step}.pt")

                if train_state.step % config.eval_interval == 0 and train_state.step > 0:
                    val_metrics = evaluate_model(
                        train_state.model, 
                        val_loader, 
                        device,
                        halt_max_steps=config.arch.halt_max_steps,
                        num_batches=128
                    )
                    val_metrics_log = {f"val/{k}": v for k, v in val_metrics.items()}
                    wandb.log(val_metrics_log, step=train_state.step)
                    
                    print(f"\n[Validation] Step {train_state.step}")
                    print(f"  Val Loss: {val_metrics['lm_loss']:.4f}")
                    print(f"  Val Accuracy: {val_metrics['accuracy_pct']:.4f}")
                    print(f"  Val Exact Accuracy: {val_metrics['exact_accuracy_pct']:.4f}")
                    print(f"  Val Avg Steps: {val_metrics['steps']:.2f}\n")

            # update epoch metrics
            count = batch_metrics.get("count", 0.0)
            if count > 0:
                for k in ["lm_loss", "q_halt_loss", "accuracy", "exact_accuracy", "q_halt_accuracy", "steps"]:
                    epoch_metrics[k] += batch_metrics[k]
                epoch_metrics["count"] += count
                epoch_metrics["act_iterations"] += act_step

            ### BATCH LEVEL LOGGING
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Step: {train_state.step} | LR: {lr_this_step:.8f} | Loss: {loss.item():.4f}")

                batch_loss = batch_metrics["lm_loss"] / count if count > 0 else 0
                batch_acc = batch_metrics["accuracy"] / count if count > 0 else 0
                batch_exact_acc = batch_metrics["exact_accuracy"] / count if count > 0 else 0
                
                print(f"Epoch {epoch} | Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"Step: {train_state.step} | LR: {lr_this_step:.8f} | "
                    f"Loss: {batch_loss:.4f} | Acc: {batch_acc:.4f} | "
                    f"ExactAcc: {batch_exact_acc:.4f} | ACT: {act_step}")
                
                # am i doing this too often? 
                wandb.log({
                    "train/batch_loss": batch_loss,
                    "train/batch_accuracy": batch_acc,
                    "train/batch_exact_accuracy": batch_exact_acc,
                    "train/batch_act_steps": act_step,
                    "train/lr": lr_this_step,
                }, step=train_state.step)
            
    
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
