import os
import torch
import hydra
import wandb
import coolname
import math
from omegaconf import DictConfig
from adam_atan2 import AdamATan2

## Dataset imports
from torch.utils.data import DataLoader
from datasets import load_from_disk

## Project imports
from models.losses import ACTLossHead
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from dataset.build_addition_dataset import ArithmeticTorchDataset, ArithmeticTokenizer, dynamic_padding_collate_fn
from my_dataclasses import PretrainConfig, TrainState




DATASET_PATH = "data/addition_dataset"

def create_model(config: PretrainConfig, vocab_size, seq_len):
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
        batch_size=config.global_batch_size,
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_puzzle_identifiers=0, # Required till we remove it from model
        causal=True # Enable autoregressive modeling
    )

    model = TinyRecursiveReasoningModel_ACTV1(model_cfg)
    model = ACTLossHead(model, loss_type='stablemax_cross_entropy') #type: ignore

    # Optimizers and lr
    optimizer = AdamATan2(
            model.parameters(),
            lr=0,  # Needs to be set by scheduler
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2))
    
    optimizer_lr = config.lr

    return model, optimizer, optimizer_lr


def init_train_state(config: PretrainConfig, vocab_size, seq_len):
    # Estimated total training steps
    total_steps = int(config.epochs / config.global_batch_size)

    # Model
    model, optimizer, optimizer_lr = create_model(config, vocab_size, seq_len)

    return TrainState(
        step=0,
        total_steps=total_steps,
        model=model,
        optimizer=optimizer,
        optimizer_lr=optimizer_lr,
        carry=None
    )



def load_config(hydra_config: DictConfig) -> PretrainConfig:
    config = PretrainConfig(**hydra_config) #type: ignore

    # Set project name, run name, and checkpoint path if not provided
    if config.project_name is None:
        config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-ACT-torch"
    if config.run_name is None:
        config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
    if config.checkpoint_path is None:
        config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

    return config



@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):



    # Assemble hydra dict into dataclass
    config = load_config(hydra_config)

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
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=dynamic_padding_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=32, collate_fn=dynamic_padding_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=32, collate_fn=dynamic_padding_collate_fn)

    vocab_size = len(tokenizer)
    seq_len = 1000 # Need to sleep on this. Does rotary encoding need max_seq_len?


    # Encapsulate model and iterative data in a single state dataclass
    train_state = init_train_state(config, vocab_size=vocab_size, seq_len=seq_len) 

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
                # make puzzle identifiers all zeros since we dont use them
                "puzzle_identifiers": torch.zeros_like(batch["input_ids"], dtype=torch.long)
            }

            # for now, we re-initialize carry every batch
            # but persist within the batch
            if train_state.carry is None:
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

            train_state.step += 1

            # training metrics
            epoch_metrics["loss"] += loss.item()
            for k in ["lm_loss", "q_halt_loss", "accuracy", "exact_accuracy", "steps", "count"]:
                    if k in metrics:
                        epoch_metrics[k] += metrics[k].item() if isinstance(metrics[k], torch.Tensor) else metrics[k]

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Step: {train_state.step} | LR: {lr_this_step:.8f} | Loss: {loss.item():.4f}")
                # print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | "
                #         f"Loss: {loss.item():.4f} | "
                #         f"Acc: {metrics['accuracy'].item() / metrics['count'].item() if metrics['count'].item() > 0 else 0:.4f} | "
                #         f"Exact Acc: {metrics['exact_accuracy'].item() / metrics['count'].item() if metrics['count'].item() > 0 else 0:.4f}")

            wandb.log(metrics, step=train_state.step)
    
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