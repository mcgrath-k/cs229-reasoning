import os
import torch
import hydra
import pydantic
import coolname
from typing import Optional, Any, Sequence, List


from omegaconf import DictConfig
from transformers import AutoTokenizer
from datasets import load_from_disk

from adam_atan2 import AdamATan2
from puzzle_dataset import PuzzleDatasetMetadata


## Project imports
import train_utils
from models.losses import ACTLossHead
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from my_dataclasses import PretrainConfig, TrainState


## Dataset imports
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk, DatasetDict

from dataset.build_addition_dataset import ArithmeticTorchDataset, ArithmeticTokenizer, dynamic_padding_collate_fn


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


    # Encapsulate model and iterative data in a single state dataclass
    train_state = init_train_state(config, vocab_size=len(tokenizer), 


    # for epoch in range(config.epochs):
    #     train_state.model.train()


if __name__ == "__main__":
    launch()