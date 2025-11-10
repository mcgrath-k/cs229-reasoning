import os
import json
import numpy as np
from typing import Optional, List
from pydantic import BaseModel
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from common import PuzzleDatasetMetadata

class DataConfig(BaseModel):
    tokenizer_name: str = "gpt2"
    dataset_name: str = "InfiniAILab/gsm_infinite_medium_8k"
    max_input_length: int = 5120  # Fits the ~4700 token inputs
    max_target_length: int = 512   # Output is small
    output_dir: str = "data/gsm-processed"
    ops_splits: Optional[List[str]] = None
    subsample_size: Optional[int] = None
    ignore_label_id: int = -100  # for loss computation

def process_gsm_for_trm(config: DataConfig):
    """
    Process GSM dataset into TRM format
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    dataset = load_dataset(config.dataset_name)
    
    # Select splits
    if config.ops_splits is not None:
        selected_splits = {k: dataset[k] for k in config.ops_splits if k in dataset}
    else:
        selected_splits = dataset # default to all splits
        
    # Combine all selected splits
    all_examples = []
    for split_name, split_data in selected_splits.items():
        all_examples.extend(list(split_data))
    
    # Subsample if requested
    if config.subsample_size is not None and config.subsample_size < len(all_examples):
        import random
        all_examples = random.sample(all_examples, config.subsample_size)
        
    # Process examples
    inputs = []
    labels = []
    
    for example in tqdm(all_examples):
        # Construct input from problem + question
        input_text = f"Problem: {example['problem']}\nQuestion: {example['question']}\nSolution:"
        target_text = example['solution']
        
        # Tokenize
        input_ids = tokenizer(
            input_text,
            truncation=True,
            max_length=config.max_input_length,
            add_special_tokens=True
        )['input_ids']
        
        target_ids = tokenizer(
            target_text,
            truncation=True,
            max_length=config.max_target_length,
            add_special_tokens=False
        )['input_ids']
        
        # For generation: concatenate input + target
        # TRM will learn to generate the target part
        full_sequence = input_ids + target_ids + [tokenizer.eos_token_id]
        
        # Pad to max length
        max_seq_len = config.max_input_length + config.max_target_length + 1
        padded_input = full_sequence + [tokenizer.pad_token_id] * (max_seq_len - len(full_sequence))
        
        # Create labels: -100 (ignore) for input portion, actual tokens for target
        padded_labels = (
            [config.ignore_label_id] * len(input_ids) +  # Don't compute loss on input
            target_ids + 
            [tokenizer.eos_token_id] +
            [config.ignore_label_id] * (max_seq_len - len(full_sequence))
        )
        
        inputs.append(padded_input[:max_seq_len])
        labels.append(padded_labels[:max_seq_len])
    
    # Convert to numpy
    inputs_array = np.array(inputs, dtype=np.int32)
    labels_array = np.array(labels, dtype=np.int32)
    
    num_examples = len(inputs)
    
    results = {
        "inputs": inputs_array,
        "labels": labels_array,
        "puzzle_indices": np.arange(num_examples + 1, dtype=np.int32),  # Each example is one "puzzle"
        "puzzle_identifiers": np.zeros(num_examples, dtype=np.int32),  # All same type
        "group_indices": np.arange(num_examples + 1, dtype=np.int32),  # Each example is its own group
    }
    
    # Create metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=config.max_input_length + config.max_target_length + 1, # max sequence length
        vocab_size=len(tokenizer),
        pad_id=tokenizer.pad_token_id,
        ignore_label_id=-100,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=num_examples,
        mean_puzzle_examples=1.0,
        total_puzzles=num_examples,
        sets=["all"]
    )
    
    # Save everything
    save_dir = os.path.join(config.output_dir, "train")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save metadata
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f, indent=2)
    
    # Save data arrays
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
    
    # Save tokenizer info
    with open(os.path.join(config.output_dir, "tokenizer_info.json"), "w") as f:
        json.dump({
            "tokenizer_name": config.tokenizer_name,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "vocab_size": len(tokenizer)
        }, f, indent=2)
    
    print(f"\nDataset saved to {config.output_dir}")
    print(f"Total examples: {num_examples}")
    print(f"Sequence length: {metadata.seq_len}")
    print(f"Vocabulary size: {metadata.vocab_size}")


if __name__ == "__main__":
    # TODO: change to argparse or similar for config
    config = DataConfig(
        tokenizer_name="gpt2",
        dataset_name="InfiniAILab/gsm_infinite_medium_8k",
        max_input_length=5120,
        max_target_length=512,
        output_dir="data/gsm-trm",
        ops_splits=['ops_2', 'ops_3'],  # Start with a couple splits
        subsample_size=1000  # Start small for testing
    )
    
    process_gsm_for_trm(config)