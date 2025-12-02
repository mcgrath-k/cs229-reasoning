import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset, DatasetDict

SEED = 229

def generate_n_length_integer(n: int) -> int:
    """Generates specific length integer

    Args:
        n (int): Len of integer

    Returns:
        int: output integer
    """
    if n == 0:
        raise ValueError("Digit length must be greater than 0")
    if n == 1:
        return random.randint(1,9)
    return random.randint(10**(n-1), 10**n-1)

class AdditionDataset():
    """Generate Addition Dataset


    Args:
        max_len (int): Maximum length of a single digit
        total_samples(int): Total number of samples
        thinking_tokens (Optional(int)): Optional number of thinking tokens to insert 

    # Design
    The goal is to sweep a graph from 0 -> 100 digit length and
    demonstrate performance over each position. Thus, we need to 
    ensure we generate numbers in each category of sequence length.

    We will likely need look up table identities (I.E 5+3 = 8) to learn the
    underlying addition algorithm besides long seq_len numbers.

    Let's do a uniform distribution? -> yes:
    per: https://proceedings.neurips.cc/paper_files/paper/2024/hash/c35986bc1ee29b31c1011481b77fe540-Abstract-Conference.html
    """

    def __init__(self, min_len, max_len: int, total_samples: int = 0, samples_per_pair=None, thinking_tokens: int = 0) -> None:
        self.min_len = min_len
        self.max_len = max_len
        self.total_samples = total_samples
        self.samples_per_pair = samples_per_pair
        self.thinking_tokens = thinking_tokens # In case we need 'scratchpad' tokens

    def _generate_pair_list(self):
        # generate permutation of (i, j)
        pairs = []
        for i in range(self.min_len, self.max_len + 1):
            for j in range(self.min_len, self.max_len + 1):
                pairs.append((i,j))
        return pairs
        
    def generate_examples(self):
        # Uniformly sample all (i, j) digit-length pairs up to max_len
        pairs = self._generate_pair_list()
        n_pairs = len(pairs)
        if not self.samples_per_pair:
            self.samples_per_pair = self.total_samples // n_pairs

        rows = []
        
        for i, j in tqdm(pairs, desc="Generating (i, j) length pairs", total=len(pairs)):
            for _ in tqdm(range(self.samples_per_pair),
                  desc=f"len(A)={i}, len(B)={j}",
                  leave=False,  # keeps the outer bar visible
                  ncols=80):
                A = generate_n_length_integer(i)
                B = generate_n_length_integer(j)
                C = A + B
                
                if self.thinking_tokens:
                    insert_string = f":{'.' * self.thinking_tokens}:"
                else:
                    insert_string = ""
                example = f"{A}+{B}{insert_string}="


                rows.append({
                        "prompt": example,
                        "operand_len_a": i,
                        "operand_len_b": j,
                        "answer": str(C),
                    })
                
        ds = Dataset.from_list(rows)
        return ds
    

def build_train_val_test_dataset(train_range =(1,20), 
                                 val_range=(1,20), 
                                 test_range=(1,100), 
                                 total_train_samples=5_000_000, 
                                 test_samples_per_pair=10,
                                 gen_extrapolation_dataset=True, # Mirror test set length
                                 thinking_tokens=0):
    assert train_range == val_range, "Split not supported"

    train_val_gen = AdditionDataset(train_range[0], train_range[1], total_samples=total_train_samples)
    test_gen= AdditionDataset(test_range[0], test_range[1], samples_per_pair=test_samples_per_pair)

    print("Generating train examples")
    train_val_ds = train_val_gen.generate_examples()
    print("Generating test examples")
    test_ds = test_gen.generate_examples()

    # Shuffle and split validation
    train_val_ds = train_val_ds.shuffle(seed=SEED)
    split = train_val_ds.train_test_split(test_size=0.10, seed=SEED)

    if gen_extrapolation_dataset:
        print("Generating validation extrapolation examples")
        val_extrapolate_gen = AdditionDataset(test_range[0], test_range[1], samples_per_pair=test_samples_per_pair)
        val_extrapolate_ds = val_extrapolate_gen.generate_examples()

        return DatasetDict({
            "train": split["train"],
            "val": split["test"],
            "val_extrapolate": val_extrapolate_ds,
            "test": test_ds
        })
    else:
        return DatasetDict({
            "train": split["train"],
            "val": split["test"],
            "test": test_ds
        })

def save_dataset_to_disk(dataset_dict, save_dir='data/addition_dataset_big'):
    os.makedirs(save_dir, exist_ok=True)
    dataset_dict.save_to_disk(save_dir)
    print("Saved Huggingface Dataset to disk")

def push_dataset_to_hub(dataset_dict: DatasetDict, repo_id: str):
    """Push dataset to Huggingface

    REQUIRES LOGIN prior to running this script:
    `hf auth login`
    
    Args:
        dataset_dict (DatasetDict): Huggingface Dataset
        repo_id (str): The name of pre-existing repo or new repo you want to create.
    """

    dataset_dict.push_to_hub(repo_id)

if __name__ == '__main__':
    random.seed(SEED)

    dataset_dict = build_train_val_test_dataset()

    save_dataset_to_disk(dataset_dict)

    
    # push_dataset_to_hub(dataset_dict, 'mkyle/addition_dataset_5_million')