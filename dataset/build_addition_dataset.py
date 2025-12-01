import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from datasets import load_from_disk

VOCAB_CHARS = list("0123456789+=")
SPECIAL_TOKENS = ["[PAD]", "[THINK]", "[EOS]"]


class DifficultySampler(Sampler):
    """Samples examples based on difficulty threshold for curriculum learning."""

    def __init__(self, torch_dataset, starting_difficulty, name="Sampler", mixed=False):
        """

        torch_dataset.dataset   -> underlying HF Dataset loaded as .arrow
        torch_dataset.tokenizer -> tokenizer

        Args:
            torch_dataset (_type_): _description_
            starting_difficulty (_type_): _description_
            name (str): Name to identify this sampler in logs (e.g., "train", "val")
        """
        self.name = name
        self.difficulties = self.load_difficulty_tensor(torch_dataset.dataset)
        self.max_difficulty = int(self.difficulties.max().item())

        # Mix 1/5 regular data + 4/5 curriculum data.
        self.mixed = mixed
        self.is_capped = False

        print(f"{self.name} Sampler: Maximum difficulty in dataset = {self.max_difficulty}")

        # Set starting difficulty & print this
        self.update_difficulty(starting_difficulty)
        self.filtered = None

    def load_difficulty_tensor(self, ds, col="difficulty", dtype=torch.uint8):
        # Load column chunks one by one
        # Computing list(dataset['difficulty]) too expensive
        chunks = []
        # Arbitrary batch size, this is for efficiency / hence hardcoded
        for batch in ds.iter(batch_size=65536): 
            chunks.append(torch.tensor(batch[col], dtype=dtype))
        return torch.cat(chunks)
    

    def update_difficulty(self, new_difficulty):
        self.curr_difficulty = new_difficulty
        count = int((self.difficulties <= self.curr_difficulty).sum().item())

        # Create mask
        self.mask = self.difficulties <= self.curr_difficulty
        # return indices where mask == True
        self.idxs = self.mask.nonzero(as_tuple=True)[0]

        # Store all indices for mixed sampling
        self.all_idxs = torch.arange(len(self.difficulties))

        self.length = int((self.difficulties <= self.curr_difficulty).sum().item())

        print(f"{self.name} Sampler: Running at difficulty {new_difficulty} with {count} examples")


    def __iter__(self):

        # If threshold is too low, return an empty iterator.
        if len(self.idxs) == 0:
            return iter(())

        if self.mixed:
            # Mix 4/5 curriculum data + 1/5 regular data
            num_curriculum = int(0.8 * self.length)  # 4/5
            num_regular = self.length - num_curriculum  # 1/5

            # Sample curriculum indices (with difficulty threshold)
            curriculum_perm = torch.randperm(len(self.idxs))[:num_curriculum]
            curriculum_samples = self.idxs[curriculum_perm]

            # Sample regular indices (from entire dataset)
            regular_perm = torch.randperm(len(self.all_idxs))[:num_regular]
            regular_samples = self.all_idxs[regular_perm]

            # Concatenate and shuffle
            mixed_idxs = torch.cat([curriculum_samples, regular_samples])
            final_perm = torch.randperm(len(mixed_idxs))

            for i in final_perm:
                yield mixed_idxs[i].item()
        else:
            # Pure curriculum learning (original behavior)
            perm = torch.randperm(len(self.idxs))
            for i in perm:
                yield self.idxs[i].item()

    def __len__(self):
        return self.length


class ArithmeticTokenizer:
    def __init__(self):
        # Build vocab: special tokens first, then characters
        self.vocab = SPECIAL_TOKENS + VOCAB_CHARS
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

    def encode(self, text):
        # Convert a string into a list of token IDs
        return [self.token_to_id[i] for i in text]

    def decode(self, token_ids):
        # Convert a list of token IDs back into a string
        return "".join(self.id_to_token.get(i, "[UNK]") for i in token_ids)

    def __len__(self):
        return len(self.vocab)
    

class ArithmeticTorchDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # need to do teacher forcing for auto-regressive training
        # otherwise model is not learning 
        item = self.dataset[idx]

        # input_ids = self.tokenizer.encode(item["prompt"])
        # target_ids = self.tokenizer.encode(item["answer"])

        prompt = self.tokenizer.encode(item["prompt"])
        answer = self.tokenizer.encode(item["answer"])
        eos_id = self.tokenizer.token_to_id["[EOS]"]

        full = prompt + answer + [eos_id]
        input_ids = full[:-1]
        labels    = full[1:]

        IGNORE = -100
        prompt_len = len(prompt)

        labels = torch.tensor(labels, dtype=torch.long)
        # Keep index (prompt_len-1) so the model learns to output the first answer token
        if prompt_len > 1:
            labels[:prompt_len-1] = IGNORE

        result = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": labels,  # Already a tensor, no need to convert again
            "operand_len_a": torch.tensor(item["operand_len_a"]),
            "operand_len_b": torch.tensor(item["operand_len_b"]),
            "difficulty": torch.tensor(item['difficulty'])
        }

        return result

def dynamic_padding_collate_fn(batch):
    """
    Pads all sequences in the batch dynamically to the max length in that batch.
    """
    pad_token_id = torch.tensor([0], dtype=torch.long) # PAD: 0

    input_ids = [b["input_ids"] for b in batch]
    labels = [b["labels"] for b in batch]
    operand_len_a = torch.stack([b["operand_len_a"] for b in batch])
    operand_len_b = torch.stack([b["operand_len_b"] for b in batch])
    difficulty = torch.stack([b["difficulty"] for b in batch])

    # Compute max lengths in this batch
    max_input_len = max(x.size(0) for x in input_ids)
    max_label_len = max(x.size(0) for x in labels)
    # pad everything to 1000 since we set seq_len=1000 in launch fn

    def pad_sequence(seq_list, max_len):
        padded = []
        for seq in seq_list:
            pad_len = max_len - seq.size(0)
            padded.append(torch.cat([seq, pad_token_id.repeat(pad_len)]))
        return torch.stack(padded)

    padded_inputs = pad_sequence(input_ids, max_input_len)
    padded_labels = pad_sequence(labels, max_label_len)

    # Set labels to -100 for cross entropy loss
    padded_labels[padded_labels == pad_token_id] = -100

    return {
        "input_ids": padded_inputs,
        "labels": padded_labels,
        "operand_len_a": operand_len_a,
        "operand_len_b": operand_len_b,
        "difficulty": difficulty
    }

def main():
    # This path is local to __name__ == '__main__', hence defined down here.
    DATASET_PATH = "data/addition_dataset"

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

    # Example batch
    batch = next(iter(train_loader))
    print("Input shape:", batch["input_ids"].shape)
    print("Labels shape:", batch["labels"].shape)

    raw_sample = train_ds.dataset[0]
    print("Raw prompt:", raw_sample.get("prompt"))
    print("Raw answer:", raw_sample.get("answer"))
    print("Encoded input:", tokenizer.encode(raw_sample["prompt"]))
    print("Encoded label:", tokenizer.encode(raw_sample["answer"]))






if __name__ == "__main__":
    main()