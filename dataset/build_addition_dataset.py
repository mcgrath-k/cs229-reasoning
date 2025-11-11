import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk

DATASET_PATH = "data/addition_dataset"
VOCAB_CHARS = list("0123456789+=")
SPECIAL_TOKENS = ["[PAD]", "[THINK]"]


class ArithmeticTokenizer:
    def __init__(self):
        # Build vocab: special tokens first, then characters
        self.vocab = SPECIAL_TOKENS + VOCAB_CHARS
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

    def encode(self, text):
        # Convert a string into a list of token IDs
        return [self.token_to_id.get(ch, self.token_to_id["[PAD]"]) for ch in text]

    def decode(self, token_ids):
        # Convert a list of token IDs back into a string
        return "".join(self.id_to_token.get(i, "") for i in token_ids)

    def __len__(self):
        return len(self.vocab)
    

class ArithmeticTorchDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        input_ids = self.tokenizer.encode(item["input_text"])
        target_ids = self.tokenizer.encode(item["answer"])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(target_ids, dtype=torch.long),
            "operand_len_a": torch.tensor(item["operand_len_a"]),
            "operand_len_b": torch.tensor(item["operand_len_b"]),
        }

def dynamic_padding_collate_fn(batch):
    """
    Pads all sequences in the batch dynamically to the max length in that batch.
    """
    pad_token_id = batch[0]["input_ids"].new_tensor([0])  # [PAD] = 0

    input_ids = [b["input_ids"] for b in batch]
    labels = [b["labels"] for b in batch]
    operand_len_a = torch.stack([b["operand_len_a"] for b in batch])
    operand_len_b = torch.stack([b["operand_len_b"] for b in batch])

    # Compute max lengths in this batch
    max_input_len = max(x.size(0) for x in input_ids)
    max_label_len = max(x.size(0) for x in labels)

    def pad_sequence(seq_list, max_len):
        padded = []
        for seq in seq_list:
            pad_len = max_len - seq.size(0)
            padded.append(torch.cat([seq, pad_token_id.repeat(pad_len)]))
        return torch.stack(padded)

    padded_inputs = pad_sequence(input_ids, max_input_len)
    padded_labels = pad_sequence(labels, max_label_len)

    return {
        "input_ids": padded_inputs,
        "labels": padded_labels,
        "operand_len_a": operand_len_a,
        "operand_len_b": operand_len_b,
    }

def main():
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






if __name__ == "__main__":
    main()