import os
import torch
from train_utils import load_checkpoint
from datasets import load_from_disk, load_dataset

from torch.utils.data import DataLoader

from dataset.build_addition_dataset import ArithmeticTorchDataset, ArithmeticTokenizer, dynamic_padding_collate_fn

from constants import HF_DATASET_REPO

def generate(model, tokenizer, prompt_text, max_length=100, device='cuda', max_reasoning_steps=10):
    model.eval()
    
    current_ids = tokenizer.encode(prompt_text)
    current_tensor = torch.tensor([current_ids], dtype=torch.long, device=device)
    
    seq_len = 200  # TODO: Should match config
    if current_tensor.shape[1] < seq_len:
        pad_len = seq_len - current_tensor.shape[1]
        current_tensor = torch.cat([
            current_tensor,
            torch.zeros(1, pad_len, dtype=torch.long, device=device)
        ], dim=1)
    
    # Initialize carry
    batch = {
        "inputs": current_tensor,
        "labels": current_tensor,  # Dummy labels for generation
    }
    with torch.device(device):
        carry = model.initial_carry(batch)
    
    eos_token = tokenizer.token_to_id.get("[EOS]", None)
    # Should not be None, but putting just in case
    if eos_token is None:
        print("Warning: [EOS] token not found in tokenizer!")
    
    generated_length = len(current_ids)
    
    with torch.no_grad():
        for _ in range(max_length):
            for step in range(max_reasoning_steps):
                final = (step == max_reasoning_steps - 1)
                
                carry, loss, metrics, outputs, halted = model(
                    return_keys=["logits", "preds"],
                    carry=carry,
                    batch=batch,
                )
                
                if final:            
                    # Get prediction for the last non-padding position
                    last_pos = generated_length - 1
                    next_token_logits = outputs["logits"][0, last_pos, :]
                    next_token = torch.argmax(next_token_logits).item()
            
            if eos_token is not None and next_token == eos_token:
                break
            
            current_ids.append(next_token)
            generated_length += 1
            
            # update the next position in the current tensor
            # maybe there is better way to do this, but for now just copy full tensor 
            current_tensor[0, generated_length - 1] = next_token
            
            # Update batch
            batch["inputs"] = current_tensor
            batch["labels"] = current_tensor
            
            # if reached seq_len since model cant handle longer
            if generated_length >= seq_len:
                break
    
    # Decode result
    generated_text = tokenizer.decode(current_ids)
    return generated_text


def evaluate_model(model, val_loader, device, num_batches=None, max_reasoning_steps=10):
    # TODO: do we want to set a max number of reasoning steps here too?
   
    model.eval()
    
    total_metrics = {
        "lm_loss": 0.0,
        "accuracy": 0.0,
        "exact_accuracy": 0.0,
        "steps": 0.0,
        "count": 0.0,
    }
    
    batch_count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if num_batches is not None and batch_idx >= num_batches:
                break

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            model_batch = {
                "inputs": batch["input_ids"],
                "labels": batch["labels"],
            }

            with torch.device(device):
                carry = model.initial_carry(model_batch)

            # Run fixed number of reasoning steps
            for step_idx in range(max_reasoning_steps):
                is_final_step = (step_idx == max_reasoning_steps - 1)
                
                # Forward pass
                carry, loss, metrics, output = model(
                    carry=carry,
                    batch=model_batch,
                    return_keys=["logits", "preds"]
                )
                
                # Only collect metrics on final step
                if is_final_step:
                    for k in ["lm_loss", "accuracy", "exact_accuracy", "steps", "count"]:
                        if k in metrics:
                            total_metrics[k] += metrics[k].item() if isinstance(metrics[k], torch.Tensor) else metrics[k]

            batch_count += 1

    # Average metrics (avoid division by zero)
    if batch_count > 0:
        for k in total_metrics:
            total_metrics[k] /= batch_count
    else:
        print("Warning: No validation batches processed!")
    
    # Compute accuracy percentages
    if total_metrics["count"] > 0:
        total_metrics["accuracy_pct"] = total_metrics["accuracy"] / total_metrics["count"]
        total_metrics["exact_accuracy_pct"] = total_metrics["exact_accuracy"] / total_metrics["count"]
    else:
        total_metrics["accuracy_pct"] = 0.0
        total_metrics["exact_accuracy_pct"] = 0.0
    
    model.train()
    return total_metrics



if __name__ == "__main__":

    CHECKPOINT_PATH = 'checkpoints/arithmetic/TinyRecursiveReasoningModel_ACTV1_sage-wolf/step_7000.pt'

    dataset_dict = load_dataset(HF_DATASET_REPO)
    
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Saved model weights: {CHECKPOINT_PATH} can not be found")

    prompt_text = "10+60="
    batch_size = 128
    tokenizer = ArithmeticTokenizer()
    model = load_checkpoint(CHECKPOINT_PATH, tokenizer, optimizer=None)

    # move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    generated_text = generate(model, tokenizer, prompt_text)

    print(generated_text)


    ## Test evaluate model
    dataset_dict = load_dataset(HF_DATASET_REPO)
    tokenizer = ArithmeticTokenizer()
    val_ds = ArithmeticTorchDataset(dataset_dict["val"], tokenizer)
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=dynamic_padding_collate_fn)

    total_metrics = evaluate_model(model, val_loader, device, num_batches=batch_size)


    print(total_metrics)