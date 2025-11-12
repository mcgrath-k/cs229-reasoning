import torch

def generate(model, tokenizer, prompt_text, max_length=100, device='cuda'):
    model.eval()
    
    current_ids = tokenizer.encode(prompt_text)
    current_tensor = torch.tensor([current_ids], dtype=torch.long, device=device)
    
    seq_len = 200  # Should match config
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
        "puzzle_identifiers": torch.zeros_like(current_tensor, dtype=torch.long)
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
            carry, loss, metrics, outputs, all_halted = model(
                return_keys=["logits", "preds"],
                carry=carry,
                batch=batch
            )
            
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


def evaluate_model(model, val_loader, device, num_batches=None):
    model.eval()
    
    total_metrics = {
        "loss": 0.0,
        "lm_loss": 0.0,
        "q_halt_loss": 0.0,
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
                "puzzle_identifiers": torch.zeros_like(batch["input_ids"], dtype=torch.long)
            }
            
            with torch.device(device):
                carry = model.initial_carry(model_batch)
            
            # Forward pass
            new_carry, loss, metrics, outputs, all_halted = model(
                return_keys=["logits", "preds"],
                carry=carry,
                batch=model_batch,
            )
            
            # Accumulate metrics
            total_metrics["loss"] += loss.item()
            for k in ["lm_loss", "q_halt_loss", "accuracy", "exact_accuracy", "steps", "count"]:
                if k in metrics:
                    total_metrics[k] += metrics[k].item() if isinstance(metrics[k], torch.Tensor) else metrics[k]

            batch_count += 1
    
    # Average metrics
    for k in total_metrics:
        total_metrics[k] /= batch_count
    
    # Compute accuracy percentages
    if total_metrics["count"] > 0:
        total_metrics["accuracy_pct"] = total_metrics["accuracy"] / total_metrics["count"]
        total_metrics["exact_accuracy_pct"] = total_metrics["exact_accuracy"] / total_metrics["count"]
    else:
        total_metrics["accuracy_pct"] = 0.0
        total_metrics["exact_accuracy_pct"] = 0.0
    
    model.train()
    return total_metrics