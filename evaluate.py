import torch
from train_utils import load_checkpoint
from datasets import load_from_disk
from typing import List

from torch.utils.data import DataLoader

from dataset.build_addition_dataset import ArithmeticTorchDataset, ArithmeticTokenizer, dynamic_padding_collate_fn


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


def evaluate_model(model, val_loader, device, halt_max_steps, num_batches=None):
    model.eval()
    
    total_metrics = {
        "loss": 0.0,
        "lm_loss": 0.0,
        "q_halt_loss": 0.0,
        "accuracy": 0.0,
        "exact_accuracy": 0.0,
        "q_halt_accuracy": 0.0,
        "steps": 0.0,
        "count": 0,
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
            
            # During eval, run for FULL halt_max_steps for consistency 
            # TODO: does this make sense?
            all_halted = False
            act_step = 0
            
            while not all_halted and act_step < halt_max_steps:
                carry, loss, metrics, outputs, all_halted = model(
                    return_keys=["logits", "preds"],
                    carry=carry,
                    batch=model_batch,
                )
                act_step += 1
            
            # Accumulate metrics from this batch (only from halted sequences)
            count = metrics["count"].item() if torch.is_tensor(metrics["count"]) else metrics["count"]
            
            if count > 0:
                for k in ["lm_loss", "q_halt_loss", "accuracy", "exact_accuracy", "q_halt_accuracy", "steps"]:
                    if k in metrics:
                        val = metrics[k].item() if torch.is_tensor(metrics[k]) else metrics[k]
                        total_metrics[k] += val
                
                total_metrics["count"] += count
            
            batch_count += 1
    
    # Normalize by count 
    if total_metrics["count"] > 0:
        for k in ["lm_loss", "q_halt_loss", "accuracy", "exact_accuracy", "q_halt_accuracy", "steps"]:
            total_metrics[k] /= total_metrics["count"]
        
        # These are already percentages
        total_metrics["accuracy_pct"] = total_metrics["accuracy"]
        total_metrics["exact_accuracy_pct"] = total_metrics["exact_accuracy"]
        total_metrics["q_halt_accuracy_pct"] = total_metrics["q_halt_accuracy"]
    else:
        total_metrics["accuracy_pct"] = 0.0
        total_metrics["exact_accuracy_pct"] = 0.0
        total_metrics["q_halt_accuracy_pct"] = 0.0
    
    model.train()
    return total_metrics


def generate_answer(
    model,
    prompt_tokens: List[int],
    tokenizer,
    device: torch.device,
    max_answer_len: int = None,
    halt_max_steps: int = None
) -> List[int]:
    """
    Generate answer tokens autoregressively with persistent carry state.
    
    Args:
        model: The TRM model
        prompt_tokens: List of token IDs for the prompt (e.g., "123+456=")
        tokenizer: Tokenizer with token_to_id dict
        device: Device to run on
        max_answer_len: Maximum answer length (defaults to config value)
        halt_max_steps: Maximum ACT steps (defaults to config value)
    
    Returns:
        List of generated token IDs (without EOS)
    """
    model.eval()
    
    # Use config defaults if not specified
    if max_answer_len is None:
        max_answer_len = model.config.generation_max_answer_len
    if halt_max_steps is None:
        halt_max_steps = model.config.halt_max_steps
    
    pad_token_id = tokenizer.token_to_id["[PAD]"]
    eos_token_id = tokenizer.token_to_id["[EOS]"]
    
    # Pre-allocate space: prompt + max answer length
    max_total_len = len(prompt_tokens) + max_answer_len
    input_ids = prompt_tokens + [pad_token_id] * max_answer_len
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    # Create batch
    batch = {
        "inputs": input_ids,
        "labels": torch.full_like(input_ids, -100)  # Not used during generation
    }
    
    # Initialize carry ONCE with full length
    with torch.no_grad():
        with torch.device(device):
            carry = model.initial_carry(batch)
        
        current_pos = len(prompt_tokens)
        generated = []
        
        for gen_step in range(max_answer_len):
            # ACT loop: model "thinks" about current state
            all_halted = False
            
            for act_step in range(halt_max_steps):
                carry, _, _, outputs, all_halted = model(
                    carry=carry,
                    batch=batch,
                    return_keys=["logits"]
                )
                if all_halted:
                    break
            
            # Get logits for current position
            # Note: -1 because model predicts next token (autoregressive shift)
            logits = outputs["logits"][0, current_pos - 1, :]
            next_token = torch.argmax(logits).item()
            
            # Check for end of sequence
            if next_token == eos_token_id:
                break
            
            generated.append(next_token)
            
            # Update input at next position (in-place, no shape change!)
            input_ids[0, current_pos] = next_token
            current_pos += 1
            
            # Update batch with new input (carry persists automatically)
            batch["inputs"] = input_ids
            
            # Carry state persists with same shape throughout!
    
    return generated


def generate_answer_string(
    model,
    prompt_str: str,
    tokenizer,
    device: torch.device,
    **kwargs
) -> str:
    """
    Convenience function that handles tokenization and decoding.
    
    Args:
        model: The TRM model
        prompt_str: Prompt string (e.g., "123+456=")
        tokenizer: Tokenizer
        device: Device to run on
        **kwargs: Additional arguments for generate_answer
    
    Returns:
        Generated answer as string (e.g., "579")
    """
    prompt_tokens = tokenizer.encode(prompt_str)
    generated_tokens = generate_answer(model, prompt_tokens, tokenizer, device, **kwargs)
    return tokenizer.decode(generated_tokens)



if __name__ == "__main__":
    DATASET_PATH = "data/addition_dataset"
    CHECKPOINT_PATH = 'checkpoints/arithmetic/TinyRecursiveReasoningModel_ACTV1_sage-wolf/step_7000.pt'

    prompt_text = "10+60="
    tokenizer = ArithmeticTokenizer()
    model = load_checkpoint(CHECKPOINT_PATH, tokenizer, optimizer=None)

    # move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    generated_text = generate(model, tokenizer, prompt_text)

    print(generated_text)


    ## Test evaluate model
    dataset_dict = load_from_disk(DATASET_PATH)
    tokenizer = ArithmeticTokenizer()
    val_ds = ArithmeticTorchDataset(dataset_dict["val"], tokenizer)
    val_loader = DataLoader(val_ds, batch_size=128, collate_fn=dynamic_padding_collate_fn)

    total_metrics = evaluate_model(model, val_loader, device, num_batches=128)


    print(total_metrics)