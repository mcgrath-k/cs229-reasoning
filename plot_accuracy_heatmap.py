# Designed by Aarohi Gupta - vectorized/committed by mcgrathk

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from train_utils import load_checkpoint
from datasets import load_dataset
from torch.utils.data import DataLoader
from dataset.build_addition_dataset import ArithmeticTorchDataset, ArithmeticTokenizer, dynamic_padding_collate_fn
from collections import defaultdict
from tqdm import tqdm
import pickle
from pathlib import Path

from constants import HF_DATASET_REPO

def evaluate_by_length(model, dataloader, tokenizer, device, max_batches=None, max_reasoning_steps=10):
    model.eval()

    length_accuracies = defaultdict(list)

    # Move token lookups outside the loop for efficiency
    eq_token = tokenizer.token_to_id.get("=", None)
    eos_token = tokenizer.token_to_id.get("[EOS]", -1)
    pad_token = 0

    batch_count = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating OOD accuracy"):
            if batch is None:  # Skip empty batches (shouldnt happen with max_seq_len 1000)
                continue

            if max_batches is not None and batch_count >= max_batches:
                break

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}

            model_batch = {
                "inputs": batch["input_ids"],
                "labels": batch["labels"],
            }
            with torch.device(device):
                carry = model.initial_carry(model_batch)

            for step_idx in range(max_reasoning_steps):
                # Forward pass
                carry, loss, metrics, outputs = model(
                    return_keys=["logits", "preds"],
                    carry=carry,
                    batch=model_batch,
                )

            # TODO [kmcgrath]: Consolidate duplicated logic on above lines.

            preds = outputs["preds"]
            labels = batch["labels"]

            # Move all metadata to CPU in one bulk transfer
            operand_len_a_cpu = batch["operand_len_a"].tolist()
            operand_len_b_cpu = batch["operand_len_b"].tolist()

            if eq_token is not None:
                # Find = positions for entire batch at once
                eq_mask = (batch["input_ids"] == eq_token)  # [batch_size, seq_len]
                has_eq = eq_mask.any(dim=1)  # [batch_size] - which examples have =
                eq_positions = eq_mask.float().argmax(dim=1)  # [batch_size] - first = position

                # Compute answer_start for entire batch
                answer_start_positions = eq_positions + 1  # [batch_size]

                # Find answer_end for entire batch
                batch_size, seq_len = labels.shape

                # Create a range tensor for sequence positions
                seq_range = torch.arange(seq_len, device=labels.device).unsqueeze(0)  # [1, seq_len]
                answer_start_expanded = answer_start_positions.unsqueeze(1)  # [batch_size, 1]

                # Mask for positions after answer_start
                after_answer_start = seq_range >= answer_start_expanded  # [batch_size, seq_len]

                # Find end tokens (EOS, PAD, or -100)
                end_token_mask = (labels == eos_token) | (labels == pad_token) | (labels == -100)
                end_token_mask = end_token_mask & after_answer_start  # Only after answer_start

                # For each example, find first end token position (or use seq_len if none found)
                # Add a large value where no end token exists, then take min
                end_positions = torch.where(
                    end_token_mask,
                    seq_range.expand(batch_size, -1),
                    torch.full((batch_size, seq_len), seq_len, device=labels.device)
                ).min(dim=1).values  # [batch_size]

                # Check correctness for entire batch at once
                # Create masks for valid answer regions
                answer_lengths = end_positions - answer_start_positions  # [batch_size]
                has_answer = (has_eq) & (answer_lengths > 0)  # [batch_size]

                # For examples with answers, check if predictions match labels
                # Single GPU-to-CPU transfer for answer metadata
                answer_lengths_cpu = answer_lengths.cpu().tolist()
                answer_start_positions_cpu = answer_start_positions.cpu().tolist()
                has_answer_cpu = has_answer.cpu().tolist()

                max_answer_len = max(answer_lengths_cpu) if any(has_answer_cpu) else 0

                if max_answer_len > 0:
                    # Create tensors to hold answers (padded to max_answer_len)
                    pred_answers = torch.full((batch_size, max_answer_len), -100,
                                             dtype=preds.dtype, device=preds.device)
                    true_answers = torch.full((batch_size, max_answer_len), -100,
                                             dtype=labels.dtype, device=labels.device)

                    # Fill in the actual answers - using CPU data to avoid GPU-to-CPU sync in loop
                    for i in range(batch_size):
                        if has_answer_cpu[i]:
                            ans_len = answer_lengths_cpu[i]
                            ans_start = answer_start_positions_cpu[i]
                            pred_answers[i, :ans_len] = preds[i, ans_start:ans_start + ans_len]
                            true_answers[i, :ans_len] = labels[i, ans_start:ans_start + ans_len]

                    # Vectorized correctness check
                    valid_mask = true_answers != -100  # [batch_size, max_answer_len]
                    matches = (pred_answers == true_answers) | (~valid_mask)  # [batch_size, max_answer_len]
                    is_correct_batch = matches.all(dim=1)  # [batch_size]

                    # Single GPU-to-CPU transfer for correctness results
                    is_correct_cpu = is_correct_batch.cpu().tolist()

                    # Store results using CPU data only
                    for i in range(batch_size):
                        if has_answer_cpu[i]:
                            len1 = operand_len_a_cpu[i]
                            len2 = operand_len_b_cpu[i]
                            length_accuracies[(len1, len2)].append(float(is_correct_cpu[i]))
            
            batch_count += 1
    
    model.train()
    return length_accuracies

def create_heatmap(length_accuracies, title="Positional Embedding Accuracy", footer_text=None):
    if not length_accuracies:
        print("No data to plot!")
        return
    
    all_lengths = list(length_accuracies.keys())
    len1_values = sorted(set(l[0] for l in all_lengths))
    len2_values = sorted(set(l[1] for l in all_lengths))
        
    max_len1 = max(len1_values)
    max_len2 = max(len2_values)
    
    # Create accuracy matrix - Initialize with NaN (will show as white/missing)
    accuracy_matrix = np.full((max_len1 + 1, max_len2 + 1), np.nan)
    count_matrix = np.zeros((max_len1 + 1, max_len2 + 1))
    
    # ONLY fill cells where we actually have evaluation results
    for (len1, len2), accuracies in length_accuracies.items():
        if accuracies:  # Only if we evaluated this combination
            accuracy_matrix[len1, len2] = np.mean(accuracies) * 100
            count_matrix[len1, len2] = len(accuracies)
    
    # Print statistics
    print("\nAccuracy Statistics:")
    for (len1, len2), accuracies in sorted(length_accuracies.items()):
        mean_acc = np.mean(accuracies) * 100
        print(f"  ({len1}, {len2}): {mean_acc:.1f}% ({len(accuracies)} examples)")
    
    accuracy_matrix_masked = np.ma.masked_invalid(accuracy_matrix)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap - masked values won't be shown
    im = ax.imshow(accuracy_matrix_masked, origin='lower', cmap='YlGnBu_r', # reverse = YlGnBu_r
                   vmin=0, vmax=100, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy', rotation=270, labelpad=20, fontsize=16)
    
    # Set labels
    ax.set_xlabel('Length of Operand Two', fontsize=14, labelpad=30)
    ax.set_ylabel('Length of Operand One', fontsize=14, labelpad=30)
    # Make tick marks bigger
    ax.tick_params(axis='both', which='major', length=8, width=1.5)

    ax.set_title(title, fontsize=16)
    
    ax.set_xticks(np.arange(min(len2_values), max(len2_values) + 1, 
                            max(1, (max(len2_values) - min(len2_values)) // 10)))
    ax.set_yticks(np.arange(min(len1_values), max(len1_values) + 1, 
                            max(1, (max(len1_values) - min(len1_values)) // 10)))
    ax.set_xlim(min(len2_values) - 0.5, max(len2_values) + 0.5)
    ax.set_ylim(min(len1_values) - 0.5, max(len1_values) + 0.5)

    if footer_text is not None:
        plt.tight_layout(rect=(0, 0.08, 1, 1)) # Some space for cropping watermark.

        fig.text(
            0.99, 0.01, # (x, y) in figure coordinates (0–1)
            footer_text,
            ha="right", va="bottom", # anchor bottom-right
            fontsize=8,
            alpha=0.6  
        )
    else:
        plt.tight_layout()

    
    return fig, accuracy_matrix, count_matrix

if __name__ == "__main__":
    CHECKPOINT_PATH = 'results_collection/FINAL_58598.pt'
    OUTPUT_FOLDER = 'plots'
    MAX_REASONING_STEPS = 10

    filename = Path(CHECKPOINT_PATH).stem
    
    output_dir = Path(Path(OUTPUT_FOLDER) / filename)
    os.makedirs(output_dir, exist_ok=True)
    
    tokenizer = ArithmeticTokenizer()

    model = load_checkpoint(CHECKPOINT_PATH, tokenizer, optimizer=None)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    
    dataset_dict = load_dataset(HF_DATASET_REPO)
    
    # using test set
    test_ds = ArithmeticTorchDataset(dataset_dict["val_extrapolate"], tokenizer)
    test_loader = DataLoader(test_ds, batch_size=128, collate_fn=dynamic_padding_collate_fn)

    
    length_accuracies = evaluate_by_length(model, 
                                           test_loader, 
                                           tokenizer, 
                                           device, 
                                           max_batches=None, 
                                           max_reasoning_steps=MAX_REASONING_STEPS)
    
    # save output length_accuracies for debugging
    with open(output_dir / "length_accuracies.pkl", "wb") as f:
        pickle.dump(length_accuracies, f)
    
    fig, accuracy_matrix, count_matrix = create_heatmap(length_accuracies, 
                                                        title="Model Exact Accuracy by Operand Length",
                                                        footer_text=filename)
    
    np.save(output_dir / "accuracy_matrix.npy", accuracy_matrix)
    np.save(output_dir / "count_matrix.npy", count_matrix)

    output_path = output_dir / "length_accuracy_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()