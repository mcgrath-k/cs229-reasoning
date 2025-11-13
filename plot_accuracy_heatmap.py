import torch
import numpy as np
import matplotlib.pyplot as plt
from train_utils import load_checkpoint
from datasets import load_from_disk
from torch.utils.data import DataLoader
from dataset.build_addition_dataset import ArithmeticTorchDataset, ArithmeticTokenizer, dynamic_padding_collate_fn
from collections import defaultdict
from tqdm import tqdm
import pickle


def evaluate_by_length(model, dataset, tokenizer, device, max_batches=None):
    model.eval()
    
    length_accuracies = defaultdict(list)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=dynamic_padding_collate_fn, shuffle=False)
    
    batch_count = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if batch is None:  # Skip empty batches (shouldnt happen with max_seq_len 1000)
                continue

            if max_batches is not None and batch_count >= max_batches:
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
            new_carry, loss, metrics, outputs, all_halted = model(
                return_keys=["logits", "preds"],
                carry=carry,
                batch=model_batch,
            )

            preds = outputs["preds"]
            labels = batch["labels"]
            operand_len_a = batch["operand_len_a"]
            operand_len_b = batch["operand_len_b"]

            for i in range(preds.shape[0]):
                len1 = operand_len_a[i].item()
                len2 = operand_len_b[i].item()

                input_ids = batch["input_ids"][i]
                eq_token = tokenizer.token_to_id.get("=", None)

                if eq_token is not None:
                    eq_positions = (input_ids == eq_token).nonzero(as_tuple=True)[0]
                    if len(eq_positions) > 0:
                        eq_pos = eq_positions[0].item()
                        
                        # find end of answer
                        eos_token = tokenizer.token_to_id.get("[EOS]", -1)
                        pad_token = 0
                        
                        answer_start = eq_pos + 1
                        answer_end = answer_start
                        
                        # Find end of answer in labels
                        for j in range(answer_start, len(labels[i])):
                            label_val = labels[i, j].item()
                            if label_val in [eos_token, pad_token, -100]:
                                answer_end = j
                                break
                        
                        if answer_end > answer_start:
                            # Check exact match
                            pred_answer = preds[i, answer_start:answer_end]
                            true_answer = labels[i, answer_start:answer_end]
                            # Filter out -100 (ignored tokens)
                            valid_mask = true_answer != -100
                            if valid_mask.any():
                                is_correct = torch.all(pred_answer[valid_mask] == true_answer[valid_mask]).item()
                                length_accuracies[(len1, len2)].append(float(is_correct))
            
            batch_count += 1
    
    model.train()
    return length_accuracies

def create_heatmap(length_accuracies, title="Positional Embedding Accuracy"):
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
    print(f"\nAccuracy Statistics:")
    for (len1, len2), accuracies in sorted(length_accuracies.items()):
        mean_acc = np.mean(accuracies) * 100
        print(f"  ({len1}, {len2}): {mean_acc:.1f}% ({len(accuracies)} examples)")
    
    accuracy_matrix_masked = np.ma.masked_invalid(accuracy_matrix)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap - masked values won't be shown
    im = ax.imshow(accuracy_matrix_masked, origin='lower', cmap='RdYlGn', 
                   vmin=0, vmax=100, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy', rotation=270, labelpad=20, fontsize=12)
    
    # Set labels
    ax.set_xlabel('Length of Operand Two', fontsize=14)
    ax.set_ylabel('Length of Operand One', fontsize=14)
    ax.set_title(title, fontsize=16)
    
    ax.set_xticks(np.arange(min(len2_values), max(len2_values) + 1, 
                            max(1, (max(len2_values) - min(len2_values)) // 10)))
    ax.set_yticks(np.arange(min(len1_values), max(len1_values) + 1, 
                            max(1, (max(len1_values) - min(len1_values)) // 10)))
    ax.set_xlim(min(len2_values) - 0.5, max(len2_values) + 0.5)
    ax.set_ylim(min(len1_values) - 0.5, max(len1_values) + 0.5)
    
    plt.tight_layout()
    return fig, accuracy_matrix, count_matrix

if __name__ == "__main__":
    DATASET_PATH = "data/addition_dataset"
    CHECKPOINT_PATH = 'checkpoints/arithmetic/step_4900.pt'
    
    tokenizer = ArithmeticTokenizer()
    model = load_checkpoint(CHECKPOINT_PATH, tokenizer, optimizer=None)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    
    dataset_dict = load_from_disk(DATASET_PATH)
    
    # using test set
    test_ds = ArithmeticTorchDataset(dataset_dict["test"], tokenizer)
    
    length_accuracies = evaluate_by_length(model, test_ds, tokenizer, device, max_batches=None)
    # save output length_accuracies for debugging
    with open("length_accuracies.pkl", "wb") as f:
        pickle.dump(length_accuracies, f)
    
    fig, accuracy_matrix, count_matrix = create_heatmap(length_accuracies, 
                                                        title="Model Accuracy by Operand Length")
    
    np.save("accuracy_matrix.npy", accuracy_matrix)
    np.save("count_matrix.npy", count_matrix)

    output_path = "length_accuracy_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()