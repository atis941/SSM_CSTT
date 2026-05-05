import shutil

import torch
import os
import json
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import matplotlib.pyplot as plt
import re
from typing import Any, Dict, Iterable, Optional, Union, List

####### function for creating the VOCABULARY from the present dataset #######
def build_phoneme_vocab(root_dir, save_dir=None):
    phoneme_set = set()
    
    # iterate over the phoneme files and extract the phonemes
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".PHN"):
                phn_path = os.path.join(dirpath, file)

                with open(phn_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        phoneme = parts[2]
                        phoneme_set.add(phoneme)
    
    # turn the list of extracted phonemes into a set -> for uniqueness
    phoneme_list = sorted(list(phoneme_set))

    # explicitly define blank
    phoneme_to_idx_vocab = {"<blank>": 0}

    # define the indices for all the other phonemes in the vocabulary
    for i, ph in enumerate(phoneme_list):
        phoneme_to_idx_vocab[ph] = i + 1

    # create a vocabulary where the indices are the keys and the phonemes are the values
    idx_to_phoneme_vocab = {i: ph for ph, i in phoneme_to_idx_vocab.items()}

    # save the created vocabularies in json files
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        with open(os.path.join(save_dir, "phoneme_to_idx.json"), "w") as f:
            json.dump(phoneme_to_idx_vocab, f, indent=4)

        with open(os.path.join(save_dir, "idx_to_phoneme.json"), "w") as f:
            json.dump(idx_to_phoneme_vocab, f, indent=4)

    return phoneme_to_idx_vocab, idx_to_phoneme_vocab


####### Collate functions for the dataloaders #####
def collate_fn_ctc(batch: list) -> tuple:
    """
    Collate function for variable-length TIMIT samples.

    Parameters
    ----------
    batch : list
        List of tuples (features, labels):
        - features has shape [T, 80]
        - labels has shape [L]

    Returns
    -------
    tuple: 
        1. element: padded_features : torch.Tensor
                    Shape [B, T_max, 80]

        2. element: feature_lengths : torch.Tensor
                    Shape [B]

        3. element: padded_labels : torch.Tensor
                    Shape [B, L_max]

        4. element: label_lengths : torch.Tensor
                    Shape [B]
    """

    # unzip batch into two tuples
    feature_list, label_list = zip(*batch)

    # store original lengths before padding
    feature_lengths_original = torch.tensor(
        [x.shape[0] for x in feature_list],
        dtype=torch.long
    )

    label_lengths_original = torch.tensor(
        [y.shape[0] for y in label_list],
        dtype=torch.long
    )

    # pad features along time dimension
    # each feature tensor has shape [T, 80]
    # result: [B, T_max, 80]
    padded_features = pad_sequence(
        feature_list,
        batch_first=True,
        padding_value=0.0
    )

    concatenated_labels = torch.cat(label_list, dim=0)

    return padded_features, feature_lengths_original, concatenated_labels, label_lengths_original

###### FUNCTTIONS for saving the results #####
def make_json_serializable(obj: Any) -> Any:
    """
    Convert common Python / PyTorch objects into JSON-serializable values.
    """
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, torch.device):
        return str(obj)

    if isinstance(obj, torch.dtype):
        return str(obj)

    if isinstance(obj, torch.Tensor):
        return {
            "type": "torch.Tensor",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "device": str(obj.device),
        }

    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [make_json_serializable(v) for v in obj]

    # fallback
    return str(obj)


def save_training_result(
    figures: Optional[Union[plt.Figure, Iterable[plt.Figure]]] = None,
    model: Optional[torch.nn.Module] = None,
    config: Optional[Dict[str, Any]] = None,
    base_results_dir: str = "results",
    model_filename: str = "ssm_model_state_dict.pt",
    log_files: Optional[List[str]] = None,
    config_filename: str = "run_config.json",) -> Path:
    """
    Create a new results/result_<n> directory and save:
      1. matplotlib figure(s)
      2. model state_dict
      3. JSON config / metadata

    Parameters
    ----------
    figures : matplotlib.figure.Figure or iterable of Figure, optional
        One figure or multiple figures to save.
    model : torch.nn.Module, optional
        Trained model to save.
    config : dict, optional
        Dictionary containing experiment settings / variable names and values.
    base_results_dir : str
        Root directory in which result_<n> folders are created.
    model_filename : str
        Filename for the saved model weights.
    log_files: List, optional
        List of the txt log files to save in the results_<n> directory
    config_filename : str
        Filename for the saved JSON config.

    Returns
    -------
    Path
        Path to the created result directory.
    """
    base_dir = Path(base_results_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # find existing result_<n> directories
    pattern = re.compile(r"^result_(\d+)$")
    existing_indices = []

    for entry in base_dir.iterdir():
        if entry.is_dir():
            match = pattern.match(entry.name)
            if match:
                existing_indices.append(int(match.group(1)))

    next_index = 0 if len(existing_indices) == 0 else max(existing_indices) + 1
    result_dir = base_dir / f"result_{next_index}"
    result_dir.mkdir(parents=True, exist_ok=False)

    # ---------- SAVE FIGURES ----------
    if figures is not None:
        if isinstance(figures, plt.Figure):
            figures = [figures]

        for i, fig in enumerate(figures):
            fig.savefig(result_dir / f"figure_{i}.png", dpi=300, bbox_inches="tight")
            fig.savefig(result_dir / f"figure_{i}.pdf", bbox_inches="tight")

    # ---------- SAVE MODEL ----------
    if model is not None:
        torch.save(model.state_dict(), result_dir / model_filename)

    # ---------- SAVE CONFIG ----------
    if config is not None:
        serializable_config = make_json_serializable(config)
        with open(result_dir / config_filename, "w", encoding="utf-8") as f:
            json.dump(serializable_config, f, indent=4, ensure_ascii=False)

    # ---------- SAVE LOG FILES ----------
    if log_files is not None:
        for log_file_path in log_files:
            src_path = Path(log_file_path)

            if src_path.exists():
                dst_path = result_dir / src_path.name
                shutil.copy(src_path, dst_path)
            else:
                print(f"Warning: log file {log_file_path} not found.")

    return result_dir


###### HELPER FUNCTIONS FOR CTC DECODING AND PER ######
def ctc_greedy_decode(pred_ids, blank_idx=0):
    """
    Collapse repeats and remove blank symbols from a single predicted sequence.
    pred_ids: list[int]
    """
    decoded = []
    prev = None

    for idx in pred_ids:
        if idx != blank_idx and idx != prev:
            decoded.append(idx)
        prev = idx

    return decoded


import torch

def edit_distance(seq1, seq2, device=None):
    """
    Compute Levenshtein edit distance between two sequences using a PyTorch tensor
    as the dynamic-programming table.

    Parameters
    ----------
    seq1 : list[int] or torch.Tensor
        First sequence.
    seq2 : list[int] or torch.Tensor
        Second sequence.
    device : torch.device or None
        Device for the DP tensor.

    Returns
    -------
    int
        Edit distance between seq1 and seq2.
    """
    if not torch.is_tensor(seq1):
        seq1 = torch.tensor(seq1, dtype=torch.long, device=device)
    else:
        seq1 = seq1.to(device=device, dtype=torch.long)

    if not torch.is_tensor(seq2):
        seq2 = torch.tensor(seq2, dtype=torch.long, device=device)
    else:
        seq2 = seq2.to(device=device, dtype=torch.long)

    m = seq1.numel()
    n = seq2.numel()

    dp = torch.zeros((m + 1, n + 1), dtype=torch.long, device=seq1.device)

    dp[:, 0] = torch.arange(m + 1, device=seq1.device)
    dp[0, :] = torch.arange(n + 1, device=seq1.device)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = 1 + torch.min(torch.stack([
                    dp[i - 1, j],     # deletion
                    dp[i, j - 1],     # insertion
                    dp[i - 1, j - 1]  # substitution
                ]))

    return int(dp[m, n].item())


def compute_batch_per_from_log_probs(log_probs, 
                                     input_lengths, 
                                     targets, 
                                     target_lengths, 
                                     blank_idx=0):
    """
    Compute PER statistics for one batch.

    Parameters
    ----------
    log_probs : torch.Tensor
        Shape [batch, time, vocab_size]
    input_lengths : torch.Tensor
        Shape [batch]
    targets : torch.Tensor
        Concatenated target phoneme indices, shape [sum(target_lengths)]
    target_lengths : torch.Tensor
        Shape [batch]
    blank_idx : int
        Blank index used in CTC.

    Returns
    -------
    batch_per : float
        Batch phoneme error rate
    total_edit_distance : int
        Sum of edit distances over all utterances in batch
    total_target_length : int
        Sum of target phoneme lengths over all utterances in batch
    """
    pred_ids_batch = torch.argmax(log_probs, dim=-1)   # [B, T]
    batch_size = pred_ids_batch.size(0)

    total_edit_distance = 0
    total_target_length = 0

    target_offset = 0

    for b in range(batch_size):
        current_input_length = int(input_lengths[b].item())
        current_target_length = int(target_lengths[b].item())

        # predicted sequence for one utterance
        pred_ids = pred_ids_batch[b, :current_input_length].detach().cpu().tolist()
        decoded_pred = ctc_greedy_decode(pred_ids, blank_idx=blank_idx)

        # target sequence for one utterance
        target_seq = targets[target_offset:target_offset + current_target_length].detach().cpu().tolist()
        target_offset += current_target_length

        # edit distance
        dist = edit_distance(decoded_pred, target_seq)

        total_edit_distance += dist
        total_target_length += current_target_length

    batch_per = total_edit_distance / total_target_length if total_target_length > 0 else 0.0

    return batch_per, total_edit_distance, total_target_length

def decode_batch_to_phonemes(log_probs, input_lengths, targets, target_lengths, idx_to_phoneme, blank_idx=0):
    """
    Decode a batch and return list of (predicted_phonemes, target_phonemes)
    """
    pred_ids = torch.argmax(log_probs, dim=-1)  # [B, T]

    decoded_results = []

    offset = 0
    for i in range(pred_ids.shape[0]):
        # prediction
        pred_seq = pred_ids[i, :input_lengths[i]].cpu().tolist()
        decoded_pred = ctc_greedy_decode(pred_seq, blank_idx=blank_idx)
        pred_phonemes = [idx_to_phoneme[p] for p in decoded_pred]

        # target
        tgt_len = target_lengths[i].item()
        tgt_seq = targets[offset:offset + tgt_len].cpu().tolist()
        offset += tgt_len
        tgt_phonemes = [idx_to_phoneme[t] for t in tgt_seq]

        decoded_results.append((pred_phonemes, tgt_phonemes))

    return decoded_results

def log_message(msg, file_handle):
    print(msg)
    file_handle.write(msg + "\n")