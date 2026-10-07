import shutil

import torch
import os
import json
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import matplotlib.pyplot as plt
import re
from typing import Any, Dict, Iterable, Optional, Union, List
import torchaudio
import numpy as np
import editdistance

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


def collate_fn_ctc_time_domain_padding(batch: list,
                                       mel_transform: torchaudio.transforms.MelSpectrogram,
                                       hop_length: int) -> tuple:
    """
    Collate function for variable-length TIMIT samples.
    Also calculates the mel spectrogram for the padded time audio samples

    Parameters
    ----------
    batch : list
        List of tuples (waveform_tensor: tensor, 
                        original_waveform_length: tensor, 
                        phoneme_labels: tensor):
        - features has shape [T, max_waveform_length]
        - original_waveform_length is a scalar tensor
        - phoneme_labels has shape [number_of_labels]

    mel_spectrogram: torchaudio.transforms.MelSpectrogram
        the mel spectrogram function to turn the time domain samples into frequency domain features

    hop_length: int
        the hop length of the mel spectrogram. Necesary to be a parameter to calculate the length of the valid scpectrogram features

    Returns
    -------
    tuple: 
        1. element: mel_specs : torch.Tensor
                    The spectrogram features -> inputs of the model
                    Shape [B, T_max, 80]

        2. element: orig_feature_lengths : torch.Tensor
                    the original length of the spectrogram features without the padding -> input to ctc
                    Shape [B]

        3. element: concatenated_labels : torch.Tensor
                    The concatenated labels -> input to ctc
                    Shape [B, L_max]

        4. element: label_lengths : torch.Tensor
                    The label length of each feature in the batch -> input to ctc
                    Shape [B]
    """
    # zip the batch
    waveform_list, waveform_length_list, phoneme_label_list = zip(*batch)

    # [B, 1, max_waveform_length]
    waveform_batch = torch.stack(waveform_list, dim=0)

    # [B]
    waveform_length_batch = torch.stack(waveform_length_list, dim=0)

    # Calculate Mel spectrogram from padded waveforms
    # [B, 1, n_mels, T_max]
    mel_specs = mel_transform(waveform_batch)

    # Remove channel dimension:
    # [B, 1, n_mels, T_max] -> [B, n_mels, T_max]
    if mel_specs.dim() == 4:
        mel_specs = mel_specs.squeeze(1)

    # log compression -> mel spectrogram values can vary over several orders of magnitude.
    # taking the logarithm, compresses this dynamic range and makes learning easier for the neural network
    mel_specs = torch.log(mel_specs + 1e-6)

    # Change shape for the model:
    # [B, n_mels, T_max] -> [B, T_max, n_mels]
    mel_specs = mel_specs.transpose(1, 2)

    # Calculate real Mel lengths from original waveform lengths.
    # This assumes center=True in MelSpectrogram.
    orig_feature_lengths = waveform_length_batch // hop_length + 1

    # Safety: CTC input lengths must not be larger than actual T_max
    orig_feature_lengths = torch.clamp(
        orig_feature_lengths, max=mel_specs.shape[1])

    # Label lengths: [B]
    label_lengths = torch.tensor(
        [labels.shape[0] for labels in phoneme_label_list],
        dtype=torch.long
    )

    # CTC expects concatenated labels, not padded labels
    concatenated_labels = torch.cat(phoneme_label_list, dim=0)

    return mel_specs, orig_feature_lengths, concatenated_labels, label_lengths

def collate_fn_ctc_time_domain_features_during_training(batch: list) -> tuple:
    """
    Collate already padded time-domain TIMIT waveforms for CTC.

    Each dataset item contains:
        waveform: [1, T_max]
        original_waveform_length: scalar
        phoneme_labels: [L]

    Returns
    -------
    waveform_batch:
        Time-domain model inputs with shape [B, T_max, 1]

    waveform_lengths:
        Original unpadded waveform lengths with shape [B]

    concatenated_labels:
        All target phoneme sequences concatenated into one tensor, shape
        [sum(label_lengths)]

    label_lengths:
        Length of each target phoneme sequence, shape [B]
    """

    waveform_list, waveform_length_list, phoneme_label_list = zip(*batch)

    # Each waveform: [1, T_max]
    # Result: [B, 1, T_max]
    waveform_batch = torch.stack(waveform_list, dim=0)

    # Change to sequence-first feature representation:
    # [B, 1, T_max] -> [B, T_max, 1]
    waveform_batch = waveform_batch.transpose(1, 2)

    # [B]
    waveform_lengths = torch.stack(
        waveform_length_list,
        dim=0
    ).to(dtype=torch.long)

    # [B]
    label_lengths = torch.tensor(
        [labels.shape[0] for labels in phoneme_label_list],
        dtype=torch.long
    )

    # [sum(label_lengths)]
    concatenated_labels = torch.cat(
        phoneme_label_list,
        dim=0
    )

    return (
        waveform_batch,
        waveform_lengths,
        concatenated_labels,
        label_lengths
    )


def calculate_phoneme_mel_statistics(
    mel_spectrogram,
    phoneme_intervals,
    hop_length,
    sample_rate
):
    """
    Calculate the average Mel value of every Mel-frequency
    band during each phoneme interval.

    Parameters
    ----------
    mel_spectrogram : torch.Tensor
        Shape [T, n_mels]

    phoneme_intervals : list[dict]
        Output of load_phoneme_intervals().

    hop_length : int
        Mel spectrogram hop length in waveform samples.

    sample_rate : int
        Audio sample rate.

    Returns
    -------
    list[dict]
        Statistics for every phoneme occurrence.
    """

    time_step_seconds = (
        hop_length / sample_rate
    )

    phoneme_mel_statistics = []


    for interval in phoneme_intervals:

        phoneme = interval["phoneme"]
        start_time = interval["start_time"]
        end_time = interval["end_time"]


        #######################################################
        # Convert time interval → Mel-frame indices

        start_frame = int(
            np.ceil(start_time / time_step_seconds)
        )

        end_frame = int(
            np.ceil(end_time / time_step_seconds)
        )


        #######################################################
        # Make sure indices stay inside the spectrogram
        start_frame = max(
            0,
            min(start_frame, mel_spectrogram.shape[0])
        )

        end_frame = max(
            start_frame,
            min(end_frame, mel_spectrogram.shape[0])
        )


        #######################################################
        # Select all Mel frames belonging to this phoneme
        #
        # Shape:
        #
        # [number_of_phoneme_frames, 80]

        phoneme_mel = mel_spectrogram[
            start_frame:end_frame,
            :
        ]


        #######################################################
        # Average across TIME
        #
        # [N_frames, 80]
        #        ↓
        #      [80]

        if phoneme_mel.shape[0] > 0:

            average_mel_values = (
                phoneme_mel.mean(dim=0)
            )

            strongest_mel_bin = int(
                torch.argmax(
                    average_mel_values
                ).item()
            )

            strongest_average_value = float(
                average_mel_values[
                    strongest_mel_bin
                ].item()
            )

        else:

            average_mel_values = None
            strongest_mel_bin = None
            strongest_average_value = None


        #######################################################

        phoneme_mel_statistics.append({

            "phoneme": phoneme,

            "start_time": start_time,
            "end_time": end_time,

            "start_frame": start_frame,
            "end_frame": end_frame,

            "number_of_frames":
                end_frame - start_frame,

            "average_mel_values":
                average_mel_values,

            "strongest_mel_bin":
                strongest_mel_bin,

            "strongest_average_value":
                strongest_average_value
        })


    return phoneme_mel_statistics

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

def load_phoneme_intervals(phn_file_path: str,
                           sample_rate: int = 16000) -> list[dict]:
    """
    Load phoneme intervals from a TIMIT .PHN file.

    Each line of the PHN file has the form:

        start_sample end_sample phoneme

    Example:

        11240 12783 iy

    The sample positions are converted to seconds.

    Returns
    -------
    phoneme_intervals : list of dictionaries

        Example element:

        {
            "phoneme": "iy",
            "start_sample": 11240,
            "end_sample": 12783,
            "start_time": 0.7025,
            "end_time": 0.7989375
        }
    """

    phoneme_intervals = []

    with open(phn_file_path, "r") as phn_file:

        for line in phn_file:

            start_sample, end_sample, phoneme = (
                line.strip().split()
            )

            start_sample = int(start_sample)
            end_sample = int(end_sample)

            start_time = (
                start_sample / sample_rate
            )

            end_time = (
                end_sample / sample_rate
            )

            phoneme_intervals.append({
                "phoneme": phoneme,
                "start_sample": start_sample,
                "end_sample": end_sample,
                "start_time": start_time,
                "end_time": end_time
            })

    return phoneme_intervals


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
            fig.savefig(
                result_dir / f"figure_{i}.png", dpi=300, bbox_inches="tight")
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


def ctc_greedy_decode_ids( # for the inference and feature maps cell
    prediction_ids,
    blank_index=0
):
    """
    Collapse repeated CTC predictions and remove blank tokens.

    Example:
        [0, 5, 5, 0, 8, 8, 8, 3]

    becomes:
        [5, 8, 3]
    """

    decoded_ids = []
    previous_id = None

    for current_id in prediction_ids:

        if (
            current_id != blank_index
            and current_id != previous_id
        ):
            decoded_ids.append(current_id)

        previous_id = current_id

    return decoded_ids

def align_phoneme_sequences( # for inference and feature maps cells
    predicted,
    target
):
    """
    Align predicted and target phoneme sequences using
    Levenshtein dynamic programming.
    """

    num_target = len(target)
    num_predicted = len(predicted)

    distance = [
        [0] * (num_predicted + 1)
        for _ in range(num_target + 1)
    ]

    for target_index in range(num_target + 1):
        distance[target_index][0] = target_index

    for predicted_index in range(num_predicted + 1):
        distance[0][predicted_index] = predicted_index

    # ---------------------------------------------------------
    # Calculate edit-distance matrix
    # ---------------------------------------------------------

    for target_index in range(1, num_target + 1):

        for predicted_index in range(1, num_predicted + 1):

            target_phoneme = target[target_index - 1]
            predicted_phoneme = predicted[predicted_index - 1]

            substitution_cost = (
                0
                if target_phoneme == predicted_phoneme
                else 1
            )

            distance[target_index][predicted_index] = min(

                # deletion
                distance[target_index - 1][predicted_index] + 1,

                # insertion
                distance[target_index][predicted_index - 1] + 1,

                # match / substitution
                distance[target_index - 1][predicted_index - 1]
                + substitution_cost
            )

    # ---------------------------------------------------------
    # Backtracking
    # ---------------------------------------------------------

    aligned_targets = []
    aligned_predictions = []

    substitutions = 0
    insertions = 0
    deletions = 0
    correct = 0

    target_index = num_target
    predicted_index = num_predicted

    while target_index > 0 or predicted_index > 0:

        # Match or substitution
        if target_index > 0 and predicted_index > 0:

            target_phoneme = target[target_index - 1]
            predicted_phoneme = predicted[predicted_index - 1]

            substitution_cost = (
                0
                if target_phoneme == predicted_phoneme
                else 1
            )

            if (
                distance[target_index][predicted_index]
                ==
                distance[target_index - 1][predicted_index - 1]
                + substitution_cost
            ):

                aligned_targets.append(target_phoneme)
                aligned_predictions.append(predicted_phoneme)

                if substitution_cost == 0:
                    correct += 1
                else:
                    substitutions += 1

                target_index -= 1
                predicted_index -= 1

                continue

        # Deletion
        if (
            target_index > 0
            and
            distance[target_index][predicted_index]
            ==
            distance[target_index - 1][predicted_index] + 1
        ):

            aligned_targets.append(
                target[target_index - 1]
            )

            aligned_predictions.append("-")

            deletions += 1
            target_index -= 1

            continue

        # Insertion
        aligned_targets.append("-")

        aligned_predictions.append(
            predicted[predicted_index - 1]
        )

        insertions += 1
        predicted_index -= 1

    aligned_targets.reverse()
    aligned_predictions.reverse()

    operation_counts = {
        "correct": correct,
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "edit_distance": (
            substitutions
            + insertions
            + deletions
        )
    }

    return (
        aligned_predictions,
        aligned_targets,
        operation_counts
    )

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

def sweep(current_scale, remaining_depth, upper_bound=16):
    """Generates a large number of possible step_scale configurations and sweeping over them"""
    if remaining_depth == 0:
        return [[]]

    results = []

    for i in range(1, upper_bound+1):

        if current_scale*i > upper_bound+1:
            break

        results += [
            [current_scale*i, *ans]
            for ans in sweep(
                current_scale*i,
                remaining_depth - 1,
                upper_bound=upper_bound
            )
        ]

    return results


def compute_batch_per_from_log_probs__inefficient(log_probs,
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
        pred_ids = pred_ids_batch[b,
                                  :current_input_length].detach().cpu().tolist()
        decoded_pred = ctc_greedy_decode(pred_ids, blank_idx=blank_idx)

        # target sequence for one utterance
        target_seq = targets[target_offset:target_offset +
                             current_target_length].detach().cpu().tolist()
        target_offset += current_target_length

        # edit distance
        dist = edit_distance(decoded_pred, target_seq)

        total_edit_distance += dist
        total_target_length += current_target_length

    batch_per = total_edit_distance / \
        total_target_length if total_target_length > 0 else 0.0

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

def compute_batch_per_from_log_probs(
    log_probs,
    output_lengths, # contains the lengths of the unpadded temporal lengths of each sample produced as output by the model
    targets,
    target_lengths,
    blank_idx=0
):
    """
    Compute PER statistics for one batch.

    All GPU-to-CPU transfers are performed once per batch rather than
    once per utterance.
    """

    # Argmax does not need gradients.
    with torch.no_grad():
        pred_ids_batch = torch.argmax(
            log_probs.detach(),
            dim=-1
        )

    # Transfer everything to CPU once.
    pred_ids_batch = pred_ids_batch.cpu()
    output_lengths = output_lengths.cpu().tolist()
    targets = targets.detach().cpu().tolist()
    target_lengths = target_lengths.cpu().tolist()

    total_edit_distance = 0 # the edit distance of the whole batch
    total_target_length = 0 # the target lengths of the whole batch
    target_offset = 0 # target offset to be able to follow where the target of the next sample begins during iteration

    for sample, (current_input_length, current_target_length) in enumerate(
        zip(output_lengths, target_lengths)
    ):
        # Prediction is already on CPU.
        pred_ids = pred_ids_batch[
            sample, :current_input_length
        ].tolist()

        # collapse repeated phonemes and delete blanks from the prediction sequence
        decoded_pred = ctc_greedy_decode(
            pred_ids,
            blank_idx=blank_idx
        )

        # define the target sequence of the current sample in the batch
        target_seq = targets[
            target_offset:
            target_offset + current_target_length
        ]

        # update the target offset for the next sample
        target_offset += current_target_length

        # calculate the edit distance of the current sample (edit distance -> (S+D+I))
        # dist = edit_distance( # own implementation
        #    decoded_pred,
        #    target_seq
        #)

        # use the optimized editdistance.eval() implementation for the Levenshtein distance
        dist = editdistance.eval(
            decoded_pred,
            target_seq
        )

        # update the edit distance and target lengths values for the batch PER calculation
        total_edit_distance += dist
        total_target_length += current_target_length

    # calculate the Batch per based on the sum of the edit distances and the sum of the target lengths
    batch_per = (
        total_edit_distance / total_target_length
        if total_target_length > 0
        else 0.0
    )

    return (
        float(batch_per),
        int(total_edit_distance),
        int(total_target_length)
    )


def log_message(msg, file_handle):
    print(msg)
    file_handle.write(msg + "\n")
