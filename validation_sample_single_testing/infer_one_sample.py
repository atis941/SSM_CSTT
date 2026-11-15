# -----------------------------------------------------------------------------------------------------------------------------
# ##################################################### LIBRARY IMPORTS #######################################################
# -----------------------------------------------------------------------------------------------------------------------------
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchaudio
import json
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

from TIMITDataset import TIMITDataset
import random
from utils import (collate_fn_ctc, save_training_result, collate_fn_ctc_time_domain_padding, 
                   collate_fn_ctc_time_domain_features_during_training, compute_batch_per_from_log_probs, decode_batch_to_phonemes, log_message, 
                   load_phoneme_intervals, calculate_phoneme_mel_statistics, ctc_greedy_decode_ids, 
                   align_phoneme_sequences, check_timit_boundaries)
from functools import partial # for the new collate function
import yaml
from train_eval_utils import (apply_scheduler, calculate_PER_for_batch, calculate_and_update_epoch_loss, 
                              check_gradients_for_inf, create_checkpoint, evaluate_one_batch, 
                              log_batch_loss_and_PER, log_early_stopping, log_epoch_info, 
                              log_training_time, qualitative_predictions, restore_best_model, 
                              stop_early, train_one_batch, calculate_test_dataset_metrics,
                              append_test_results, save_test_results_to_csv, count_invalid_ctc_samples,
                              log_test_phoneme_predictions, initialize_test_predictions_log, print_cuda_memory)
from Decoder import CTCDecoder
from Upsampler import Upsampler
from SSM_TU.patches.scheduler import CosineAnnealingWarmupRestarts
from Augments import augments_weak


from validation_sample_single_testing.utils import load_vocabulary, select_model, create_mel_transform_object
import os
from gui import InferenceVisualizationGUI
from PyQt6.QtWidgets import QApplication

seed = 42 # define the seed for pytorch repeatability
print(f"✅ Libraries are imported without error")
# --------------------------------------------------------------------------------------------------------------------------
# ##################################################### DEFINE PATHS #######################################################
# --------------------------------------------------------------------------------------------------------------------------
model_path = ""
test_data_csv = r"/Users/atis/Desktop/MasterArbeit/Code/test_data.csv"
#train_data_csv = r"/Users/atis/Desktop/MasterArbeit/Code/train_data.csv"
project_root_dir = os.getcwd()
device = torch.device("cpu")

print(f"✅ model_path: {model_path}")
print(f"✅ test_data_csv: {test_data_csv}")
print(f"✅ project_root_dir: {project_root_dir}")
print(f"✅ device: {device}")
# --------------------------------------------------------------------------------------------------------------------------
# ##################################################### LOAD VOCABULARY ####################################################
# --------------------------------------------------------------------------------------------------------------------------
phoneme_set_version = 61
vocabulary_dir = r"/Users/atis/Desktop/MasterArbeit/Code/vocabulary"

phoneme_to_idx_vocab, idx_to_phoneme_vocab = load_vocabulary(phoneme_set_version=phoneme_set_version,
                                                             vocabulary_dir=vocabulary_dir)
print(f"✅ vocabulary version: {phoneme_set_version}")
# --------------------------------------------------------------------------------------------------------------------------
# ##################################################### MEL OR TEMPORAL ####################################################
# --------------------------------------------------------------------------------------------------------------------------
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

use_mel_transform = False

# The Mel spectrogram does the following:
# 1. Split signal into windows
# 2. Apply Fourier Transform (FFT)
# 3. Convert frequencies to Mel scale
# 4. Output energy per Mel band
mel_transform_n_mels = 80 # number of features in each timing window after transformation
mel_transform_nfft = 300
mel_transform_hop_length = 160
mel_transform_win_length = 300
mel_transform_sample_rate = 16000

# --------------------------------------------------------------------------------------------------------------------------
# ##################################################### MODEL DEFINITION ###################################################
# --------------------------------------------------------------------------------------------------------------------------
yaml_name = "S2S_Edge_small.yaml"
yaml_path = os.path.join("/Users/atis/Desktop/MasterArbeit/Code/S2S_model_descreptions", yaml_name)

ssm_model, rzoh_probabilities = select_model(yaml_path=yaml_path,
                                             use_mel_transform=use_mel_transform,
                                             ssm_vocab=phoneme_to_idx_vocab,
                                             mel_transform_n_mels=mel_transform_n_mels,
                                             device=device)

print(f"✅ model name: {yaml_name}")
# --------------------------------------------------------------------------------------------------------------------------
# ################################################## TEST DATASET DEFINITION ###############################################
# --------------------------------------------------------------------------------------------------------------------------
if use_mel_transform:
    mel_transform = create_mel_transform_object(mel_transform_sample_rate=mel_transform_sample_rate,
                                                mel_trasform_nfft=mel_transform_nfft, # FFT window size => how many samples are used for frequency analysis
                                                mel_transform_hop_length=mel_transform_hop_length, # step size between windows => at 16kHz a hop length of 160 means a stepsize of 10ms
                                                mel_transform_win_length=mel_transform_win_length, # Actual window size applied before FFT => at 16kHz a window size of 400 means 25ms window length
                                                mel_transform_n_mels=mel_transform_n_mels) # Number of frequency bins after Mel conversion => number of frequency elements in each window AFTER conversion
    # mel spectrogram settings
    mel_transform_dict = {
        "mel_transform_n_mels": mel_transform_n_mels,
        "mel_transform_nfft": mel_transform_nfft,
        "mel_transform_hop_length": mel_transform_hop_length,
        "mel_transform_win_length": mel_transform_win_length,
        "mel_transform_sample_rate": mel_transform_sample_rate,
    }
    print(f"✅ Mel spectrogram transformation is used with the following settings: {mel_transform_dict}")
else: 
    mel_transform_dict = None
    print(f"✅ Mel spectrogram transformation is  NOT used")

batch_size = 8

collate_function = ""
padding_after_mel_transform = False
apply_rzoh = True if rzoh_probabilities is not None else False

if padding_after_mel_transform:
    dataloader_collate_fn = collate_fn_ctc
    collate_function = "collate_fn_ctc"
else:
    if not use_mel_transform:
        dataloader_collate_fn = (
            collate_fn_ctc_time_domain_features_during_training
        )
        collate_function = (
            "collate_fn_ctc_time_domain_features_during_training"
        )
        mel_transform_dict = None
        print(f"✅ Mel spectrogram transformation is NOT used")
    else:
        dataloader_collate_fn = partial(
            collate_fn_ctc_time_domain_padding,
            mel_transform=mel_transform,
            hop_length=mel_transform_hop_length,
            apply_rzoh=apply_rzoh
        )

        collate_function = "collate_fn_ctc_time_domain_padding"
        print(f"✅ Mel spectrogram transformation is used")

# train_dataset = TIMITDataset(
#     csv_file=train_data_csv,
#     train_or_test="TRAIN",
#     root_dir=project_root_dir,
#     phoneme_to_idx_vocab=phoneme_to_idx_vocab,
#     transform=None,
# )

# Test dataset
val_dataset = TIMITDataset(
    csv_file=test_data_csv,
    train_or_test="TEST",
    root_dir=project_root_dir,
    phoneme_to_idx_vocab=phoneme_to_idx_vocab,
    transform=None,
)
print(f"✅ Dataset created successfully")

val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=dataloader_collate_fn
)
print(f"✅ Dataloader created successfully")
# --------------------------------------------------------------------------------------------------------------------------
# ################################################## DECODER DEFINITION ####################################################
# --------------------------------------------------------------------------------------------------------------------------
blank_idx = 0
beam_width = 5
top_paths = 1
decoding_method = "greedy" # Options: 1. greedy
                           #          2. beam_search
ctc_decoder = CTCDecoder(blank_idx=blank_idx,
                         beam_width=beam_width,
                         top_paths=top_paths,
                         method=decoding_method)

print(f"✅ Decoder beam width: {beam_width}")
print(f"✅ Decoder top paths: {top_paths}")
print(f"✅ Decoder decoding method: {decoding_method}")


# --------------------------------------------------------------------------------------------------------------------------
# ################################################## LOAD MODEL ####################################################
# --------------------------------------------------------------------------------------------------------------------------
trained_models_dir = os.path.join(project_root_dir, "S2S_Edge_trained_models")
model_name = "S2S_edge_small_augmentation_50 epochs"
model_weights_path = os.path.join(trained_models_dir,model_name, "ssm_model_state_dict.pt")

# load the weights into a dictionary
state_dict = torch.load(
    model_weights_path,
    map_location=device,
    weights_only=True
)

# load the weights into the model
ssm_model.load_state_dict(state_dict)
print(f"✅ Model successfully loaded: {model_name}")


# --------------------------------------------------------------------------------------------------------------------------
# ################################################# INFER ONE SAMPLE #######################################################
# --------------------------------------------------------------------------------------------------------------------------
sample_idx = 242

# 1. Retrieve the original dataset sample.
sample = val_dataset[sample_idx]

# 2. Apply the same collate function as validation.
# A batch containing only one sample.
(   model_input,
    input_lengths,
    target_ids,
    target_lengths
) = dataloader_collate_fn([sample])

print("model_input shape:", model_input.shape)
print("input_lengths shape:", input_lengths.shape)
print("target_ids:", target_ids.shape)
print("target_lengths:",target_lengths.shape)

model_input = model_input.to(device)

# No subsampling for this first inference.
ssm_model.set_step_scale([1] * len(ssm_model.ssm_layers))

# 3. Inference.
ssm_model.eval()

with torch.inference_mode():
    model_output = ssm_model(model_input)

    # Use the model's output-length method
    # if available, particularly for subsampling.
    if hasattr(ssm_model, "get_output_lengths"):
        output_lengths = ssm_model.get_output_lengths(
            input_lengths
        )
        output_length = int(output_lengths[0].item())
    else:
        if int(input_lengths[0]) != model_output.shape[1]:
            raise ValueError(
                "The model changes temporal length, "
                "but no output-length calculation is available."
            )
        output_length = int(input_lengths[0])

    # Extract classifier logits for the selected sample.
    # [1, T, C] -> [T, C]
    sample_logits = model_output[0, :output_length].detach().cpu()

    # Convert logits into probabilities.
    sample_probabilities = torch.softmax(
        sample_logits,
        dim=-1
    ).numpy()

    T, num_classes = sample_probabilities.shape

    print(f"Classifier logits shape: {sample_logits.shape}")
    print(f"Classifier probabilities shape: {sample_probabilities.shape}")



# --------------------------------------------------------------------------------------------------------------------------
# ########################################### RECORDING INFORMATION ########################################################
# --------------------------------------------------------------------------------------------------------------------------

sample_name = val_dataset.samples[sample_idx]["base_name"]

phn_file_path = val_dataset.samples[sample_idx]["phoneme_path"]

phoneme_intervals = load_phoneme_intervals(
    phn_file_path,
    sample_rate=mel_transform_sample_rate
)

# --------------------------------------------------------------------------------------------------------------------------
# ################################################## TIME AXIS #############################################################
# --------------------------------------------------------------------------------------------------------------------------

# No additional SSM subsampling for this first version.
# For Mel features, one time step corresponds to one Mel hop.
# For raw time-domain features, one time step corresponds
# to one audio sample, assuming the model preserves time.

if use_mel_transform:
    base_stride = mel_transform_hop_length
else:
    base_stride = 1

time_step_seconds = base_stride / mel_transform_sample_rate

time_edges = np.arange(T + 1) * time_step_seconds

# --------------------------------------------------------------------------------------------------------------------------
# ################################################# CLASS LABELS ###########################################################
# --------------------------------------------------------------------------------------------------------------------------

class_labels = [
    idx_to_phoneme_vocab[i]
    for i in range(num_classes)
]

# Original recording duration
original_num_samples = int(sample[1].item())

recording_duration = original_num_samples / mel_transform_sample_rate

# Duration represented by the model output
model_output_duration = T * time_step_seconds

print(f"Recording duration: {recording_duration:.4f} s")
print(f"Model output duration: {model_output_duration:.4f} s")
print(f"Number of output time steps: {T}")
print(f"Number of original audio samples: {original_num_samples}")

app = QApplication.instance()

if app is None:
    app = QApplication(sys.argv)

window = InferenceVisualizationGUI(
    sample_probabilities=sample_probabilities,
    phoneme_intervals=phoneme_intervals,
    sample_name=sample_name,
    class_labels=class_labels,
    recording_duration=recording_duration,
    blank_idx=blank_idx
)

window.show()

app.exec()









