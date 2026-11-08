###### LIBRARY IMPORTS ######
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
from SSM_model_V1 import SSMPhonemeModel as SSMPhonemeModelV1
from SSM_model_V2 import SSMPhonemeModel as SSMPhonemeModelV2
from SSM_model_V3 import SSMPhonemeModel as SSMPhonemeModelV3
from SSM_model_V4 import SSMPhonemeModel as SSMPhonemeModelV4
from SSM_model_V5 import SSMPhonemeModel as SSMPhonemeModelV5
from S2T_CNN_time import S2T_CNN_time_V1
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
                              log_test_phoneme_predictions, initialize_test_predictions_log)
from Decoder import CTCDecoder
import os

seed = 42 # define the seed for pytorch repeatability

project_root_dir = os.getcwd() # get the root directory of the project
vocab_root_dir = project_root_dir
data_root_dir = project_root_dir
base_results_dir = os.path.join(project_root_dir, "results")
ssm_tu_root_dir = vocab_root_dir

vocabulary_dir = os.path.join(vocab_root_dir, "vocabulary")
train_data_csv = os.path.join(data_root_dir, "data", "train_data.csv")
test_data_csv = os.path.join(data_root_dir, "data", "test_data.csv")
ssm_tu_root_dir = os.path.join(ssm_tu_root_dir, "SSM_TU")

print(f"Project root directory: {project_root_dir}")
print(f"Data root directory: {data_root_dir}")
print(f"Vocabulary directory: {vocabulary_dir}")
print(f"SSM_TU_root_dir: {ssm_tu_root_dir}")
print(f"train_data.csv: {train_data_csv}")
print(f"test_data.csv: {test_data_csv}")