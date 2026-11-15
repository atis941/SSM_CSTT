import os
import json
import torch.nn as nn
import yaml
from SSM_model_V4 import SSMPhonemeModel as SSMPhonemeModelV4
import torchaudio

def load_vocabulary(phoneme_set_version: int,
                    vocabulary_dir: str) -> list[dict, dict]:
    """
    Loads the relevant vocabulary

    Paramaters
    ----------
    phoneme_set_version: int
        decides whether use the original 61 phoneme vocabulary or the reduced version with 39 phonemes

    vocabulary_dir: str
        path to the directory of the vocabulary

    Returns
    -------
    list:
        1. element -> phoneme_to_idx_vocab: dict
        2. element -> idx_to_phoneme_vocab: dict

    """
    # ---------------------------------------------------------
    # Validate phoneme set
    # ---------------------------------------------------------
    if phoneme_set_version not in {39, 61}:
        raise ValueError(
            f"Unsupported phoneme_set_version={phoneme_set_version}. "
            "Expected either 39 or 61."
        )

    # ---------------------------------------------------------
    # Define vocabulary filenames
    # ---------------------------------------------------------
    phoneme_to_idx_vocab_filename = (
        f"phoneme_to_idx_{phoneme_set_version}.json"
    )

    idx_to_phoneme_vocab_filename = (
        f"idx_to_phoneme_{phoneme_set_version}.json"
    )

    phoneme_to_idx_vocab_path = os.path.join(
        vocabulary_dir,
        phoneme_to_idx_vocab_filename
    )


    idx_to_phoneme_vocab_path = os.path.join(
        vocabulary_dir,
        idx_to_phoneme_vocab_filename
    )

    # ---------------------------------------------------------
    # Check if the vocabulary files exist
    # ---------------------------------------------------------
    if not os.path.exists(phoneme_to_idx_vocab_path):
        raise FileNotFoundError(
            f"Vocabulary file not found:\n{phoneme_to_idx_vocab_path}"
        )

    if not os.path.exists(idx_to_phoneme_vocab_path):
        raise FileNotFoundError(
            f"Vocabulary file not found:\n{idx_to_phoneme_vocab_path}"
        )

    # ---------------------------------------------------------
    # Load phoneme -> index vocabulary
    # ---------------------------------------------------------
    with open(phoneme_to_idx_vocab_path, "r", encoding="utf-8") as file:
        phoneme_to_idx_vocab = json.load(file)

    # ---------------------------------------------------------
    # Load index -> phoneme vocabulary
    # ---------------------------------------------------------
    with open(idx_to_phoneme_vocab_path, "r", encoding="utf-8") as file:
        idx_to_phoneme_vocab = json.load(file)

        # JSON stores dictionary keys as strings, therefore convert them back to integers.
        idx_to_phoneme_vocab = {
            int(idx): phoneme
            for idx, phoneme in idx_to_phoneme_vocab.items()
        }

    print(
        f"Successfully loaded the {phoneme_set_version}-phoneme vocabulary.\n"
        f"Vocabulary size (including CTC blank): {len(phoneme_to_idx_vocab)}"
    )

    return [phoneme_to_idx_vocab, idx_to_phoneme_vocab]

def select_model(yaml_path: str,
                 use_mel_transform: bool,
                 ssm_vocab: dict,
                 mel_transform_n_mels: int,
                 device: any) -> nn.Module:
    """Selects the model architecture
    
    Parameters
    ----------
    yaml_path: str
        the path to the yaml file where the model architceture is defined

    use_mel_transform: bool
        decides whether to load a model where mel transform is used or not
    
    ssm_vocab: dict
        the used vocabulary
    
    mel_transform_n_mels: int
        the number of features in each timing window

    device: any
        the used device during inference
    
    Returns
    -------
    tuple:
        nn.Module:  
            the created S2S classifier model
        list[float]
            the list of the used rzoh_probabilities during training
    """
    ssm_vocab_size = len(ssm_vocab)
    ssm_dropout = 0.0
    linear_dropout = 0.2 # best -> 0.2

    # decide if use the configuration of supervisors from S_edge classification task
    import_model_from_S_Edge_yaml = True

    if use_mel_transform:
        d_in_first_layer = mel_transform_n_mels
    else:
        d_in_first_layer = 1

    yaml_path = os.path.join(yaml_path)
    
    with open(yaml_path, "r") as file:
        model_config = yaml.safe_load(file)
        
        d_out = model_config["output_sizes"].copy()
        d_state = model_config["hidden_sizes"].copy()  
        
        trainable_SkipLayer = model_config["trainable_skip_connections"]
        # if trainable_Skiplayer is false then d_in and d_out must have to look the same -> mel_transform_n_mels must be changed
        if not trainable_SkipLayer and use_mel_transform:
            mel_transform_n_mels = d_out[0]
            d_in_first_layer = mel_transform_n_mels

        d_in = [d_in_first_layer, *d_out[:-1]]

        input_bias = model_config["input_bias"]
        output_bias = model_config["output_bias"]
        complex_output = model_config["complex_output"]
        act = model_config["act"]
        norm = model_config["norm"]
        norm_type = model_config["norm_type"]
        stability = model_config["stability"]

        rzoh_probabilities = model_config["zeroOrderHoldRegularization"]
        if len(rzoh_probabilities) == 0: # rzoh prob values list shouldnt be empty
            rzoh_probabilities = None # None is handled inside the class definition

        B_C_init = model_config["B_C_init"]
        bias_init = model_config["bias_init"]
        ssm_num_layers = len(d_out)

        ssm_model = SSMPhonemeModelV4(d_state=d_state,
                                      d_in=d_in,
                                      d_out=d_out,
                                      num_of_ssm_layers=ssm_num_layers,
                                      vocab_size=ssm_vocab_size,
                                      ssm_dropout=ssm_dropout,
                                      input_bias = input_bias,
                                      output_bias = output_bias,
                                      complex_output = complex_output,
                                      B_C_init=B_C_init,
                                      bias_init = bias_init,
                                      stability=stability,
                                      linear_dropout=linear_dropout,
                                      norm=norm,
                                      norm_type=norm_type,
                                      act=act,
                                      trainable_SkipLayer=trainable_SkipLayer,
                                      rzoh_probabilities=rzoh_probabilities)
        ssm_model.to(device=device)

    return ssm_model, rzoh_probabilities

def create_mel_transform_object(mel_transform_sample_rate: int,
                                mel_transform_nfft: int,
                                mel_transform_hop_length: int,
                                mel_transform_win_length: int,
                                mel_transform_n_mels: int) -> any:
    """Creates the object of mel transformation
    
    Parameters
    ----------
    mel_transform_sample_rate: int
        The sample rate of the audiofile used during digitalization

    mel_transform_nfft: int
        FFT window size => how many samples are used for frequency analysis

    mel_transform_hop_length: int
        Step size between windows => at 16kHz a hop length of 160 means a stepsize of 10ms

    mel_transform_win_length: int
        Actual window size applied before FFT => at 16kHz a window size of 400 means 25ms window length

    mel_transform_n_mels: int
        Number of frequency bins after Mel conversion => number of frequency elements in each window AFTER conversion

    Returns
    -------
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=mel_transform_sample_rate,
        n_fft=mel_transform_nfft, # FFT window size => how many samples are used for frequency analysis
        hop_length=mel_transform_hop_length, # step size between windows => at 16kHz a hop length of 160 means a stepsize of 10ms
        win_length=mel_transform_win_length, # Actual window size applied before FFT => at 16kHz a window size of 400 means 25ms window length
        n_mels=mel_transform_n_mels # Number of frequency bins after Mel conversion => number of frequency elements in each window AFTER conversion
    )

    return mel_transform