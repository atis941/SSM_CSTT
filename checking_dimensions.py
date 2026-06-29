from TIMITDataset import TIMITDataset
import os
from utils import build_phoneme_vocab, collate_fn_ctc, collate_fn_ctc_time_domain_padding
import torchaudio
from torch.utils.data import Dataset, DataLoader
import json
from functools import partial

####### DEFINITIONS ########
print("Definitions...")
project_root_dir = os.getcwd()  # get the root directory of the project
data_root_dir = project_root_dir

vocabulary_dir = os.path.join(project_root_dir, "vocabulary")
phoneme_to_idx_vocab_filename = "phoneme_to_idx.json"
idx_to_phoneme_vocab_filename = "idx_to_phoneme.json"
train_data_csv = os.path.join(data_root_dir, "train_data.csv")
test_data_csv = os.path.join(data_root_dir, "test_data.csv")


####### BUILDING the VOCABULARY #######
print("Building the Vocabulary...")
if (os.path.exists(os.path.join(vocabulary_dir, phoneme_to_idx_vocab_filename)) == False or
        os.path.exists(os.path.join(vocabulary_dir, idx_to_phoneme_vocab_filename)) == False):
    phoneme_to_idx_vocab, idx_to_phoneme_vocab = build_phoneme_vocab(root_dir=data_root_dir,
                                                                     save_dir=vocabulary_dir)
    print(
        f"The following vocabulary files were created:\n{phoneme_to_idx_vocab_filename}\n{idx_to_phoneme_vocab_filename}")
else:
    with open(os.path.join(vocabulary_dir, phoneme_to_idx_vocab_filename), "r") as file:
        phoneme_to_idx_vocab = json.load(file)
    with open(os.path.join(vocabulary_dir, idx_to_phoneme_vocab_filename), "r") as file:
        idx_to_phoneme_vocab = json.load(file)

        # convert keys back to int
        idx_to_phoneme_vocab = {
            int(k): v for k, v in idx_to_phoneme_vocab.items()}
    print(
        f"the vocabulary files already exist in the vocabulary directory:\n{phoneme_to_idx_vocab_filename}\n{idx_to_phoneme_vocab_filename}")


###### MEL TRANSFORM #######
print("Defining Mel transform...")
# number of features in each timing window after transformation
mel_transform_n_mels = 80
mel_transform_nfft = 400
mel_transform_hop_length = 160
mel_transform_win_length = 400
mel_transform_sample_rate = 16000
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=mel_transform_sample_rate,
    # FFT window size => how many samples are used for frequency analysis
    n_fft=mel_transform_nfft,
    # step size between windows => at 16kHz a hop length of 160 means a stepsize of 10ms
    hop_length=mel_transform_hop_length,
    # Actual window size applied before FFT => at 16kHz a window size of 400 means 25ms window length
    win_length=mel_transform_win_length,
    # Number of frequency bins after Mel conversion => number of frequency elements in each window AFTER conversion
    n_mels=mel_transform_n_mels
)

batch_size = 8

######## DATASET OBJECT ########
print("Creating Dataset object")
train_dataset = TIMITDataset(
    csv_file=train_data_csv,
    train_or_test="TRAIN",
    root_dir=project_root_dir,
    phoneme_to_idx_vocab=phoneme_to_idx_vocab,
    transform=mel_transform,
    apply_padding_on_time_domain=True
)

# creating the partial function to be able to set the collate_fn with more parameters in the dataloader
my_collate_fn = partial(
    collate_fn_ctc_time_domain_padding,
    mel_transform=mel_transform,
    hop_length=mel_transform_hop_length
)


######### DATALOADER OBJECT #########
print("Creating Dataloader object")
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=my_collate_fn
)

for mel_specs, orig_feature_lengths, concatenated_labels, label_lengths in train_dataloader:
    print("mel_specs")
    print(type(mel_specs))
    print(mel_specs.shape)
    print("--------------")
    # print("orig_feature_lengths")
    # print(type(orig_feature_lengths))
    # print(orig_feature_lengths.shape)
    # print("--------------")
    # print("concatenated_labels")
    # print(type(concatenated_labels))
    # print(concatenated_labels.shape)
    # print("--------------")
    # print("label_lengths")
    # print(type(label_lengths))
    # print(label_lengths.shape)
    # print("--------------")
