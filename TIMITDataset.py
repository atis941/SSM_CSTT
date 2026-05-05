from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import torch
import torchaudio

class TIMITDataset(Dataset):
    def __init__(self,
                 csv_file: str,
                 train_or_test: str,
                 root_dir: str,
                 phoneme_to_idx_vocab: dict, 
                 transform: any = None):
        """Constructor of the class
        
        Parameters
        ----------
        csv_file: str
            the path to the csv file, containing the informations about the dataset

        train_or_test: str
            to select the train and test parts in the dataset, if they are present 
        
        root_dir : str
            the root directory of the dataset

        phoneme_to_idx_vocab: dict
            the vocabulary of the phonemes

        transform: any
            the transformation object to apply on the raw waveform => e.g. mel spectrogram
        """
        
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.phoneme_to_idx_vocab = phoneme_to_idx_vocab
        self.transform = transform


        # check if the dataframe contains only train or test data
        if (self.df["test_or_train"] == train_or_test).all():
            print(f"All samples are from the {train_or_test} ")
        else:
            print(f"Filtering dataframe to only include {train_or_test} samples")
            self.df = self.df.loc[self.df["test_or_train"] == train_or_test]

        # get the root directory of the dataset
        self.root_dir = os.path.join(self.root_dir, "TIMIT/data")
        
        # create a base name column once
        self.df["utterance_id"] = self.df["path_from_data_dir"].str.rsplit(".", n=1).str[0]
        # self.df["base_name"] = self.df["filename"].str.split(".").str[0] not unique enough because if dialects

        # build the a sample list where each element is a dictionary.
        # keys -> base_name, audio_path, phoneme_path
        # values -> base name, path to the audio file, path to the phoneme file
        # build the sample list once
        self.samples = []
        grouped = self.df.groupby("utterance_id")
        #grouped = self.df.groupby("base_name")


        for utterance_id, group in grouped:
            audio_rows = group[(group["filename"].str.endswith(".WAV", na=False))]
            phoneme_rows = group[group["filename"].str.endswith(".PHN", na=False)]

            # check for anomalies in the dataset
            if len(audio_rows) != 1 or len(phoneme_rows) != 1:
                continue
            
            # define the full path to the audio files
            audio_path = os.path.join(self.root_dir, audio_rows.iloc[0]["path_from_data_dir"])
            phoneme_path = os.path.join(self.root_dir, phoneme_rows.iloc[0]["path_from_data_dir"])

            # adding the dictionaries to the samples list
            self.samples.append({
                "base_name": utterance_id,
                "audio_path": audio_path,
                "phoneme_path": phoneme_path
            })



    def __len__(self) -> int:
        """Returns the length of the dataset
        
        Parameters
        ----------
        None
        
        Returns
        -------
        int: 
            the length of the dataset
        """
        return len(self.samples)


    def __getitem__(self, 
                    idx: int) -> tuple:
        """Returns one element from the dataset (feature, label)
        
        Parameters
        ----------
        idx: int
            the index of the element in the dataset
        
        Returns
        -------
        tuple:
            1. element: the melspectrogram transformed tensor of the audio file
            2. element: the phoneme sequence of the the audio file, as integers from the vocabulary
        """
        # get the base string of the filename
        my_sample = self.samples[idx]

        # build the names of the audio and phoneme files
        path_audio_file = my_sample["audio_path"]
        path_phoneme_file = my_sample["phoneme_path"]

        # turning the audio file into a tensor
        if os.path.exists(path_audio_file):
            waveform, sample_rate = torchaudio.load(path_audio_file)
        else:
            raise Exception(f"Audio file {path_audio_file} does no exist")

        phoneme_ids = []

        # turning the phonemes into a list of integer numbers
        if os.path.exists(path_phoneme_file):
            with open(path_phoneme_file, "r") as f:
                for line in f:
                    phoneme_str = (line.strip().split())[-1]
                    try:
                        phoneme_ids.append(self.phoneme_to_idx_vocab[phoneme_str])
                    except KeyError as k:
                        raise KeyError(f"No such key in the vocabulary as {phoneme_str}")
        else:
            raise Exception(f"Phoneme file {path_phoneme_file} does not exist")
        
        # turning the list of integers into a 1D tensor of integers
        phoneme_labels = torch.tensor(phoneme_ids, dtype=torch.long)
        
        # applying the melspectrogram transform on the samples
        if self.transform is not None:
            mel_spec = self.transform(waveform)
            mel_spec = mel_spec.squeeze(0).transpose(0,1) # (T, nmels) -> T = number of windows
            mel_spec = torch.log(mel_spec + 1e-6) #stabilizes training and compresses dynamic range
        else:
            mel_spec = waveform
        return mel_spec, phoneme_labels
        

        