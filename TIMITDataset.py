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
                 transform: any = None,
                 apply_padding_on_time_domain: bool = True):
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

        apply_padding_on_time_domain: bool
            if True, then all the samples in the dataset will be saved into one big tensor, and all of them will be
            padded, before tranforming the data into spectrogram representation
        """

        print(f"{train_or_test} Dataset creation process...")
        self.df = pd.read_csv(csv_file)
        self.train_or_test = train_or_test
        self.root_dir = root_dir
        self.phoneme_to_idx_vocab = phoneme_to_idx_vocab
        self.transform = transform
        self.max_waveform_length = 0
        self.waveforms = []
        self.orig_waveform_lengths = []
        self.apply_padding_on_time_domain = apply_padding_on_time_domain
        self.phonemes = []

        # check if the dataframe contains only train or test data
        self.check_dataset_for_pure_set_type()

        # get the root directory of the dataset
        self.root_dir = os.path.join(self.root_dir, "TIMIT/data")

        ####### CACHE CREATION ######
        # create the cache directory if necessary for saving the time_domain_padded_tensors
        # FOR FREQUENCY DOMAIN PADDING NOT YET IMPLEMENTED
        print("Creating cache directory")
        self.create_cache_directory()

        # create the cache filenames for checking if they exist
        self.cache_file = os.path.join(
            self.cache_dir, f"{train_or_test}_cache_statistics.pt")

        print("Creating the samples dictionary...")
        self.create_samples_dictionary()

        # using time padding on the time domain waveforms
        if apply_padding_on_time_domain:
            if os.path.exists(self.cache_file):
                print("Loading Cache dataset...")
                self.load_cache()

            else:
                # creates the self.samples dictionary
                print("Load the waveforms...")
                self.load_all_waveforms_from_samples_dictionary()

                print("Pad the time domain tensors...")
                self.pad_all_time_domain_tensors()

                print(
                    "Stack the padded waveforms and save their original lengths...")
                self.stack_waveforms_and_list_original_waveform_lengths()

                print("Calculate the mean and std values...")
                if train_or_test == "TRAIN":
                    self.compute_mean_and_std_values_for_stacked_waveforms()
                elif train_or_test == "TEST":
                    self.load_train_mean_and_std()

                print("Normalize the stacked waveform tensor...")
                self.normalize_waveforms()

                print(
                    f"Save the processed data into {train_or_test}_cache_statistics.pt file...")
                self.save_cache()

            print(f"{train_or_test} dataset created ")

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
        if self.apply_padding_on_time_domain:
            return self.prepare_one_sample_time_domain(idx=idx)
        else:
            return self.prepare_one_sample_frequency_domain(idx=idx)

    def create_cache_directory(self) -> None:
        """Creates the cache directory for faster loading

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.cache_dir = os.path.join(self.root_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def save_cache(self) -> None:
        """Saves the padded and stacked time domain tensors, the phoneme labels and the original waveform lengths into a .pt file

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        cache = {
            "waveforms": self.waveforms,
            "orig_waveform_lengths": self.orig_waveform_lengths,
            "phonemes": self.phonemes,
            "mean": self.mean,
            "std": self.std,
        }

        torch.save(cache, self.cache_file)

        print(f"Saved Cache dataset to {self.cache_file}")

    def normalize_waveforms(self) -> None:
        """Normalizes the stacked and padded audio waveform tensor with the calculated mean and std values

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.waveforms = (self.waveforms - self.mean) / \
            (self.std + 1e-5)  # using 1e-5 because of SubsetSC

    def compute_mean_and_std_values_for_stacked_waveforms(self) -> None:
        """Computes the mean and std values on the padded, stacked time domain tensor

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.mean = self.waveforms.mean()
        self.std = self.waveforms.std()

        print(f"Dataset mean: {self.mean:.6f}")
        print(f"Dataset std: {self.std:.6f}")

    def stack_waveforms_and_list_original_waveform_lengths(self) -> None:
        """Stack all the time domain padded waveform tensors into one big tensor and
        creates a 2D tensor for original waveform lengths (Necessary information for CTC loss)

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # stack all the padded time domain waveforms
        self.waveforms = torch.stack(self.padded_waveforms, dim=0)

        # save the length of the original UNPADDED waveforms
        self.orig_waveform_lengths = torch.tensor(
            self.orig_waveform_lengths, dtype=torch.long)

    def load_train_mean_and_std(self) -> None:
        """for normalizing the test dataset, the train set's mean and std values should be used

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        train_cache_path = os.path.join(
            self.cache_dir,
            "TRAIN_cache_statistics.pt"
        )

        if not os.path.exists(train_cache_path):
            raise FileNotFoundError(
                f"Train cache not found: {train_cache_path}. "
                "Create the TRAIN dataset first so that mean and std can be computed."
            )

        train_cache_file = torch.load(train_cache_path, weights_only=False)

        self.mean = train_cache_file["mean"]
        self.std = train_cache_file["std"]

    def load_cache(self) -> None:
        """Loads the cache if the cache file exist

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        cache = torch.load(self.cache_file, weights_only=False)

        self.waveforms = cache["waveforms"]
        self.orig_waveform_lengths = cache["orig_waveform_lengths"]
        self.phonemes = cache["phonemes"]
        self.mean = cache["mean"]
        self.std = cache["std"]

    def pad_all_time_domain_tensors(self) -> None:
        """Pads the waveform tensors to the same length, which is the length of the longest audio file

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.padded_waveforms = []
        total_waveform_length = len(self.waveforms)

        for idx, waveform in enumerate(self.waveforms):

            current_length = waveform.shape[1]
            padding_amount = self.max_waveform_length - current_length

            padded_waveform = torch.nn.functional.pad(
                waveform,
                pad=(0, padding_amount)
            )

            self.padded_waveforms.append(padded_waveform)
            phoneme_label = self.load_phoneme_labels(idx=idx)
            # save all the phonemes in a list
            self.phonemes.append(phoneme_label)

            if (idx + 1) % 100 == 0 or (idx + 1) == total_waveform_length:
                print(f"{idx + 1}/{total_waveform_length} waveforms padded")

    def create_samples_dictionary(self) -> None:
        """Processes the given dataset and creates the self.samples list where each element represents a sample as a dictionary in 
        the following form

        self.samples = {
            'base_name': utterance_id,
            'audio_path': audio_path,
            'phoneme_path': phoneme_path
        }

        Paramaters
        ----------
        None

        Returns
        -------
        None
        """
        # create a base name column once
        self.df["utterance_id"] = self.df["path_from_data_dir"].str.rsplit(
            ".", n=1).str[0]
        # self.df["base_name"] = self.df["filename"].str.split(".").str[0] not unique enough because if dialects

        # build the a sample list where each element is a dictionary.
        # keys -> base_name, audio_path, phoneme_path
        # values -> base name, path to the audio file, path to the phoneme file
        # build the sample list once
        self.samples = []
        grouped = self.df.groupby("utterance_id")
        # grouped = self.df.groupby("base_name")

        for utterance_id, group in grouped:
            audio_rows = group[(
                group["filename"].str.endswith(".WAV", na=False))]
            phoneme_rows = group[group["filename"].str.endswith(
                ".PHN", na=False)]

            # check for anomalies in the dataset
            if len(audio_rows) != 1 or len(phoneme_rows) != 1:
                continue

            # define the full path to the audio files
            audio_path = os.path.join(
                self.root_dir, audio_rows.iloc[0]["path_from_data_dir"])
            phoneme_path = os.path.join(
                self.root_dir, phoneme_rows.iloc[0]["path_from_data_dir"])

            # adding the dictionaries to the samples list
            self.samples.append({
                "base_name": utterance_id,
                "audio_path": audio_path,
                "phoneme_path": phoneme_path
            })

    def load_all_waveforms_from_samples_dictionary(self) -> None:
        """Loads the waveforms from the self.samples dictionary and saves them into pytorch tensors

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        # get the the number of samples of the longest audio
        for sample in self.samples:
            audio_path = sample["audio_path"]

            waveform, self.sample_rate = torchaudio.load(audio_path)

            # save the waveform to the list
            self.waveforms.append(waveform)

            # save the original length of the waveform to the list
            waveform_length = waveform.shape[1]
            self.orig_waveform_lengths.append(waveform_length)

            self.max_waveform_length = (
                waveform.shape[1]
                if waveform.shape[1] > self.max_waveform_length
                else self.max_waveform_length
            )

    def check_dataset_for_pure_set_type(self) -> None:
        """Checks if the dataset loaded into the pandas dataframe contains only TRAIN or TEST datas,
        depending on the situation

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if (self.df["test_or_train"] == self.train_or_test).all():
            print(f"All samples are from the {self.train_or_test} ")
        else:
            print(
                f"Filtering dataframe to only include {self.train_or_test} samples")
            self.df = self.df.loc[self.df["test_or_train"]
                                  == self.train_or_test]

        print(f"Checked -> whole dataset belongs to {self.train_or_test}")

    def prepare_one_sample_frequency_domain(self,
                                            idx: int) -> tuple:
        """
        Returns one sample from the dataset if apply_padding_on_time_domain is FALSE

        !!! In this case the PADDING is happening in the collate_fn_ctc function on the mel spectrogram features in the FREQUENCY DOMAIN !!!

        Parameters
        ----------
        idx: int
            the index of the sample to return

        Returns
        -------
        tuple: 2 elements
            0. element -> mel_spec: mel spectrogram of the waveform
            1. element -> phoneme_labels: the phoneme labels of the original waveform
        """
        # get the base string of the filename
        my_sample = self.samples[idx]

        # build the names of the audio and phoneme files
        path_audio_file = my_sample["audio_path"]

        # turning the audio file into a tensor
        if os.path.exists(path_audio_file):
            waveform, sample_rate = torchaudio.load(path_audio_file)
        else:
            raise Exception(f"Audio file {path_audio_file} does no exist")

        # load the phoneme labels
        phoneme_labels = self.load_phoneme_labels(idx=idx)

        # applying the melspectrogram transform on the samples
        if self.transform is not None:
            mel_spec = self.transform(waveform)
            mel_spec = mel_spec.squeeze(0).transpose(
                0, 1)  # (T, nmels) -> T = number of windows
            # stabilizes training and compresses dynamic range
            mel_spec = torch.log(mel_spec + 1e-6)
        else:
            mel_spec = waveform
        return mel_spec, phoneme_labels

    def prepare_one_sample_time_domain(self,
                                       idx: int) -> tuple:
        """
        Returns one sample from the dataset if apply_padding_on_time_domain is TRUE

        !!! In this case the PADDING happens here in the dataset on the waveform features in the TIME DOMAIN !!!!

        Parameters
        ----------
        idx: int
            the index of the sample to return
        Returns
        -------
        tuple: 3 elements
            0. element -> the padded waveform tensor with shape [1, self.max_waveform_length]
            1. element -> the length of the original waveform tensor WITHOUT PADDING
            2. element -> list of the target phoneme labels in integer form
        """

        # get the padded waveform tensor of the sample
        waveform_tensor = self.waveforms[idx, :, :]
        original_waveform_length = self.orig_waveform_lengths[idx]
        phoneme_labels = self.phonemes[idx]

        return waveform_tensor, original_waveform_length, phoneme_labels

    def load_phoneme_labels(self,
                            idx: int) -> torch.Tensor:
        """
        Loads the phoneme labels

        Parameters
        ----------
        idx: int
            the index of the dictionary, which holds the phoneme path to the current sample

        Returns
        -------
        list:
            list if the phonemes of the current sample in integer format from the dictionary
        """

        # get the phoneme labels
        path_phoneme_file = self.samples[idx]["phoneme_path"]

        phoneme_ids = []

        # turning the phonemes into a list of integer numbers
        if os.path.exists(path_phoneme_file):
            with open(path_phoneme_file, "r") as f:
                for line in f:
                    phoneme_str = (line.strip().split())[-1]
                    try:
                        phoneme_ids.append(
                            self.phoneme_to_idx_vocab[phoneme_str])
                    except KeyError as k:
                        raise KeyError(
                            f"No such key in the vocabulary as {phoneme_str}")
        else:
            raise Exception(f"Phoneme file {path_phoneme_file} does not exist")

        # turning the list of integers into a 1D tensor of integers
        phoneme_labels = torch.tensor(phoneme_ids, dtype=torch.long)

        return phoneme_labels
