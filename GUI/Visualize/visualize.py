import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np

example_wav_path = r"/Users/atis/Desktop/MasterArbeit/Code/TIMIT/data/TRAIN/DR1/FCJF0/SA2.WAV"
example_pho_path = r"/Users/atis/Desktop/MasterArbeit/Code/TIMIT/data/TRAIN/DR1/FCJF0/SA1.PHN"

def load_audio(audio_path: str) -> tuple:
    """Loading the audio file with pytorch into a tensor
    
    Paramaters
    ----------
    audio_path: str
        the path to the audio file

    Returns
    -------
    tuple:
        0. element -> waveform with type pytorch.Tensor
        1. sample_rate -> the sample rate, used for creating the .wav file, type 'int'
    """
    waveform, sample_rate = torchaudio.load(audio_path)

    return waveform, sample_rate

def plot_waveform(waveform_tensor: torch.Tensor,
                  sample_rate: int) -> None:
    """Plots the waveform according to the time axis
    
    Parameters
    ----------
    waveform_tensor: torch.Tensor
        the tensor containint the recorded amplitudes of the .wav file
    
    sample_rate: int
        the sample rate used for creating the .wav file

    Returns
    -------
    None
    """
    # create the time axis for the current audio file (how long the audio file is)
    time_axis = np.arange(waveform_tensor.shape[1]) / sample_rate
    
    # take first channel
    waveform_1d = waveform_tensor[0, :]

    # plot the waveform
    plt.plot(time_axis, waveform_1d)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Waveform")
    plt.show()

if __name__ == "__main__":
    # load the audio file
    waveform_tensor, sample_rate = load_audio(example_wav_path)

    # plot the audio waveform
    plot_waveform(waveform_tensor=waveform_tensor,
                  sample_rate=sample_rate)