from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import torch
import numpy as np


class SpectrogramCanvas(FigureCanvas):
    def __init__(self,
                 height: int = None, 
                 width: int = None):
        """
        Constructor

        Parameters
        ----------
        height: int
            the height of the Figure
        
        width: int
            the width of the Figure
        """
        
        if height is None and width is None:
            self.figure = Figure()
        else:
            self.figure = Figure(figsize=(width, height))

        self.ax = self.figure.add_subplot(111)

        super().__init__(self.figure)

    def plot_spectrogram(self,
                         waveform_tensor: torch.Tensor,
                         sample_rate: int,
                         n_fft: int = 512,
                         hop_length: int = 160) -> None:
        """
        Plots the frequency Spectrogram of the waveform
        
        Excecutes a Shirt-Time-Fourier-Transform (STFT),y comouting many FFTs over small windows instead of
        one FFT over the whole signal. controlled by the parameters 'n_fft' and 'hop_length'


        Parameters
        ----------
        waveform_tensor: torch.Tensor
            the audio file in tensor format

        sample_rate: int
            the sample_rate used, while creating the audio file

        n_fft: int
            defines how many samples are used for one FFT window
            LARGE n_fft -> better frequency resolution, worse time resolution -> timing becomes blurrier
            SMALL n_fft -> better time resolution, worse frequency resolution
        
        hop_length: int
            defines how far the analysis window moves each step

        Returns
        -------
        None
        """

        self.ax.clear()

        waveform_1d = waveform_tensor[0, :]
        
        # computes a Short-Time-Fourier-Transform
        stft = torch.stft(input=waveform_1d,
                          n_fft=n_fft,
                          hop_length=hop_length,
                          return_complex=True) # return complex for magnitude and phase in complex form
        
        # convert STFT to magnitude -> how strong each frequency is, represented by the brightness
        spectrogram = torch.abs(stft)

        # convert to decibel scale
        spectrogram_db = 20 * torch.log10(spectrogram + 1e-10) # adding small offset because log10(0) would be undefined

        spectrogram_np = spectrogram_db.numpy()

        num_frames = spectrogram_np.shape[1]
        num_freq_bins = spectrogram_np.shape[0]
        
        # compute the axis range
        time_axis_end = num_frames * hop_length / sample_rate
        freq_axis_end = sample_rate / 2 # Nyquist frequency

        self.ax.imshow(
            spectrogram_np,
            origin="lower", # low frequencies appear at the bottom, and higher frequencies at the top
            aspect="auto", # allows automatic stretching to fit the figure
            extent=[0, time_axis_end, 0, freq_axis_end] # without this, axes would show pixel indices
        )

        self.ax.set_xlabel("Time [s]")
        self.ax.set_ylabel("Frequency [Hz]")
        self.ax.set_title("Spectrogram")

        self.draw()