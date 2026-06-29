from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import torch
import numpy as np

class WaveFormCanvas(FigureCanvas):
    def __init__(self,
                 height: int = None,
                 width: int = None):
        """
        Constructor
        
        Parameters
        ----------
        height: int
            the height of the figure

        width: int
            the width of the figure
        """
        if height is None and width is None:
            self.figure = Figure()
        else:
            self.figure = Figure(figsize=(width, height))

        self.ax = self.figure.add_subplot(111) # 111 -> rows, cols, index
        self.boundary_lines = [] # lines in case of zooming -> to be able to remove old lines and draw new lines
        
        super().__init__(self.figure)

    def plot_waveform(self, 
                      waveform_tensor: torch.Tensor, 
                      sample_rate: int) -> None:
        """
        Plots the given waveform
        
        Parameters
        ----------
        waveform_tensor: torch.Tensor
            the waveform saved in a pytorch tensor

        sample_rate: int
            the sample rate, used creating the audio file

        Returns
        -------
        None
        """
        self.ax.clear() # to clear the canvas before redraw something

        self.time_axis = np.arange(waveform_tensor.shape[1]) / sample_rate
        self.waveform_np = waveform_tensor[0, :].numpy()

        self.ax.plot(self.time_axis, self.waveform_np)
        self.ax.set_xlabel("Time [s]")
        self.ax.set_ylabel("Amplitude")
        self.ax.set_title("Audio as Waveform")

        self.draw()

    def zoom_to_time_range(self, 
                           start_time: float, 
                           end_time: float,
                           boundary_start: float = None,
                           boundary_end: float = None) -> None:
        """
        Zooms in on the given time interval in the Plot

        Parameters
        ----------
        start_time: float
            the starting time for the zooming (with padding)

        end_time: float
            the ending time for the zooming (with padding)

        boundary_start: float
            the real start time where the boundary is drawn (without padding)

        boundary_end: float
            the real end time where the boundary is drawn (without padding)

        Returns
        -------
        None
        """
        # remove old boundary lines from the plot
        for line in self.boundary_lines:
            line.remove()

        # emtpy the list of the lines
        self.boundary_lines.clear()

        # zoom with padding
        self.ax.set_xlim(start_time, end_time)

        # draw real phoneme boundaries
        if boundary_start is not None:
            start_line = self.ax.axvline(
                x = boundary_start,
                linewidth = 1.5,
                color="black",
                linestyle="--"
            )
            self.boundary_lines.append(start_line)

        if boundary_end is not None:
            end_line = self.ax.axvline(
                x = boundary_end,
                linewidth = 1.5,
                color="black",
                linestyle="--"
            )
            self.boundary_lines.append(end_line)

        self.draw()

    def reset_zoom(self) -> None:
        """
        Zooms out to the original view of the audio plot

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # remove boundary lines
        for line in self.boundary_lines:
            line.remove()
        
        # delete the boundary line objects from the list
        self.boundary_lines.clear()

        start_time = self.time_axis[0]
        end_time = self.time_axis[-1]
        
        self.ax.set_xlim(start_time, end_time)

        self.draw()