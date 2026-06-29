from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import torch
import numpy as np

class PhonemeCanvas(FigureCanvas):
    def __init__(self,
                 height:int = None,
                 width: int = None):
        if height is None and width is None:
            self.figure = Figure()
        else:
            self.figure = Figure(figsize=(width, height))

        self.ax = self.figure.add_subplot(111)

        self.phonemes = []
        self.selected_index = None
        self.colors = ["tab:blue", "tab:green", "tab:red", "tab:orange"] # color palette for display

        super().__init__(self.figure)

    def plot_phonemes(self,
                      phonemes: list,
                      selected_index: int = None) -> None:
        """
        Plots the phonemes in the time axis

        Parameters
        ----------
        phonemes: list
            the phonemes list, holding the time intervals and phonemes

        selected_index: int
            upon zooming, the canvas will be redrawn selecting the phoneme with the given index
        
        Returns
        -------
        None
        """
        self.phonemes = phonemes
        self.selected_index = selected_index

        self.ax.clear()

        for i, (start_time, end_time, phoneme) in enumerate(self.phonemes):
            color = self.colors[i % len(self.colors)]
            is_selected = i == self.selected_index

            span_alpha = 0.75 if is_selected else 0.2
            text_color = "white" if is_selected else color
            text_weight = "bold" if is_selected else "normal"

            self.ax.axvspan(xmin=start_time,
                            xmax=end_time,
                            alpha=span_alpha,
                            color=color if is_selected else "white")
            self.ax.axvline(x=start_time,
                            linewidth=1.2 if is_selected else 0.5,
                            color=color)
            
            mid_time = (start_time + end_time) / 2 # to place the phoneme in the middle
            y_pos = 0.35 if i % 2 == 0 else 0.65 # alternate the y position so that close phonemes can also be displayed clearly

            self.ax.text(x = mid_time,
                         y = y_pos,
                         s = phoneme,
                         ha = "center",
                         va = "center",
                         fontsize = 9,
                         color=text_color,
                         fontweight=text_weight)
            
        self.ax.axvline(phonemes[-1][1], linewidth=0.5) # for the last line at the end of the phoneme sequence
            
        self.ax.set_ylim(0,1)
        self.ax.set_yticks([])
        self.ax.set_xlim(0, phonemes[-1][1])
        self.ax.set_xlabel("Time [s]")
        self.ax.set_title("Phoneme Alignment")

        self.draw()

    def reset_selection(self) -> None:
        """
        Reset highlighted phoneme back to the normal display state.
        """
        if self.phonemes:
            self.plot_phonemes(
                phonemes=self.phonemes,
                selected_index=None
        )