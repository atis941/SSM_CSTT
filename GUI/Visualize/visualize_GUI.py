import sys

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QToolBar, 
    QWidget, QVBoxLayout, QLabel, 
    QHBoxLayout, QFrame, QFileDialog,
    QScrollArea, QListWidget, QPushButton)
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt

from visualize import load_audio, plot_waveform

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from WaveformCanvas import WaveFormCanvas
from PhonemeCanvas import PhonemeCanvas
from SpectrogramCanvas import SpectrogramCanvas

import torchaudio



example_wav_path = r"/Users/atis/Desktop/MasterArbeit/Code/TIMIT/data/TRAIN/DR1/FCJF0/SA1.WAV"
example_pho_path = r"/Users/atis/Desktop/MasterArbeit/Code/TIMIT/data/TRAIN/DR1/FCJF0/SA1.PHN"

class MainWindow(QMainWindow):
    """Creates the window object
    
    Attributes
    ----------
    None
    """
    def __init__(self,
                 wav_path_choose:str = False):
        """Constructor
        
        Parameters
        ----------
        wav_path: str
            the path to the wav file
            if None -> file must be chosen through FileDialog
        """
        super().__init__()

        self.wav_path = None
        self.phn_path = ""
        self.transcript_path = ""
        self.waveform_tensor = None
        self.sample_rate = None
        self.wav_path_choose = wav_path_choose
        self.phonemes = []

        self.setWindowTitle("Audio Phoneme Visualizer")
        self.resize(1500, 900)
        
        # create ToolBar
        self.toolbar = QToolBar("Main Toolbar")
        self.addToolBar(self.toolbar)

        # create action/button
        self.open_wav_button = QAction("Open WAV", self)
        self.open_wav_button.triggered.connect(self.open_wav_file)
        self.toolbar.addAction(self.open_wav_button)

        self.show_waveform_button = QAction("Waveform", self)
        self.show_waveform_button.triggered.connect(self.show_waveform_view)
        self.toolbar.addAction(self.show_waveform_button)

        self.show_spectrogram_button = QAction("Spectrogram", self)
        self.show_spectrogram_button.triggered.connect(self.show_spectrogram_view)
        self.toolbar.addAction(self.show_spectrogram_button)

        ###### CENTRAL WIDGET ######
        # create and set the central wiget of the window
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # create and set the layout of the central widget
        self.central_layout = QHBoxLayout()
        self.central_widget.setLayout(self.central_layout)

        ###### LEFT COLUMN #####
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout()
        self.left_panel.setLayout(self.left_layout)
        
        self.central_layout.addWidget(self.left_panel, 1)

        self.left_title_section = QWidget()
        self.left_title_section_layout = QHBoxLayout()
        self.left_title_section.setLayout(self.left_title_section_layout)
        
        self.left_label = QLabel("Phoneme Labels")
        self.left_label.setAlignment(Qt.AlignLeft)
        self.left_title_section_layout.addWidget(self.left_label)

        self.reset_zoom_button = QPushButton("Zoom Out", self)
        self.reset_zoom_button.clicked.connect(self.reset_waveform_zoom)
        self.left_title_section_layout.addWidget(self.reset_zoom_button)

        self.left_layout.addWidget(self.left_title_section)

        self.phoneme_list_widget = QListWidget()
        self.phoneme_list_widget.itemDoubleClicked.connect(self.zoom_to_selected_phoneme)
        self.left_layout.addWidget(self.phoneme_list_widget)

        ####### SEPARATOR #######
        self.separator = QFrame()
        self.separator.setFrameShape(QFrame.VLine)
        self.separator.setFrameShadow(QFrame.Sunken)

        self.central_layout.addWidget(self.separator)

        ####### RIGHT COLUMN ######
        self.right_panel_scroll_area = QScrollArea()
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout()
        self.right_panel.setLayout(self.right_layout)

        self.right_panel_scroll_area.setWidget(self.right_panel)
        self.right_panel_scroll_area.setWidgetResizable(True)
        self.central_layout.addWidget(self.right_panel_scroll_area, 5)

        ####### ORIGINAL TRASNCRIPT LABEL #######
        self.transcript_label = QLabel("Original transcript")
        self.transcript_label.setWordWrap(True)

        ####### CANVASES #######
        self.waveform_canvas = WaveFormCanvas()
        self.spectrogram_canvas = SpectrogramCanvas()
        self.phoneme_canvas = PhonemeCanvas() 

        self.plot_toolbar = NavigationToolbar(self.waveform_canvas, self) 

        self.right_layout.addWidget(self.transcript_label, 1)
        self.right_layout.addWidget(self.plot_toolbar)
        self.right_layout.addWidget(self.waveform_canvas, 5)
        self.right_layout.addWidget(self.spectrogram_canvas, 5)
        self.right_layout.addWidget(self.phoneme_canvas, 2)
        self.spectrogram_canvas.hide()

    def show_waveform_view(self) -> None:
        """
        Shows the waveform amplitude plot upon pressing the respective button in the toolbar

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.spectrogram_canvas.hide()
        self.waveform_canvas.show()

        self.plot_toolbar.setParent(None)
        self.plot_toolbar = NavigationToolbar(self.waveform_canvas, self)
        self.right_layout.insertWidget(1, self.plot_toolbar) # insert the toolbar between the Waveform canvas and the transcript label

    def show_spectrogram_view(self) -> None:
        """
        Shows the spectrogram upon pressing the respective button in the toolbar

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.waveform_canvas.hide()
        self.spectrogram_canvas.show()
        
        self.plot_toolbar.setParent(None)
        self.plot_toolbar = NavigationToolbar(self.spectrogram_canvas, self)
        self.right_layout.insertWidget(1, self.plot_toolbar) # insert the toolbar between the Waveform canvas and the transcript label

    def open_wav_file(self) -> None:
        """
        Opens up a file browser to select the WAV file to open
        Sets the path to the phoneme file

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        
        if self.wav_path_choose:
            self.wav_path, _ = QFileDialog.getOpenFileName(
                self,
                "Open WAV File",
                "",
                "WAV Files (*.wav)"
            )
        else:
            self.wav_path = example_wav_path

        if self.wav_path:
            # load the audio file into a pytorch tensor
            self.waveform_tensor, self.sample_rate = torchaudio.load(self.wav_path)

            # show the loaded audio file as a waveform
            self.waveform_canvas.plot_waveform(waveform_tensor=self.waveform_tensor, 
                                               sample_rate=self.sample_rate)
            
            self.spectrogram_canvas.plot_spectrogram(waveform_tensor=self.waveform_tensor,
                                                     sample_rate=self.sample_rate)

            # set the path to the phoneme file based on the wav file
            self.phn_path = self.wav_path.split(".")[0] + ".PHN"

            # get the phonemes and their timing information in seconds
            self.phonemes = self.load_phonemes(phon_path=self.phn_path,
                                               sample_rate=self.sample_rate)
            
            # set the QlistWidget of the phonemes in the left panel
            self.fill_phoneme_list()
            
            ## DEBUGGING REASONS
            for phoneme in self.phonemes:
                print(phoneme)
            
            # plot the phonemes under the waveform
            self.phoneme_canvas.plot_phonemes(phonemes=self.phonemes)

            # load the transcript file
            self.transcript_path = self.wav_path.split(".")[0] + ".TXT"
            transcript_str = self.load_transcript(txt_path = self.transcript_path)
            self.transcript_label.setText(transcript_str)

            

    def load_phonemes(self,
                      phon_path: str,
                      sample_rate: int) -> list:
        """
        Loads the phoneme file and turns the sample positions to seconds

        Parameters
        ----------
        phon_path: str
            the path to the phoneme file

        sample_rate: int
            the sample rate, used when creating the original audio file

        Returns
        -------
        list: list of tuples
            tuple 0th element -> start TIME of the phoneme
            tuple 1st element -> end TIME of the phoneme
            tuple 2nd element -> phoneme
        """
        
        phonemes = []

        with open(phon_path, "r") as fp:
            for line in fp:
                start_sample, end_sample, phoneme = line.strip().split()

                start_time = int(start_sample) / sample_rate
                end_time = int(end_sample) / sample_rate

                phonemes.append((start_time, end_time, phoneme))
        
        return phonemes
    
    def fill_phoneme_list(self) -> None:
        """Display the loaded phonemes in the left panel for each time interval
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # clear the list of the old values before updating
        self.phoneme_list_widget.clear() 

        for start_time, end_time, phoneme in self.phonemes:
            start_time_ms = start_time * 1000
            end_time_ms = end_time * 1000
            text=(
                f"{start_time_ms:.2f} ms -> "
                f"{end_time_ms:.2f} ms   "
                f"{phoneme}"
            )

            self.phoneme_list_widget.addItem(text)


    def load_transcript(self, 
                        txt_path: str) -> str:
        """
        Loads the original transcript

        Parameters
        ----------
        txt_path: str
            the path to the transcript txt

        Returns
        -------
        str:    
            the read transcript
        """

        with open(txt_path, "r") as fp:
            line = fp.readline().strip()

        parts = line.split(maxsplit=2)

        if len(parts) < 3:
            return ""

        transcript = parts[2]
        
        return transcript
    
    def zoom_to_selected_phoneme(self, 
                                 item) -> None:
        """
        The Callback function of clicking one of the elements in the QListWidget

        Parameters
        ----------
        item:
            The item clicked

        Returns
        -------
        None
        """
        # get the index if the clicked row in the QListWidget
        row = self.phoneme_list_widget.row(item)

        start_time, end_time, phoneme = self.phonemes[row]
        
        padding = 0.02

        zoom_start = max(0, start_time - padding)
        zoom_end = end_time + padding

        self.waveform_canvas.zoom_to_time_range(start_time=zoom_start,
                                                end_time=zoom_end,
                                                boundary_start=start_time,
                                                boundary_end=end_time)

        self.phoneme_canvas.plot_phonemes(phonemes=self.phonemes,
                                          selected_index=row)
    
    def reset_waveform_zoom(self) -> None:
        """
        Zoom out to the original waveform in the audio plot

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.waveform_canvas.reset_zoom()
        self.phoneme_list_widget.clearSelection()
        self.phoneme_canvas.reset_selection()


def main() -> None:
    """Shows the GUI
    
    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    # define the app
    app =QApplication(sys.argv)
    
    # create the window
    window = MainWindow(wav_path_choose=False)
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()