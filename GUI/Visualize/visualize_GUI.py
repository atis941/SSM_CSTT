
# for model imports
import json
import torch
from typing import Any
from visualize import load_audio, plot_waveform
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QToolBar,
    QWidget,
    QVBoxLayout,
    QLabel,
    QHBoxLayout,
    QFrame,
    QFileDialog,
    QScrollArea,
    QListWidget,
    QPushButton,
    QMessageBox,
    QStackedWidget,
    QPlainTextEdit,
)

import torchaudio
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from WaveformCanvas import WaveFormCanvas
from PhonemeCanvas import PhonemeCanvas
from SpectrogramCanvas import SpectrogramCanvas

import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[2]

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from SSM_model_V1 import SSMPhonemeModel as SSMPhonemeModelV1
from SSM_model_V2 import SSMPhonemeModel as SSMPhonemeModelV2
from SSM_model_V3 import SSMPhonemeModel as SSMPhonemeModelV3
from SSM_model_V4 import SSMPhonemeModel as SSMPhonemeModelV4
from S2T_CNN_time import S2T_CNN_time_V1


example_wav_path = r"/Users/atis/Desktop/MasterArbeit/Code/TIMIT/data/TRAIN/DR1/FCJF0/SA1.WAV"
example_pho_path = r"/Users/atis/Desktop/MasterArbeit/Code/TIMIT/data/TRAIN/DR1/FCJF0/SA1.PHN"

TEST_DIRECTORY_PATH = Path(
    "/Users/atis/Desktop/MasterArbeit/Code/data/TEST"
)


class MainWindow(QMainWindow):
    """Creates the window object

    Attributes
    ----------
    None
    """

    def __init__(self,
                 wav_path_choose: str = False):
        """Constructor

        Parameters
        ----------
        wav_path: str
            the path to the wav file
            if None -> file must be chosen through FileDialog
        """
        super().__init__()

        # Attributes
        self.wav_path = None
        self.phn_path = ""
        self.transcript_path = ""
        self.waveform_tensor = None
        self.sample_rate = None
        self.wav_path_choose = wav_path_choose
        self.phonemes = []

        # Maps displayed filenames to their loaded content.
        self.model_info_contents: dict[str, str] = {}

        # model and training relevant attributes
        self.model_directory_path: str | None = None

        self.run_config: dict[str, Any] | None = None
        self.losses_and_accuracies: dict[str, Any] | None = None

        self.training_log_loss_and_per: str | None = None
        self.training_log_predictions: str | None = None

        self.model_state_dict: dict[str, torch.Tensor] | None = None
        self.loaded_model: torch.nn.Module | None = None

        # Inference results
        self.predicted_phonemes: list[str] = []
        self.target_phonemes: list[str] = []

        self.aligned_predictions: list[str] = []
        self.aligned_targets: list[str] = []

        self.inference_statistics: dict[str, Any] = {}

        self.idx_to_phoneme_vocab: dict[int, str] = {}


        # Inference-set attributes
        ##########################
        self.test_directory_path = TEST_DIRECTORY_PATH

        # One dictionary entry per evaluated WAV file.
        self.inference_set_results: dict[str, dict[str, Any]] = {}

        # Overall statistics calculated across the complete set.
        self.inference_set_statistics: dict[str, Any] = {}

        # Stores the currently selected list entry.
        self.current_inference_set_key: str | None = None

        # For GUI inference, CPU is generally the simplest choice.
        self.model_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.setWindowTitle("Audio Phoneme Visualizer")
        self.resize(1500, 900)

        ####### MENU BAR #######
        # Access the QMainWindow's menu bar.
        self.menu_bar = self.menuBar()

        # Create the File dropdown menu.
        self.file_menu = self.menu_bar.addMenu("File")

        # Open WAV action
        self.open_wav_action = QAction("Open WAV", self)
        self.open_wav_action.triggered.connect(self.open_wav_file)
        self.file_menu.addAction(self.open_wav_action)

        # Import Model action
        self.import_model_action = QAction("Import Model", self)
        self.import_model_action.triggered.connect(self.import_model)
        self.file_menu.addAction(self.import_model_action)

        # Separator line inside the File menu
        self.file_menu.addSeparator()

        # Exit action
        self.exit_action = QAction("Exit", self)
        self.exit_action.triggered.connect(self.close)
        self.file_menu.addAction(self.exit_action)


        ####### TOOLBAR #######

        # Keep only the view-switching actions in the toolbar.
        self.toolbar = QToolBar("Main Toolbar")
        self.addToolBar(self.toolbar)

        self.show_waveform_action = QAction("Waveform", self)
        self.show_waveform_action.triggered.connect(
            self.show_waveform_view
        )
        self.toolbar.addAction(self.show_waveform_action)

        self.inference_action = QAction("Inference", self)
        self.inference_action.triggered.connect(
            self.run_inference
        )
        self.toolbar.addAction(self.inference_action)

        self.show_spectrogram_action = QAction("Spectrogram", self)
        self.show_spectrogram_action.triggered.connect(
            self.show_spectrogram_view
        )
        self.toolbar.addAction(self.show_spectrogram_action)

        self.model_info_action = QAction("Model Info", self)
        self.model_info_action.triggered.connect(
            self.show_model_info_view
        )
        self.toolbar.addAction(self.model_info_action)

        self.inference_set_action = QAction("Inference set", self)
        self.inference_set_action.triggered.connect(
            self.run_inference_set
        )
        self.toolbar.addAction(
            self.inference_set_action
)

        

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

        self.left_stack = QStackedWidget()
        self.central_layout.addWidget(self.left_stack, 1)

        self.left_stack.addWidget(self.left_panel)

        ##### Phoneme Infos left section ######
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
        self.phoneme_list_widget.itemDoubleClicked.connect(
            self.zoom_to_selected_phoneme)
        self.left_layout.addWidget(self.phoneme_list_widget)

        ####### MODEL INFO LEFT PAGE #######
        self.model_info_left_panel = QWidget()
        self.model_info_left_layout = QVBoxLayout()
        self.model_info_left_panel.setLayout(
            self.model_info_left_layout
        )

        self.model_info_title = QLabel("Model Files")
        self.model_info_left_layout.addWidget(
            self.model_info_title
        )

        self.model_file_list_widget = QListWidget()
        self.model_file_list_widget.itemClicked.connect(
            self.show_selected_model_file
        )

        self.model_info_left_layout.addWidget(
            self.model_file_list_widget
        )

        self.left_stack.addWidget(
            self.model_info_left_panel
        )

        ####### INFERENCE LEFT PAGE #######
        self.inference_left_panel = QWidget()
        self.inference_left_layout = QVBoxLayout()
        self.inference_left_panel.setLayout(
            self.inference_left_layout
        )

        self.inference_title_label = QLabel("Inference Summary")
        self.inference_title_label.setAlignment(Qt.AlignCenter)

        self.inference_left_layout.addWidget(
            self.inference_title_label
        )

        self.inference_statistics_view = QPlainTextEdit()
        self.inference_statistics_view.setReadOnly(True)
        self.inference_statistics_view.setPlainText(
            "Run inference to display statistics."
        )

        self.inference_left_layout.addWidget(
            self.inference_statistics_view
        )

        self.left_stack.addWidget(
            self.inference_left_panel
        )

        ####### INFERENCE SET LEFT PAGE #######
        self.inference_set_left_panel = QWidget()
        self.inference_set_left_layout = QVBoxLayout()
        self.inference_set_left_panel.setLayout(
            self.inference_set_left_layout
        )

        self.inference_set_title_label = QLabel(
            "Inference Set"
        )
        self.inference_set_title_label.setAlignment(
            Qt.AlignCenter
        )

        self.inference_set_left_layout.addWidget(
            self.inference_set_title_label
        )

        # Button that shows the overall set statistics.
        self.inference_set_info_button = QPushButton(
            "Inference info",
            self
        )
        self.inference_set_info_button.clicked.connect(
            self.show_inference_set_summary
        )

        self.inference_set_left_layout.addWidget(
            self.inference_set_info_button
        )

        # List of all processed WAV files.
        self.inference_set_file_list_widget = QListWidget()
        self.inference_set_file_list_widget.itemClicked.connect(
            self.show_selected_inference_set_result
        )

        self.inference_set_left_layout.addWidget(
            self.inference_set_file_list_widget
        )

        self.left_stack.addWidget(
            self.inference_set_left_panel
        )

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
        self.right_stack = QStackedWidget()
        self.central_layout.addWidget(
            self.right_stack,
            5
        )
        self.right_stack.addWidget(
            self.right_panel_scroll_area
        )

        ####### MODEL INFO RIGHT PAGE #######
        self.model_info_right_panel = QWidget()
        self.model_info_right_layout = QVBoxLayout()
        self.model_info_right_panel.setLayout(
            self.model_info_right_layout
        )

        self.selected_model_file_label = QLabel(
            "Select a model file from the left panel."
        )

        self.model_info_right_layout.addWidget(
            self.selected_model_file_label
        )

        self.model_file_content_view = QPlainTextEdit()
        self.model_file_content_view.setReadOnly(True)

        self.model_info_right_layout.addWidget(
            self.model_file_content_view
        )

        self.right_stack.addWidget(
            self.model_info_right_panel
        )

        ####### INFERENCE RIGHT PAGE #######
        self.inference_right_panel = QWidget()
        self.inference_right_layout = QVBoxLayout()
        self.inference_right_panel.setLayout(
            self.inference_right_layout
        )

        self.inference_results_title = QLabel(
            "Aligned Phoneme Prediction"
        )
        self.inference_results_title.setAlignment(Qt.AlignCenter)

        self.inference_right_layout.addWidget(
            self.inference_results_title
        )

        self.inference_results_view = QPlainTextEdit()
        self.inference_results_view.setReadOnly(True)
        self.inference_results_view.setLineWrapMode(
            QPlainTextEdit.NoWrap
        )

        # A monospace font makes columns align correctly.
        inference_font = self.inference_results_view.font()
        inference_font.setFamily("Monospace")
        inference_font.setStyleHint(
            inference_font.StyleHint.Monospace
        )

        self.inference_results_view.setFont(
            inference_font
        )

        self.inference_results_view.setPlainText(
            "Predicted and target phonemes will appear here."
        )

        self.inference_right_layout.addWidget(
            self.inference_results_view
        )

        self.right_stack.addWidget(
            self.inference_right_panel
        )

        ####### INFERENCE SET RIGHT PAGE #######
        self.inference_set_right_panel = QWidget()
        self.inference_set_right_layout = QVBoxLayout()
        self.inference_set_right_panel.setLayout(
            self.inference_set_right_layout
        )

        self.inference_set_result_title = QLabel(
            "Inference Set Summary"
        )
        self.inference_set_result_title.setAlignment(
            Qt.AlignCenter
        )

        self.inference_set_right_layout.addWidget(
            self.inference_set_result_title
        )

        self.inference_set_result_view = QPlainTextEdit()
        self.inference_set_result_view.setReadOnly(True)
        self.inference_set_result_view.setLineWrapMode(
            QPlainTextEdit.NoWrap
        )

        set_inference_font = self.inference_set_result_view.font()
        set_inference_font.setFamily("Monospace")
        set_inference_font.setStyleHint(
            set_inference_font.StyleHint.Monospace
        )

        self.inference_set_result_view.setFont(
            set_inference_font
        )

        self.inference_set_result_view.setPlainText(
            "Run inference on the test set to display results."
        )

        self.inference_set_right_layout.addWidget(
            self.inference_set_result_view
        )

        self.right_stack.addWidget(
            self.inference_set_right_panel
        )

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


    def show_inference_set_view(self) -> None:
        """
        Switch to the inference-set pages.
        """

        self.left_stack.setCurrentWidget(
            self.inference_set_left_panel
        )

        self.right_stack.setCurrentWidget(
            self.inference_set_right_panel
        )

    def find_test_file_pairs(self) -> list[tuple[Path, Path]]:
        """
        Recursively find every WAV file in the static TEST directory
        that has a matching PHN file.

        Returns
        -------
        list of tuples:
            [
                (wav_path, phn_path),
                ...
            ]
        """

        if not self.test_directory_path.is_dir():
            raise FileNotFoundError(
                "The configured TEST directory does not exist:\n"
                f"{self.test_directory_path}"
            )

        file_pairs = []

        # TIMIT files may use uppercase .WAV.
        wav_paths = sorted(
            list(self.test_directory_path.rglob("*.WAV"))
            + list(self.test_directory_path.rglob("*.wav"))
        )

        for wav_path in wav_paths:
            # First try the usual uppercase TIMIT extension.
            phn_path = wav_path.with_suffix(".PHN")

            # Also support lowercase files.
            if not phn_path.is_file():
                phn_path = wav_path.with_suffix(".phn")

            if phn_path.is_file():
                file_pairs.append(
                    (wav_path, phn_path)
                )

        return file_pairs

    def prepare_waveform_for_inference(self,
                                       waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare a given waveform for the currently loaded model.

        The preprocessing must match the model's training configuration.
        """

        if self.run_config is None:
            raise ValueError(
                "No model configuration has been loaded."
            )

        # Convert stereo to mono.
        if waveform.shape[0] > 1:
            waveform = waveform.mean(
                dim=0,
                keepdim=True
            )

        original_length = waveform.shape[-1]

        collate_function = self.run_config.get(
            "collate_function"
        )

        # ---------------------------------------------------------
        # Raw waveform model
        # ---------------------------------------------------------
        if (
            collate_function
            == "collate_fn_ctc_time_domain_features_during_training"
        ):
            # [1, T] -> [1, T, 1]
            model_input = waveform.transpose(
                0,
                1
            ).unsqueeze(0)

            input_length = torch.tensor(
                [original_length],
                dtype=torch.long
            )

        # ---------------------------------------------------------
        # Mel-spectrogram model
        # ---------------------------------------------------------
        elif (
            collate_function
            == "collate_fn_ctc_time_domain_padding"
        ):
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.run_config[
                    "mel_transform_sample_rate"
                ],
                n_fft=self.run_config[
                    "mel_transform_nfft"
                ],
                hop_length=self.run_config[
                    "mel_transform_hop_length"
                ],
                win_length=self.run_config[
                    "mel_transform_win_length"
                ],
                n_mels=self.run_config[
                    "mel_transform_n_mels"
                ]
            )

            mel_features = mel_transform(
                waveform
            )

            mel_features = torch.log(
                mel_features + 1e-6
            )

            model_input = mel_features.transpose(
                1,
                2
            )

            input_length = torch.tensor(
                [model_input.shape[1]],
                dtype=torch.long
            )

        else:
            raise ValueError(
                "Unsupported inference preprocessing for "
                f"collate function: {collate_function}"
            )

        return (
            model_input.to(self.model_device),
            input_length.to(self.model_device)
        )

    def load_target_phonemes_from_file(self,
                                       phn_path: Path) -> list[str]:
        """
        Load only the phoneme labels from a TIMIT PHN file.
        """

        target_phonemes = []

        with phn_path.open(
            mode="r",
            encoding="utf-8"
        ) as file:
            for line in file:
                parts = line.strip().split()

                if len(parts) < 3:
                    continue

                target_phonemes.append(
                    parts[-1]
                )

        return target_phonemes

    def show_waveform_view(self) -> None:
        """
        Switch to the normal audio view and display the waveform.
        """

        # Switch the stacked widgets back to the audio pages.
        self.left_stack.setCurrentWidget(
            self.left_panel
        )

        self.right_stack.setCurrentWidget(
            self.right_panel_scroll_area
        )

        # Show the waveform and hide the spectrogram.
        self.spectrogram_canvas.hide()
        self.waveform_canvas.show()

        # Remove the old Matplotlib toolbar cleanly.
        self.right_layout.removeWidget(
            self.plot_toolbar
        )

        self.plot_toolbar.deleteLater()

        # Create a toolbar connected to the waveform canvas.
        self.plot_toolbar = NavigationToolbar(
            self.waveform_canvas,
            self
        )

        # Insert it below the transcript label.
        self.right_layout.insertWidget(
            1,
            self.plot_toolbar
        )

        # Ask Qt and Matplotlib to redraw the canvas.
        self.waveform_canvas.draw_idle()
        self.right_panel.update()

    def show_model_info_view(self) -> None:
        """
        Switch the application to the model-information view.

        The left side displays files from the imported result directory.
        The right side displays the selected file's contents.
        """

        if self.loaded_model is None or self.run_config is None:
            QMessageBox.warning(
                self,
                "No Model Imported",
                "Import a model before opening the model-information view."
            )
            return

        self.left_stack.setCurrentIndex(1)
        self.right_stack.setCurrentIndex(1)

    def show_spectrogram_view(self) -> None:
        """
        Switch to the normal audio view and display the spectrogram.
        """

        # Switch the stacked widgets back to the audio pages.
        self.left_stack.setCurrentWidget(
            self.left_panel
        )

        self.right_stack.setCurrentWidget(
            self.right_panel_scroll_area
        )

        # Show the spectrogram and hide the waveform.
        self.waveform_canvas.hide()
        self.spectrogram_canvas.show()

        # Remove the old Matplotlib toolbar cleanly.
        self.right_layout.removeWidget(
            self.plot_toolbar
        )

        self.plot_toolbar.deleteLater()

        # Create a toolbar connected to the spectrogram canvas.
        self.plot_toolbar = NavigationToolbar(
            self.spectrogram_canvas,
            self
        )

        self.right_layout.insertWidget(
            1,
            self.plot_toolbar
        )

        self.spectrogram_canvas.draw_idle()
        self.right_panel.update()

    def show_inference_view(self) -> None:
        """
        Switch the interface to the inference result pages.
        """

        self.left_stack.setCurrentWidget(
            self.inference_left_panel
        )

        self.right_stack.setCurrentWidget(
            self.inference_right_panel
        )

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
            self.waveform_tensor, self.sample_rate = torchaudio.load(
                self.wav_path)

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

            # DEBUGGING REASONS
            for phoneme in self.phonemes:
                print(phoneme)

            # plot the phonemes under the waveform
            self.phoneme_canvas.plot_phonemes(phonemes=self.phonemes)

            # load the transcript file
            self.transcript_path = self.wav_path.split(".")[0] + ".TXT"
            transcript_str = self.load_transcript(
                txt_path=self.transcript_path)
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
            text = (
                f"{start_time_ms:.2f} ms -> "
                f"{end_time_ms:.2f} ms   "
                f"{phoneme}"
            )

            self.phoneme_list_widget.addItem(text)

    def create_model_from_run_config(self,
                                     run_config: dict[str, Any]) -> torch.nn.Module:
        """
        Creates the correct neural-network architecture from run_config.json.

        Parameters
        ----------
        run_config:
            Dictionary loaded from run_config.json.

        Returns
        -------
        torch.nn.Module
            The initialized model architecture.

        Raises
        ------
        KeyError
            If a required configuration entry is missing.

        ValueError
            If the model version is unsupported or the configuration is invalid.
        """

        model_info = run_config.get("model_info")

        if not isinstance(model_info, dict):
            raise ValueError(
                "run_config.json does not contain a valid 'model_info' dictionary."
            )

        model_type = model_info.get("model_type")
        model_version = model_info.get("model_version")

        if not model_version:
            raise ValueError(
                "The model version is missing from run_config.json."
            )

        # Vocabulary size may be stored directly or derived from the vocabulary.
        vocab_size = run_config.get("ssm_vocab_size")

        if vocab_size is None:
            phoneme_vocab = run_config.get("phoneme_to_idx_vocab")

            if not isinstance(phoneme_vocab, dict):
                raise ValueError(
                    "Could not determine vocabulary size from run_config.json."
                )

            vocab_size = len(phoneme_vocab)

        # ---------------------------------------------------------
        # SSM V1
        # ---------------------------------------------------------
        if model_type == "SSM" and model_version == "V1":
            model = SSMPhonemeModelV1(
                d_in=model_info["ssm_d_in"],
                d_state=model_info["ssm_d_state"],
                d_out=model_info["ssm_d_out"],
                vocab_size=vocab_size,
                dropout=model_info.get(
                    "ssm_dropout",
                    run_config.get("ssm_dropout", 0.0)
                )
            )

        # ---------------------------------------------------------
        # SSM V2
        # ---------------------------------------------------------
        elif model_type == "SSM" and model_version == "V2":
            model = SSMPhonemeModelV2(
                d_in=model_info["ssm_d_in"],
                d_state=model_info["ssm_d_state"],
                vocab_size=vocab_size,
                dropout=model_info.get(
                    "ssm_dropout",
                    run_config.get("ssm_dropout", 0.0)
                ),
                num_layers=model_info["ssm_num_layers"]
            )

        # ---------------------------------------------------------
        # SSM V3
        # ---------------------------------------------------------
        elif model_type == "SSM" and model_version == "V3":
            model = SSMPhonemeModelV3(
                d_state=model_info["ssm_d_state"],
                d_in=model_info["ssm_d_in"],
                d_out=model_info["ssm_d_out"],
                num_of_ssm_layers=model_info["num_of_ssm_layers"],
                linear_in_features=model_info["linear_in_features"],
                linear_out_features=model_info["linear_out_features"],
                num_of_linear_layers=model_info["num_of_linear_layers"],
                vocab_size=vocab_size,
                ssm_dropout=model_info.get(
                    "ssm_dropout",
                    run_config.get("ssm_dropout", 0.0)
                ),
                linear_dropout=model_info.get("linear_dropout", 0.0)
            )

        # ---------------------------------------------------------
        # SSM V4
        # ---------------------------------------------------------
        elif model_type == "SSM" and model_version == "V4":
            model = SSMPhonemeModelV4(
                d_state=model_info["ssm_d_state"],
                d_in=model_info["ssm_d_in"],
                d_out=model_info["ssm_d_out"],
                num_of_ssm_layers=model_info["num_of_ssm_layers"],
                vocab_size=vocab_size,
                ssm_dropout=model_info.get("ssm_dropout", 0.0),
                linear_dropout=model_info.get("linear_dropout", 0.0),
                norm=model_info.get("norm", True),
                norm_type=model_info.get("norm_type", "bn"),
                act=model_info.get("act", "LeakyRELu"),
                trainable_SkipLayer=model_info.get(
                    "trainable_SkipLayer",
                    False
                )
            )

        # ---------------------------------------------------------
        # Time-domain CNN V1
        # ---------------------------------------------------------
        elif (
            model_type == "CNN"
            and model_version == "CNN_time_domain_V1"
        ):
            padding = model_info["padding"]

            # Older JSON/configuration files may contain [2]
            # if padding was accidentally created as the tuple (2,).
            if isinstance(padding, list) and len(padding) == 1:
                padding = padding[0]

            model = S2T_CNN_time_V1(
                vocab_size=vocab_size,
                kernel_size=model_info["kernel_size"],
                padding=padding,
                padding_type=model_info["padding_type"],
                stride=model_info["stride"],
                in_channels=model_info["in_channels"],
                out_channels=model_info["out_channels"],
                lin_dropout=model_info.get(
                    "linear_dropout",
                    model_info.get("lin_dropout", 0.0)
                )
            )

        else:
            raise ValueError(
                "Unsupported model configuration:\n"
                f"model_type = {model_type}\n"
                f"model_version = {model_version}"
            )

        return model.to(self.model_device)

    def import_model(self) -> None:
        """
        Selects a saved training-result directory and loads:

        - run_config.json
        - losses_and_accuracies.json
        - ssm_model_state_dict.pt
        - training_log_loss_and_PER.txt
        - training_log_predictions.txt

        The correct model architecture is created from run_config.json.
        The state dictionary is then loaded into that model.
        """

        selected_directory = QFileDialog.getExistingDirectory(
            self,
            "Select Model Result Directory",
            ""
        )

        # User cancelled the dialog.
        if not selected_directory:
            return

        model_directory = Path(selected_directory)

        expected_files = {
            "model_state_dict":
                model_directory / "ssm_model_state_dict.pt",

            "run_config":
                model_directory / "run_config.json",

            "losses_and_accuracies":
                model_directory / "losses_and_accuracies.json",

            "training_log_loss_and_per":
                model_directory / "training_log_loss_and_PER.txt",

            "training_log_predictions":
                model_directory / "training_log_predictions.txt",
        }

        # Check all files before modifying the current application state.
        missing_files = [
            file_path.name
            for file_path in expected_files.values()
            if not file_path.is_file()
        ]

        if missing_files:
            QMessageBox.critical(
                self,
                "Invalid Model Directory",
                "The selected directory is missing the following files:\n\n"
                + "\n".join(missing_files)
            )
            return

        try:
            # -----------------------------------------------------
            # Load configuration
            # -----------------------------------------------------
            with expected_files["run_config"].open(
                mode="r",
                encoding="utf-8"
            ) as file:
                run_config = json.load(file)

            with expected_files["losses_and_accuracies"].open(
                mode="r",
                encoding="utf-8"
            ) as file:
                losses_and_accuracies = json.load(file)

            # -----------------------------------------------------
            # Create model architecture from configuration
            # -----------------------------------------------------
            loaded_model = self.create_model_from_run_config(
                run_config=run_config
            )

            # -----------------------------------------------------
            # Load state dictionary
            # -----------------------------------------------------
            loaded_object = torch.load(
                expected_files["model_state_dict"],
                map_location=self.model_device
            )

            # Support both:
            #
            # 1. torch.save(model.state_dict(), path)
            # 2. torch.save({"model_state_dict": model.state_dict()}, path)
            if (
                isinstance(loaded_object, dict)
                and "model_state_dict" in loaded_object
            ):
                model_state_dict = loaded_object["model_state_dict"]
            else:
                model_state_dict = loaded_object

            if not isinstance(model_state_dict, dict):
                raise ValueError(
                    "The .pt file does not contain a valid model state dictionary."
                )

            # strict=True detects architecture/configuration mismatches.
            # The saved weights in model_state_dict (loaded_model) are loaded into the "empty" ML model object (loaded_model)
            loaded_model.load_state_dict(
                model_state_dict,
                strict=True
            )

            # Important for inference:
            # disables dropout and uses saved BatchNorm statistics.
            loaded_model.eval()

            # -----------------------------------------------------
            # Load log files
            # -----------------------------------------------------
            with expected_files["training_log_loss_and_per"].open(
                mode="r",
                encoding="utf-8"
            ) as file:
                training_log_loss_and_per = file.read()

            with expected_files["training_log_predictions"].open(
                mode="r",
                encoding="utf-8"
            ) as file:
                training_log_predictions = file.read()

            # -----------------------------------------------------
            # Save everything only after successful loading
            # -----------------------------------------------------
            self.model_directory_path = str(model_directory)

            self.run_config = run_config

            phoneme_to_idx_vocab = run_config.get(
                "phoneme_to_idx_vocab"
            )

            if not isinstance(phoneme_to_idx_vocab, dict):
                raise ValueError(
                    "run_config.json does not contain a valid "
                    "'phoneme_to_idx_vocab' dictionary."
                )

            self.idx_to_phoneme_vocab = {
                int(index): phoneme
                for phoneme, index in phoneme_to_idx_vocab.items()
            }

            self.losses_and_accuracies = losses_and_accuracies

            self.model_state_dict = model_state_dict
            self.loaded_model = loaded_model

            self.training_log_loss_and_per = (
                training_log_loss_and_per
            )
            self.training_log_predictions = (
                training_log_predictions
            )

            # -----------------------------------------------------
            # Prepare model information for the GUI
            # -----------------------------------------------------
            state_dict_summary_lines = [
                "Model state dictionary",
                "======================",
                "",
                f"Number of stored tensors: {len(model_state_dict)}",
                ""
            ]

            for parameter_name, parameter_tensor in model_state_dict.items():
                state_dict_summary_lines.append(
                    f"{parameter_name}: "
                    f"shape={tuple(parameter_tensor.shape)}, "
                    f"dtype={parameter_tensor.dtype}"
                )

            state_dict_summary = "\n".join(
                state_dict_summary_lines
            )

            self.model_info_contents = {
                "run_config.json": json.dumps(
                    run_config,
                    indent=4,
                    ensure_ascii=False
                ),

                "losses_and_accuracies.json": json.dumps(
                    losses_and_accuracies,
                    indent=4,
                    ensure_ascii=False
                ),

                "training_log_loss_and_PER.txt":
                    training_log_loss_and_per,

                "training_log_predictions.txt":
                    training_log_predictions,

                "ssm_model_state_dict.pt":
                    state_dict_summary,
            }

            self.fill_model_file_list()

        except (
            OSError,
            KeyError,
            TypeError,
            ValueError,
            RuntimeError,
            json.JSONDecodeError
        ) as error:
            QMessageBox.critical(
                self,
                "Model Import Failed",
                "The model could not be imported.\n\n"
                f"{type(error).__name__}: {error}"
            )
            return

        model_info = self.run_config["model_info"]

        QMessageBox.information(
            self,
            "Model Imported",
            "The model was loaded successfully.\n\n"
            f"Type: {model_info.get('model_type')}\n"
            f"Version: {model_info.get('model_version')}\n"
            f"Device: {self.model_device}\n\n"
            f"Directory:\n{self.model_directory_path}"
        )

        print("Model imported successfully")
        print("Model type:", model_info.get("model_type"))
        print("Model version:", model_info.get("model_version"))
        print("Device:", self.model_device)

    def fill_model_file_list(self) -> None:
        """
        Fill the model-information list with files loaded from the
        selected result directory.
        """

        self.model_file_list_widget.clear()

        for filename in self.model_info_contents:
            self.model_file_list_widget.addItem(
                filename
            )

    def ctc_greedy_decode_ids(self,
                              prediction_ids: list[int],
                              blank_index: int = 0) -> list[int]:
        """
        Collapse repeated CTC predictions and remove blank tokens.

        Example
        -------
        Input:
            [0, 5, 5, 0, 8, 8, 8, 3]

        Output:
            [5, 8, 3]
        """

        decoded_ids = []
        previous_id = None

        for current_id in prediction_ids:
            if (
                current_id != blank_index
                and current_id != previous_id
            ):
                decoded_ids.append(current_id)

            previous_id = current_id

        return decoded_ids

    def get_target_phonemes_for_inference(self) -> list[str]:
        """
        Return the target phoneme sequence for the loaded WAV file.

        The mapping used here must match the phone-set mapping used
        during training.
        """

        target_phonemes = [
            phoneme
            for _, _, phoneme in self.phonemes
        ]

        return target_phonemes

    def align_phoneme_sequences(self, predicted: list[str], target: list[str]) -> tuple[list[str], list[str], dict[str, int]]:
        """
        Align predicted and target phoneme sequences using
        Levenshtein dynamic programming.

        A dash represents a missing phoneme caused by an insertion
        or deletion.

        Returns
        -------
        aligned_predictions
        aligned_targets
        operation_counts
        """

        num_target = len(target)
        num_predicted = len(predicted)

        # distance[i][j] represents the minimum edit distance between:
        #
        # target[:i]
        # predicted[:j]
        distance = [
            [0] * (num_predicted + 1)
            for _ in range(num_target + 1)
        ]

        for target_index in range(num_target + 1):
            distance[target_index][0] = target_index

        for predicted_index in range(num_predicted + 1):
            distance[0][predicted_index] = predicted_index

        for target_index in range(1, num_target + 1):
            for predicted_index in range(1, num_predicted + 1):

                target_phoneme = target[target_index - 1]
                predicted_phoneme = predicted[predicted_index - 1]

                substitution_cost = (
                    0
                    if target_phoneme == predicted_phoneme
                    else 1
                )

                distance[target_index][predicted_index] = min(
                    distance[target_index - 1][predicted_index] + 1,
                    distance[target_index][predicted_index - 1] + 1,
                    distance[target_index - 1][predicted_index - 1]
                    + substitution_cost
                )

        aligned_targets = []
        aligned_predictions = []

        substitutions = 0
        insertions = 0
        deletions = 0
        correct = 0

        target_index = num_target
        predicted_index = num_predicted

        while target_index > 0 or predicted_index > 0:

            # Match or substitution
            if target_index > 0 and predicted_index > 0:
                target_phoneme = target[target_index - 1]
                predicted_phoneme = predicted[predicted_index - 1]

                substitution_cost = (
                    0
                    if target_phoneme == predicted_phoneme
                    else 1
                )

                if (
                    distance[target_index][predicted_index]
                    == distance[target_index - 1][predicted_index - 1]
                    + substitution_cost
                ):
                    aligned_targets.append(target_phoneme)
                    aligned_predictions.append(predicted_phoneme)

                    if substitution_cost == 0:
                        correct += 1
                    else:
                        substitutions += 1

                    target_index -= 1
                    predicted_index -= 1
                    continue

            # Deletion: a target phoneme was not predicted
            if (
                target_index > 0
                and distance[target_index][predicted_index]
                == distance[target_index - 1][predicted_index] + 1
            ):
                aligned_targets.append(
                    target[target_index - 1]
                )
                aligned_predictions.append("-")

                deletions += 1
                target_index -= 1
                continue

            # Insertion: prediction contains an extra phoneme
            aligned_targets.append("-")
            aligned_predictions.append(
                predicted[predicted_index - 1]
            )

            insertions += 1
            predicted_index -= 1

        aligned_targets.reverse()
        aligned_predictions.reverse()

        operation_counts = {
            "correct": correct,
            "substitutions": substitutions,
            "insertions": insertions,
            "deletions": deletions,
            "edit_distance": (
                substitutions
                + insertions
                + deletions
            )
        }

        return (
            aligned_predictions,
            aligned_targets,
            operation_counts
        )      

    def calculate_inference_statistics(self,
                                       predicted: list[str],
                                       target: list[str],
                                       operation_counts: dict[str, int]) -> dict[str, Any]:
        """
        Calculate phoneme-level inference metrics.
        """

        total_target_phonemes = len(target)

        correct_phonemes = operation_counts["correct"]

        wrong_phonemes = (
            operation_counts["substitutions"]
            + operation_counts["deletions"]
        )

        edit_distance = operation_counts["edit_distance"]

        per = (
            edit_distance / total_target_phonemes
            if total_target_phonemes > 0
            else 0.0
        )

        correct_percentage = (
            correct_phonemes / total_target_phonemes * 100
            if total_target_phonemes > 0
            else 0.0
        )

        wrong_percentage = (
            wrong_phonemes / total_target_phonemes * 100
            if total_target_phonemes > 0
            else 0.0
        )

        exact_sequence_match = (
            predicted == target
        )

        return {
            "total_target_phonemes": total_target_phonemes,
            "total_predicted_phonemes": len(predicted),
            "correct_phonemes": correct_phonemes,
            "wrong_target_phonemes": wrong_phonemes,
            "substitutions": operation_counts["substitutions"],
            "insertions": operation_counts["insertions"],
            "deletions": operation_counts["deletions"],
            "edit_distance": edit_distance,
            "per": per,
            "correct_percentage": correct_percentage,
            "wrong_percentage": wrong_percentage,
            "exact_sequence_match": exact_sequence_match
        }

    def infer_single_waveform(self,
                              waveform: torch.Tensor,
                              target_phonemes: list[str]) -> dict[str, Any]:
        """
        Run inference and calculate statistics for one waveform.
        """

        if self.loaded_model is None:
            raise ValueError(
                "No trained model has been imported."
            )

        model_input, input_lengths = (
            self.prepare_waveform_for_inference(
                waveform
            )
        )

        self.loaded_model.eval()

        with torch.inference_mode():
            model_output = self.loaded_model(
                model_input
            )

            if isinstance(model_output, tuple):
                logits, model_output_lengths = model_output

                valid_output_length = int(
                    model_output_lengths[0].item()
                )

            else:
                logits = model_output
                valid_output_length = logits.shape[1]

            prediction_ids = torch.argmax(
                logits,
                dim=-1
            )[0, :valid_output_length].cpu().tolist()

        blank_index = self.run_config.get(
            "blank_index",
            0
        )

        decoded_ids = self.ctc_greedy_decode_ids(
            prediction_ids=prediction_ids,
            blank_index=blank_index
        )

        predicted_phonemes = [
            self.idx_to_phoneme_vocab[
                phoneme_id
            ]
            for phoneme_id in decoded_ids
        ]

        (
            aligned_predictions,
            aligned_targets,
            operation_counts
        ) = self.align_phoneme_sequences(
            predicted=predicted_phonemes,
            target=target_phonemes
        )

        statistics = self.calculate_inference_statistics(
            predicted=predicted_phonemes,
            target=target_phonemes,
            operation_counts=operation_counts
        )

        return {
            "predicted_phonemes": predicted_phonemes,
            "target_phonemes": target_phonemes,
            "aligned_predictions": aligned_predictions,
            "aligned_targets": aligned_targets,
            "operation_counts": operation_counts,
            "statistics": statistics
        }

    def fill_inference_set_file_list(self) -> None:
        """
        Fill the left-side list with all evaluated files.
        """

        self.inference_set_file_list_widget.clear()

        for result_key in self.inference_set_results:
            self.inference_set_file_list_widget.addItem(
                result_key
            )

    def show_inference_set_summary(self) -> None:
        """
        Show aggregate statistics for the complete inference set.
        """

        if not self.inference_set_statistics:
            self.inference_set_result_title.setText(
                "Inference Set Summary"
            )

            self.inference_set_result_view.setPlainText(
                "No inference-set results are available."
            )

            return

        self.inference_set_result_title.setText(
            "Inference Set Summary"
        )

        self.inference_set_result_view.setPlainText(
            self.format_inference_set_statistics(
                self.inference_set_statistics
            )
        )

    def format_inference_set_statistics(self,
                                        statistics: dict[str, Any]) -> str:
        """
        Format complete-set inference statistics.
        """

        return (
            "TEST SET INFERENCE SUMMARY\n"
            "==========================\n\n"

            f"Directory:\n"
            f"{self.test_directory_path}\n\n"

            f"Processed files:\n"
            f"{statistics['total_files']}\n\n"

            f"Target phonemes:\n"
            f"{statistics['total_target_phonemes']}\n\n"

            f"Predicted phonemes:\n"
            f"{statistics['total_predicted_phonemes']}\n\n"

            f"Correct phonemes:\n"
            f"{statistics['correct_phonemes']} / "
            f"{statistics['total_target_phonemes']}\n"
            f"{statistics['correct_percentage']:.2f}%\n\n"

            f"Wrong target phonemes:\n"
            f"{statistics['wrong_target_phonemes']} / "
            f"{statistics['total_target_phonemes']}\n"
            f"{statistics['wrong_percentage']:.2f}%\n\n"

            f"Substitutions:\n"
            f"{statistics['substitutions']}\n\n"

            f"Deletions:\n"
            f"{statistics['deletions']}\n\n"

            f"Insertions:\n"
            f"{statistics['insertions']}\n\n"

            f"Total edit distance:\n"
            f"{statistics['edit_distance']}\n\n"

            f"Corpus phoneme error rate:\n"
            f"{statistics['per']:.4f}\n"
            f"{statistics['per'] * 100:.2f}%\n\n"

            f"Exact sequence matches:\n"
            f"{statistics['exact_sequence_matches']} / "
            f"{statistics['total_files']}\n"
            f"{statistics['exact_match_percentage']:.2f}%"
        )

    def run_inference_set(self) -> None:
        """
        Run inference over every WAV/PHN pair in the configured TEST
        directory.
        """

        if self.loaded_model is None:
            QMessageBox.warning(
                self,
                "No Model Imported",
                "Import a trained model before running set inference."
            )
            return

        try:
            file_pairs = self.find_test_file_pairs()

            if not file_pairs:
                raise ValueError(
                    "No matching WAV/PHN pairs were found in:\n"
                    f"{self.test_directory_path}"
                )

            # Switch to the new view before inference begins.
            self.show_inference_set_view()

            self.inference_set_result_title.setText(
                "Running Test Set Inference"
            )

            self.inference_set_result_view.setPlainText(
                f"Found {len(file_pairs)} WAV/PHN pairs.\n\n"
                "Starting inference..."
            )

            self.inference_set_file_list_widget.clear()

            QApplication.processEvents()

            self.inference_set_results = {}

            for file_index, (
                wav_path,
                phn_path
            ) in enumerate(
                file_pairs,
                start=1
            ):
                waveform, sample_rate = torchaudio.load(
                    str(wav_path)
                )

                # Optional but recommended check.
                expected_sample_rate = self.run_config.get(
                    "sample_rate",
                    sample_rate
                )

                if sample_rate != expected_sample_rate:
                    waveform = torchaudio.functional.resample(
                        waveform,
                        orig_freq=sample_rate,
                        new_freq=expected_sample_rate
                    )

                target_phonemes = (
                    self.load_target_phonemes_from_file(
                        phn_path
                    )
                )

                result = self.infer_single_waveform(
                    waveform=waveform,
                    target_phonemes=target_phonemes
                )

                relative_path = wav_path.relative_to(
                    self.test_directory_path
                )

                result_key = str(relative_path)

                result["wav_path"] = wav_path
                result["phn_path"] = phn_path

                self.inference_set_results[
                    result_key
                ] = result

                # Show progress.
                self.inference_set_result_view.setPlainText(
                    f"Running inference...\n\n"
                    f"File {file_index} / {len(file_pairs)}\n\n"
                    f"{relative_path}"
                )

                QApplication.processEvents()

            self.inference_set_statistics = (
                self.calculate_inference_set_statistics()
            )

            self.fill_inference_set_file_list()
            self.show_inference_set_summary()

        except (
            FileNotFoundError,
            KeyError,
            OSError,
            TypeError,
            ValueError,
            RuntimeError
        ) as error:
            QMessageBox.critical(
                self,
                "Inference Set Failed",
                "Inference on the test set could not be completed.\n\n"
                f"{type(error).__name__}: {error}"
            )

    def show_selected_inference_set_result(self,
                                           item) -> None:
        """
        Display the aligned result for the clicked WAV file.
        """

        result_key = item.text()

        result = self.inference_set_results.get(
            result_key
        )

        if result is None:
            return

        self.current_inference_set_key = result_key

        statistics_text = self.format_inference_statistics(
            result["statistics"]
        )

        alignment_text = self.format_aligned_phonemes(
            aligned_predictions=result[
                "aligned_predictions"
            ],
            aligned_targets=result[
                "aligned_targets"
            ]
        )

        self.inference_set_result_title.setText(
            result_key
        )

        result_text = (
            f"{statistics_text}\n\n"
            "ALIGNED PHONEMES\n"
            "================\n\n"
            f"{alignment_text}"
        )

        self.inference_set_result_view.setPlainText(
            result_text
        )

    def calculate_inference_set_statistics(self) -> dict[str, Any]:
        """
        Calculate aggregate statistics over all evaluated files.
        """

        total_files = len(
            self.inference_set_results
        )

        total_target_phonemes = 0
        total_predicted_phonemes = 0

        total_correct = 0
        total_substitutions = 0
        total_insertions = 0
        total_deletions = 0
        total_edit_distance = 0

        exact_sequence_matches = 0

        for result in self.inference_set_results.values():
            statistics = result["statistics"]

            total_target_phonemes += (
                statistics["total_target_phonemes"]
            )

            total_predicted_phonemes += (
                statistics["total_predicted_phonemes"]
            )

            total_correct += (
                statistics["correct_phonemes"]
            )

            total_substitutions += (
                statistics["substitutions"]
            )

            total_insertions += (
                statistics["insertions"]
            )

            total_deletions += (
                statistics["deletions"]
            )

            total_edit_distance += (
                statistics["edit_distance"]
            )

            if statistics["exact_sequence_match"]:
                exact_sequence_matches += 1

        corpus_per = (
            total_edit_distance
            / total_target_phonemes
            if total_target_phonemes > 0
            else 0.0
        )

        correct_percentage = (
            total_correct
            / total_target_phonemes
            * 100
            if total_target_phonemes > 0
            else 0.0
        )

        wrong_target_phonemes = (
            total_substitutions
            + total_deletions
        )

        wrong_percentage = (
            wrong_target_phonemes
            / total_target_phonemes
            * 100
            if total_target_phonemes > 0
            else 0.0
        )

        exact_match_percentage = (
            exact_sequence_matches
            / total_files
            * 100
            if total_files > 0
            else 0.0
        )

        return {
            "total_files": total_files,
            "total_target_phonemes": total_target_phonemes,
            "total_predicted_phonemes": total_predicted_phonemes,
            "correct_phonemes": total_correct,
            "wrong_target_phonemes": wrong_target_phonemes,
            "substitutions": total_substitutions,
            "insertions": total_insertions,
            "deletions": total_deletions,
            "edit_distance": total_edit_distance,
            "per": corpus_per,
            "correct_percentage": correct_percentage,
            "wrong_percentage": wrong_percentage,
            "exact_sequence_matches": exact_sequence_matches,
            "exact_match_percentage": exact_match_percentage
        }

    def prepare_inference_input(self) -> tuple[torch.Tensor, torch.Tensor]:

        if self.waveform_tensor is None:
            raise ValueError(
                "No WAV file has been loaded."
            )

        return self.prepare_waveform_for_inference(
            self.waveform_tensor
        )

    def run_inference(self) -> None:
        """
        Run phoneme inference on the currently loaded WAV file.
        """

        if self.loaded_model is None:
            QMessageBox.warning(
                self,
                "No Model Imported",
                "Import a trained model before running inference."
            )
            return

        if self.waveform_tensor is None:
            QMessageBox.warning(
                self,
                "No WAV Loaded",
                "Open a WAV file before running inference."
            )
            return

        try:
            model_input, input_lengths = (
                self.prepare_inference_input()
            )

            self.loaded_model.eval()

            with torch.inference_mode():
                logits = self.loaded_model(
                    model_input
                )

                # Some newer models may return:
                # logits, output_lengths
                if isinstance(logits, tuple):
                    logits, model_output_lengths = logits

                    valid_output_length = int(
                        model_output_lengths[0].item()
                    )
                else:
                    valid_output_length = logits.shape[1]

                log_probs = logits.log_softmax(
                    dim=-1
                )

                prediction_ids = torch.argmax(
                    log_probs,
                    dim=-1
                )[0, :valid_output_length].cpu().tolist()

            blank_index = self.run_config.get(
                "blank_index",
                0
            )

            decoded_ids = self.ctc_greedy_decode_ids(
                prediction_ids=prediction_ids,
                blank_index=blank_index
            )

            predicted_phonemes = [
                self.idx_to_phoneme_vocab[
                    phoneme_id
                ]
                for phoneme_id in decoded_ids
            ]

            target_phonemes = (
                self.get_target_phonemes_for_inference()
            )

            (
                aligned_predictions,
                aligned_targets,
                operation_counts
            ) = self.align_phoneme_sequences(
                predicted=predicted_phonemes,
                target=target_phonemes
            )

            statistics = (
                self.calculate_inference_statistics(
                    predicted=predicted_phonemes,
                    target=target_phonemes,
                    operation_counts=operation_counts
                )
            )

            self.predicted_phonemes = (
                predicted_phonemes
            )
            self.target_phonemes = target_phonemes

            self.aligned_predictions = (
                aligned_predictions
            )
            self.aligned_targets = (
                aligned_targets
            )

            self.inference_statistics = statistics

            self.inference_statistics_view.setPlainText(
                self.format_inference_statistics(
                    statistics
                )
            )

            self.inference_results_view.setPlainText(
                self.format_aligned_phonemes(
                    aligned_predictions=aligned_predictions,
                    aligned_targets=aligned_targets
                )
            )

            self.show_inference_view()

        except (
            KeyError,
            TypeError,
            ValueError,
            RuntimeError
        ) as error:
            QMessageBox.critical(
                self,
                "Inference Failed",
                "Inference could not be completed.\n\n"
                f"{type(error).__name__}: {error}"
            )

    def format_aligned_phonemes(self,
                                aligned_predictions: list[str],
                                aligned_targets: list[str]) -> str:
        """
        Format aligned target and predicted phonemes for display in the
        inference results panel.

        Each aligned target phoneme is shown next to its corresponding
        predicted phoneme.

        A dash means that a phoneme is missing because of an insertion
        or deletion.
        """

        result_lines = []

        for index, (
            target_phoneme,
            predicted_phoneme
        ) in enumerate(
            zip(
                aligned_targets,
                aligned_predictions
            ),
            start=1
        ):
            if (
                target_phoneme == predicted_phoneme
                and target_phoneme != "-"
            ):
                status = "CORRECT"
            else:
                status = "WRONG"

            result_lines.append(
                f"{index:4d} | "
                f"TGT: {target_phoneme:<8} | "
                f"PRED: {predicted_phoneme:<8} | "
                f"{status}"
            )

        return "\n".join(result_lines)

    def format_inference_statistics(self,
                                    statistics: dict[str, Any]) -> str:
        """
        Format inference statistics for display.
        """

        exact_match_text = (
            "Yes"
            if statistics["exact_sequence_match"]
            else "No"
        )

        return (
            f"Model device:\n"
            f"{self.model_device}\n\n"

            f"Target phonemes:\n"
            f"{statistics['total_target_phonemes']}\n\n"

            f"Predicted phonemes:\n"
            f"{statistics['total_predicted_phonemes']}\n\n"

            f"Correct phonemes:\n"
            f"{statistics['correct_phonemes']} / "
            f"{statistics['total_target_phonemes']}\n"
            f"{statistics['correct_percentage']:.2f}%\n\n"

            f"Wrong target phonemes:\n"
            f"{statistics['wrong_target_phonemes']} / "
            f"{statistics['total_target_phonemes']}\n"
            f"{statistics['wrong_percentage']:.2f}%\n\n"

            f"Substitutions:\n"
            f"{statistics['substitutions']}\n\n"

            f"Deletions:\n"
            f"{statistics['deletions']}\n\n"

            f"Insertions:\n"
            f"{statistics['insertions']}\n\n"

            f"Edit distance:\n"
            f"{statistics['edit_distance']}\n\n"

            f"Phoneme Error Rate:\n"
            f"{statistics['per']:.4f}\n"
            f"{statistics['per'] * 100:.2f}%\n\n"

            f"Exact sequence match:\n"
            f"{exact_match_text}"
        )

    def show_selected_model_file(self, item) -> None:
        """
        Display the selected model result file in the right panel.

        Parameters
        ----------
        item:
            The clicked QListWidgetItem.
        """

        filename = item.text()

        content = self.model_info_contents.get(
            filename
        )

        if content is None:
            self.selected_model_file_label.setText(
                filename
            )

            self.model_file_content_view.setPlainText(
                "No displayable content is available."
            )

            return

        self.selected_model_file_label.setText(
            filename
        )

        self.model_file_content_view.setPlainText(
            content
        )

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
    app = QApplication(sys.argv)

    # create the window
    window = MainWindow(wav_path_choose=False)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
