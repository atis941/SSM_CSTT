
# for model imports
import json
import torch
from typing import Any
from visualize import load_audio, plot_waveform
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QToolBar,
    QWidget, QVBoxLayout, QLabel,
    QHBoxLayout, QFrame, QFileDialog,
    QScrollArea, QListWidget, QPushButton, QMessageBox)
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

        # model and training relevant attributes
        self.model_directory_path: str | None = None

        self.run_config: dict[str, Any] | None = None
        self.losses_and_accuracies: dict[str, Any] | None = None

        self.training_log_loss_and_per: str | None = None
        self.training_log_predictions: str | None = None

        self.model_state_dict: dict[str, torch.Tensor] | None = None
        self.loaded_model: torch.nn.Module | None = None

        # For GUI inference, CPU is generally the simplest choice.
        self.model_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

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
        self.show_spectrogram_button.triggered.connect(
            self.show_spectrogram_view)
        self.toolbar.addAction(self.show_spectrogram_button)

        self.import_model_button = QAction("Import Model", self)
        self.import_model_button.triggered.connect(self.import_model)
        self.toolbar.addAction(self.import_model_button)

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
        self.phoneme_list_widget.itemDoubleClicked.connect(
            self.zoom_to_selected_phoneme)
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
        # insert the toolbar between the Waveform canvas and the transcript label
        self.right_layout.insertWidget(1, self.plot_toolbar)

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
        # insert the toolbar between the Waveform canvas and the transcript label
        self.right_layout.insertWidget(1, self.plot_toolbar)

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
            self.losses_and_accuracies = losses_and_accuracies

            self.model_state_dict = model_state_dict
            self.loaded_model = loaded_model

            self.training_log_loss_and_per = (
                training_log_loss_and_per
            )
            self.training_log_predictions = (
                training_log_predictions
            )

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
