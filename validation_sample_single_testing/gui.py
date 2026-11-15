
import sys
import numpy as np
import matplotlib.pyplot as plt

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QTabWidget,
    QScrollArea,
    QLabel
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class InferenceVisualizationGUI(QMainWindow):

    def __init__(
        self,
        sample_probabilities,
        phoneme_intervals,
        sample_name,
        class_labels,
        recording_duration,
        blank_idx=0
    ):
        super().__init__()

        self.setWindowTitle(
            f"Inference Visualization — {sample_name}"
        )
        self.resize(1400, 900)

        self.probabilities = sample_probabilities
        self.intervals = phoneme_intervals
        self.sample_name = sample_name
        self.class_labels = class_labels
        self.duration = recording_duration
        self.blank_idx = blank_idx

        self.T, self.num_classes = self.probabilities.shape

        self.time_edges = np.linspace(
            0, self.duration, self.T + 1
        )

        self.time_axis = (
            self.time_edges[:-1] + self.time_edges[1:]
        ) / 2

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.create_classifier_tab()
        self.create_ctc_tab()
        self.create_top3_tab()

    
    def create_classifier_tab(self):

        figure = Figure(figsize=(14, 9))
        canvas = FigureCanvasQTAgg(figure)

        # Create a GridSpec with two rows and two columns.
        # The first column contains both plots.
        # The second column is reserved for the colorbar.
        gs = figure.add_gridspec(
            2, 2,
            width_ratios=[30, 1],
            height_ratios=[1, 9],
            hspace=0.05,
            wspace=0.08
        )

        # Ground-truth phoneme axis
        ax_phoneme = figure.add_subplot(gs[0, 0])

        # Classifier probability heatmap axis
        ax_map = figure.add_subplot(
            gs[1, 0],
            sharex=ax_phoneme
        )

        # Dedicated colorbar axis
        ax_colorbar = figure.add_subplot(gs[1, 1])

        # ---------------------------------------------------------
        # Ground-truth phoneme intervals
        # ---------------------------------------------------------

        for interval in self.intervals:

            start = interval["start_time"]
            end = interval["end_time"]
            center = (start + end) / 2

            # Draw phoneme boundary in upper plot
            ax_phoneme.axvline(
                start,
                linestyle="--",
                color="steelblue",
                alpha=0.5
            )

            # Draw phoneme label
            ax_phoneme.text(
                center,
                0.5,
                interval["phoneme"],
                rotation=90,
                ha="center",
                va="center",
                fontsize=8
            )

            # Draw the same boundary in the heatmap
            ax_map.axvline(
                start,
                linestyle="--",
                color="white",
                alpha=0.3
            )

        ax_phoneme.set_ylim(0, 1)
        ax_phoneme.set_yticks([])

        ax_phoneme.set_title(
            f"Classifier probabilities — {self.sample_name}"
        )

        # ---------------------------------------------------------
        # Classifier probability heatmap
        # ---------------------------------------------------------

        image = ax_map.imshow(
            self.probabilities.T,
            aspect="auto",
            origin="lower",
            extent=[
                0,
                self.duration,
                -0.5,
                self.num_classes - 0.5
            ],
            vmin=0,
            vmax=1,
            cmap="viridis",
            interpolation="nearest"
        )

        ax_map.set_yticks(np.arange(self.num_classes))
        ax_map.set_yticklabels(
            self.class_labels,
            fontsize=7
        )

        ax_map.set_xlabel("Time [s]")
        ax_map.set_ylabel("CTC class")

        # Both plots have exactly the same time limits.
        ax_map.set_xlim(0, self.duration)

        # Hide x-axis labels in the upper plot.
        ax_phoneme.tick_params(
            axis="x",
            labelbottom=False
        )

        # ---------------------------------------------------------
        # Colorbar
        # ---------------------------------------------------------

        figure.colorbar(
            image,
            cax=ax_colorbar,
            label="Softmax probability"
        )

        # ---------------------------------------------------------
        # Layout
        # ---------------------------------------------------------

        figure.subplots_adjust(
            left=0.09,
            right=0.92,
            top=0.92,
            bottom=0.08
        )

        self.tabs.addTab(
            canvas,
            "Classifier Probabilities"
        )


    def create_ctc_tab(self):

        figure = Figure(figsize=(14, 8))
        canvas = FigureCanvasQTAgg(figure)

        ax_phoneme, ax_pred, ax_blank = figure.subplots(
            3, 1,
            sharex=True,
            gridspec_kw={
                "height_ratios": [1, 2, 3],
                "hspace": 0.08
            }
        )

        frame_ids = self.probabilities.argmax(axis=-1)
        blank_probs = self.probabilities[:, self.blank_idx]

        nonblank_mask = frame_ids != self.blank_idx
        nonblank_ids = frame_ids[nonblank_mask]

        for interval in self.intervals:
            start = interval["start_time"]
            end = interval["end_time"]

            ax_phoneme.axvline(
                start, linestyle="--", alpha=0.4
            )

            ax_phoneme.text(
                (start + end) / 2,
                0.5,
                interval["phoneme"],
                rotation=90,
                ha="center",
                va="center",
                fontsize=8
            )

            for ax in (ax_pred, ax_blank):
                ax.axvline(
                    start,
                    linestyle="--",
                    color="steelblue",
                    alpha=0.2
                )

        ax_phoneme.set_ylim(0, 1)
        ax_phoneme.set_yticks([])
        ax_phoneme.set_title(
            f"CTC predictions — {self.sample_name}"
        )

        ax_pred.axhline(
            self.blank_idx,
            color="gray",
            linestyle="--",
            alpha=0.4
        )

        ax_pred.scatter(
            self.time_axis[nonblank_mask],
            nonblank_ids,
            s=16
        )

        visible_ids = sorted(
            set(nonblank_ids.tolist()) | {self.blank_idx}
        )

        ax_pred.set_yticks(visible_ids)
        ax_pred.set_yticklabels(
            [self.class_labels[i] for i in visible_ids]
        )
        ax_pred.set_ylabel("Predicted class")
        ax_pred.grid(alpha=0.2)

        # Bottom: total nonblank probability
        nonblank_probs = 1.0 - self.probabilities[:, self.blank_idx]

        ax_blank.plot(
            self.time_axis,
            nonblank_probs,
            color="tab:blue",
            linewidth=1,
            label="P(nonblank)"
        )

        ax_blank.set_ylim(0, 1.05)
        ax_blank.set_ylabel("Nonblank probability")
        ax_blank.set_xlabel("Time [s]")
        ax_blank.legend()
        ax_blank.grid(alpha=0.3)
        ax_blank.set_xlim(0, self.duration)

        figure.subplots_adjust(
            left=0.09, right=0.97,
            top=0.92, bottom=0.08
        )

        self.tabs.addTab(canvas, "CTC Predictions")

    def create_top3_tab(self):

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        container = QWidget()
        layout = QVBoxLayout(container)

        for index, interval in enumerate(self.intervals):

            start = interval["start_time"]
            end = interval["end_time"]
            target = interval["phoneme"]

            mask = (
                (self.time_axis >= start) &
                (self.time_axis < end)
            )

            if not np.any(mask):
                continue

            mean_probs = self.probabilities[mask].mean(axis=0)

            top_ids = np.argsort(mean_probs)[-3:][::-1]
            top_labels = [
                self.class_labels[int(i)]
                for i in top_ids
            ]
            top_values = mean_probs[top_ids]

            figure = Figure(figsize=(9, 2.5))
            canvas = FigureCanvasQTAgg(figure)

            # Fixed height allows vertical scrolling.
            canvas.setFixedHeight(230)

            ax = figure.add_subplot(111)

            ax.barh(
                top_labels[::-1],
                top_values[::-1],
                color="steelblue"
            )

            ax.set_xlim(0, 1)
            ax.set_xlabel("Mean softmax probability")

            ax.set_title(
                f"#{index + 1} — Target: {target} "
                f"({start:.3f}–{end:.3f} s)",
                loc="left",
                fontsize=10
            )

            ax.grid(axis="x", alpha=0.2)

            for bar in ax.patches:
                width = bar.get_width()

                ax.text(
                    min(width + 0.01, 0.98),
                    bar.get_y() + bar.get_height() / 2,
                    f"{width:.3f}",
                    va="center",
                    ha="left" if width < 0.9 else "right",
                    fontsize=9
                )

            figure.subplots_adjust(
                left=0.15,
                right=0.95,
                top=0.82,
                bottom=0.25
            )

            layout.addWidget(canvas)

        layout.addStretch()

        scroll_area.setWidget(container)

        self.tabs.addTab(scroll_area, "Top-3 Classes")

