"""Simple video viewer: pick a video from the configured folder, preview it."""

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from modules.qt.detyolo import YoloDetector
from modules.qt.detyolo import list_models as list_det_models
from modules.stiqy.segmodel import SegDetector
from modules.stiqy.segmodel import list_models as list_seg_models
from modules.video_loader import deinterlace, list_videos, load_config
from modules.video_player import VideoPlayer


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Viewer")
        self.resize(1200, 700)

        self.config = load_config()
        self.folder = self.config["video_folder"]

        self.player = VideoPlayer()
        self.player.processor = self.process_frame
        self.detector = SegDetector()
        self.det_detector = YoloDetector()

        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.West)
        self.tabs.addTab(self._build_video_tab(), "Video")
        self.tabs.addTab(self._build_stiqy_tab(), "Stiqy")
        self.tabs.addTab(self._build_qt_tab(), "Qt")
        self.tabs.setFixedWidth(300)

        layout = QHBoxLayout()
        layout.addWidget(self.tabs)
        layout.addWidget(self.player, 1)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.refresh_videos()
        self.refresh_models()
        self.refresh_det_models()

    def _build_video_tab(self):
        self.folder_label = QLabel()
        self.folder_label.setWordWrap(True)

        browse_button = QPushButton("Change Folder...")
        browse_button.clicked.connect(self.choose_folder)

        self.video_list = QListWidget()
        self.video_list.currentRowChanged.connect(self.open_selected)

        self.deint_check = QCheckBox("Deinterlace")
        self.deint_check.toggled.connect(self.player.refresh)

        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("<b>Videos</b>"))
        layout.addWidget(self.folder_label)
        layout.addWidget(browse_button)
        layout.addWidget(self.video_list, 1)
        layout.addWidget(self.deint_check)
        return tab

    def _build_stiqy_tab(self):
        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self.model_selected)

        self.detect_check = QCheckBox("Detection ON")
        self.detect_check.setEnabled(False)
        self.detect_check.toggled.connect(self.player.refresh)

        self.mask_check = QCheckBox("Show segmentation")
        self.mask_check.setChecked(True)
        self.mask_check.setEnabled(False)
        self.mask_check.toggled.connect(self.player.refresh)

        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.setEnabled(False)
        self.opacity_slider.valueChanged.connect(self.player.refresh)

        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("<b>Segmentation</b>"))
        layout.addWidget(self.model_combo)
        layout.addWidget(self.detect_check)
        layout.addWidget(self.mask_check)
        layout.addWidget(QLabel("Mask opacity"))
        layout.addWidget(self.opacity_slider)
        layout.addStretch(1)
        return tab

    def _build_qt_tab(self):
        self.det_combo = QComboBox()
        self.det_combo.currentIndexChanged.connect(self.det_model_selected)

        self.det_check = QCheckBox("Detection ON")
        self.det_check.setEnabled(False)
        self.det_check.toggled.connect(self.player.refresh)

        self.det_label_check = QCheckBox("Show labels")
        self.det_label_check.setChecked(True)
        self.det_label_check.setEnabled(False)
        self.det_label_check.toggled.connect(self.player.refresh)

        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("<b>YOLO Detection</b>"))
        layout.addWidget(self.det_combo)
        layout.addWidget(self.det_check)
        layout.addWidget(self.det_label_check)
        layout.addStretch(1)
        return tab

    def refresh_det_models(self):
        folder = self.config["qt_det_model_folder"]
        self.det_models = list_det_models(folder)
        self.det_combo.blockSignals(True)
        self.det_combo.clear()
        self.det_combo.addItem("-- select model --")
        self.det_combo.addItems([name for name, _ in self.det_models])
        self.det_combo.blockSignals(False)
        if not self.det_models:
            self.statusBar().showMessage(f"No .pt weights found in {folder}")

    def det_model_selected(self, index):
        if index < 1:
            self.det_check.setChecked(False)
            for widget in (self.det_check, self.det_label_check):
                widget.setEnabled(False)
            self.player.refresh()
            return
        name, path = self.det_models[index - 1]
        self.det_detector.load(path)
        for widget in (self.det_check, self.det_label_check):
            widget.setEnabled(True)
        self.det_check.setChecked(True)
        self.statusBar().showMessage(f"Loaded model {name}")
        self.player.refresh()

    def refresh_videos(self):
        self.folder_label.setText(self.folder or "(no folder set)")
        self.videos = list_videos(self.folder, self.config["extensions"])
        self.video_list.clear()
        self.video_list.addItems([name for name, _ in self.videos])
        if not self.videos:
            self.statusBar().showMessage(f"No videos found in {self.folder}")

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select video folder", self.folder)
        if folder:
            self.folder = folder
            self.refresh_videos()

    def refresh_models(self):
        folder = self.config["stiqy_seg_model_folder"]
        self.models = list_seg_models(folder)
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItem("-- select model --")
        self.model_combo.addItems([name for name, _ in self.models])
        self.model_combo.blockSignals(False)
        if not self.models:
            self.statusBar().showMessage(f"No .pt weights found in {folder}")

    def model_selected(self, index):
        if index < 1:
            self.detect_check.setChecked(False)
            for widget in (self.detect_check, self.mask_check, self.opacity_slider):
                widget.setEnabled(False)
            self.player.refresh()
            return
        name, path = self.models[index - 1]
        self.detector.load(path)
        for widget in (self.detect_check, self.mask_check, self.opacity_slider):
            widget.setEnabled(True)
        self.detect_check.setChecked(True)
        self.statusBar().showMessage(f"Loaded model {name}")
        self.player.refresh()

    def process_frame(self, frame):
        if self.deint_check.isChecked():
            frame = deinterlace(frame)
        if self.detector.model is not None and self.detect_check.isChecked():
            frame = self.detector.predict(
                frame,
                show_masks=self.mask_check.isChecked(),
                opacity=self.opacity_slider.value() / 100.0,
            )
        if self.det_detector.model is not None and self.det_check.isChecked():
            frame = self.det_detector.predict(
                frame, show_labels=self.det_label_check.isChecked()
            )
        return frame

    def open_selected(self, row):
        if 0 <= row < len(self.videos):
            name, path = self.videos[row]
            self.player.show_video(path)
            self.statusBar().showMessage(path)

    def closeEvent(self, event):
        self.player.close()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
