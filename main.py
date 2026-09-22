"""Simple video viewer: pick a video from the configured folder, preview it."""

import os
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
    QLineEdit,
    QSlider,
    QToolButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from modules.frame_tagger import save_tag
from modules.product.qt.detyolo import YoloDetector
from modules.product.qt.detyolo import list_models as list_det_models
from modules.product.stiqy.rvmseg import RvmSegmenter
from modules.product.stiqy.rvmseg import list_models as list_rvm_models
from modules.product.stiqy.sam3seg import Sam3Segmenter
from modules.product.stiqy.sam3seg import list_models as list_sam_models
from modules.product.stiqy.segmodel import SegDetector
from modules.product.stiqy.segmodel import list_models as list_seg_models
from modules.video_loader import (
    deinterlace,
    ensure_local_config,
    list_videos,
    load_config,
)
from modules.video_player import VideoPlayer

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


class MainWindow(QMainWindow):
    TAG_PANEL_WIDTH = 220

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Viewer")
        self.resize(1200, 700)

        self.local_config_path = ensure_local_config(CONFIG_PATH)
        self.config = load_config(CONFIG_PATH)
        self.folder = self.config["video_folder"]
        self.current_video = ""

        self.player = VideoPlayer()
        self.player.processor = self.process_frame
        self.detector = SegDetector()
        self.det_detector = YoloDetector()
        self.sam = Sam3Segmenter()
        self.rvm = RvmSegmenter()

        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.West)
        self.tabs.addTab(self._build_video_tab(), "Video")
        self.tabs.addTab(self._build_stiqy_tab(), "Stiqy")
        self.tabs.addTab(self._build_qt_tab(), "Qt")
        self.tabs.setFixedWidth(300)

        layout = QHBoxLayout()
        layout.addWidget(self.tabs)
        layout.addWidget(self.player, 1)
        layout.addWidget(self._build_right_panel())

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.refresh_videos()
        self.refresh_models()
        self.refresh_det_models()
        self.refresh_sam_models()
        self.refresh_rvm_models()

    def _build_right_panel(self):
        """Right edge column: config reload + collapsible 'Frame Tagging' box."""
        self.reload_button = QPushButton("Reload Config")
        self.reload_button.clicked.connect(self.reload_config)

        self.tag_toggle = QToolButton()
        self.tag_toggle.setText("Frame Tagging")
        self.tag_toggle.setCheckable(True)
        self.tag_toggle.setChecked(True)
        self.tag_toggle.setArrowType(Qt.RightArrow)
        self.tag_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.tag_toggle.toggled.connect(self._toggle_tag_panel)

        self.tag_note = QLineEdit()
        self.tag_note.setPlaceholderText("note (optional)")
        self.tag_note.returnPressed.connect(self.tag_frame)

        tag_button = QPushButton("Tag")
        tag_button.clicked.connect(self.tag_frame)

        self.tag_status = QLabel(f"output: {self.config['output_folder']}")
        self.tag_status.setWordWrap(True)

        self.tag_body = QWidget()
        body = QVBoxLayout(self.tag_body)
        body.setContentsMargins(0, 0, 0, 0)
        body.addWidget(QLabel("Note"))
        body.addWidget(self.tag_note)
        body.addWidget(tag_button)
        body.addWidget(self.tag_status)
        body.addStretch(1)

        self.tag_panel = QWidget()
        layout = QVBoxLayout(self.tag_panel)
        layout.setContentsMargins(4, 0, 0, 0)
        layout.addWidget(self.reload_button)
        layout.addWidget(self.tag_toggle)
        layout.addWidget(self.tag_body, 1)
        self.tag_panel.setFixedWidth(self.TAG_PANEL_WIDTH)
        return self.tag_panel

    def reload_config(self):
        """Re-read config.json and refresh every folder-backed list in place."""
        keep = (
            self.video_list.currentItem().text() if self.video_list.currentItem() else "",
            self.model_combo.currentText(),
            self.sam_combo.currentText(),
            self.det_combo.currentText(),
            self.rvm_combo.currentText(),
        )
        self.config = load_config(CONFIG_PATH)
        self.folder = self.config["video_folder"]

        self.refresh_videos()
        self.refresh_models()
        self.refresh_det_models()
        self.refresh_sam_models()
        self.refresh_rvm_models()

        items = [self.video_list.item(i).text() for i in range(self.video_list.count())]
        if keep[0] in items:
            self.video_list.blockSignals(True)
            self.video_list.setCurrentRow(items.index(keep[0]))
            self.video_list.blockSignals(False)
        for combo, text in zip(
            (self.model_combo, self.sam_combo, self.det_combo, self.rvm_combo), keep[1:]
        ):
            index = combo.findText(text)
            if index > 0:
                combo.blockSignals(True)
                combo.setCurrentIndex(index)
                combo.blockSignals(False)

        self.tag_status.setText(f"output: {self.config['output_folder']}")
        self.statusBar().showMessage(f"config reloaded from {self.local_config_path}")

    def _toggle_tag_panel(self, shown):
        self.tag_body.setVisible(shown)
        self.reload_button.setVisible(shown)
        self.tag_toggle.setText("Frame Tagging" if shown else "")
        self.tag_toggle.setArrowType(Qt.RightArrow if shown else Qt.LeftArrow)
        self.tag_panel.setFixedWidth(self.TAG_PANEL_WIDTH if shown else 32)

    def active_overlays(self):
        """Names of everything currently drawn on the frame, for the tag log."""
        active = []
        if self.deint_check.isChecked():
            active.append("deinterlace")
        if self.detector.model is not None and self.detect_check.isChecked():
            active.append(f"stiqy-seg:{self.model_combo.currentText()}")
        if self.sam.predictor is not None and self.sam_check.isChecked():
            active.append(
                f"stiqy-sam:{self.sam_combo.currentText()}[{self.sam_prompt.text()}]"
            )
        if self.rvm.model is not None and self.rvm_check.isChecked():
            mode = "crops" if self.rvm_crops.isChecked() else (
                "native" if self.rvm.native else "512x256"
            )
            active.append(
                f"stiqy-rvm:{self.rvm_combo.currentText()}[{self.rvm_view.currentText()},{mode}]"
            )
        if self.det_detector.model is not None and self.det_check.isChecked():
            active.append(f"qt-det:{self.det_combo.currentText()}")
        return " + ".join(active) or "none"

    def tag_frame(self):
        raw = self.player._raw_frame
        if raw is None:
            self.tag_status.setText("no frame to tag")
            return
        self.player.pause()
        original, overlay = save_tag(
            self.config["output_folder"],
            self.current_video,
            self.player.current_frame,
            raw,
            self.player.processed_frame,
            overlays=self.active_overlays(),
            note=self.tag_note.text(),
        )
        self.tag_note.clear()
        self.tag_status.setText(f"saved frame {self.player.current_frame} -> {overlay}")
        self.statusBar().showMessage(f"tagged {original}")

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

        self.box_check = QCheckBox("Show boxes")
        self.box_check.setChecked(True)
        self.box_check.setEnabled(False)
        self.box_check.toggled.connect(self.player.refresh)

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
        layout.addWidget(self.box_check)
        layout.addWidget(QLabel("Mask opacity"))
        layout.addWidget(self.opacity_slider)

        self.sam_combo = QComboBox()
        self.sam_combo.currentIndexChanged.connect(self.sam_model_selected)

        self.sam_prompt = QLineEdit()
        self.sam_prompt.setPlaceholderText("person, cricket bat, ball")
        self.sam_prompt.editingFinished.connect(self.sam_prompt_changed)

        self.sam_check = QCheckBox("Segment ON")
        self.sam_check.setEnabled(False)
        self.sam_check.toggled.connect(self.player.refresh)

        layout.addSpacing(12)
        layout.addWidget(QLabel("<b>SAM 3.1 (text prompt)</b>"))
        layout.addWidget(self.sam_combo)
        layout.addWidget(QLabel("Prompt (comma separated)"))
        layout.addWidget(self.sam_prompt)
        layout.addWidget(self.sam_check)

        self.rvm_combo = QComboBox()
        self.rvm_combo.currentIndexChanged.connect(self.rvm_model_selected)

        self.rvm_check = QCheckBox("Segmentation ON")
        self.rvm_check.setEnabled(False)
        self.rvm_check.toggled.connect(self.player.refresh)

        self.rvm_view = QComboBox()
        self.rvm_view.addItems(["overlay", "mask"])
        self.rvm_view.currentIndexChanged.connect(self.player.refresh)

        self.rvm_native = QCheckBox("Native resolution")
        self.rvm_native.toggled.connect(self.rvm_native_changed)

        self.rvm_crops = QCheckBox("Use YOLO person boxes (crops)")
        self.rvm_crops.setChecked(True)
        self.rvm_crops.toggled.connect(self.rvm_crops_changed)

        layout.addSpacing(12)
        layout.addWidget(QLabel("<b>RVM segmentation</b>"))
        layout.addWidget(self.rvm_combo)
        layout.addWidget(self.rvm_check)
        layout.addWidget(self.rvm_view)
        layout.addWidget(self.rvm_native)
        layout.addWidget(self.rvm_crops)
        layout.addStretch(1)
        return tab

    def refresh_rvm_models(self):
        folder = self.config["stiqy_rvm_ckpt_folder"]
        self.rvm_models = list_rvm_models(folder)
        self.rvm_combo.blockSignals(True)
        self.rvm_combo.clear()
        self.rvm_combo.addItem("-- select checkpoint --")
        self.rvm_combo.addItems([name for name, _ in self.rvm_models])
        self.rvm_combo.blockSignals(False)
        if not self.rvm_models:
            self.statusBar().showMessage(f"No .pth checkpoints found in {folder}")

    def rvm_model_selected(self, index):
        if index < 1:
            self.rvm_check.setChecked(False)
            self.rvm_check.setEnabled(False)
            self.player.refresh()
            return
        name, path = self.rvm_models[index - 1]
        self.rvm.load(path)
        self.rvm.native = self.rvm_native.isChecked()
        self.rvm_check.setEnabled(True)
        self.rvm_check.setChecked(True)
        self.statusBar().showMessage(f"Loaded checkpoint {name}")
        self.player.refresh()

    def rvm_crops_changed(self, _):
        self.rvm.reset_state()
        if self.rvm_check.isChecked():
            self.player.refresh()

    def rvm_native_changed(self, native):
        self.rvm.native = native
        self.rvm.reset_state()
        if self.rvm_check.isChecked():
            self.player.refresh()

    def refresh_sam_models(self):
        folder = self.config["stiqy_sam_model_folder"]
        self.sam_models = list_sam_models(folder)
        self.sam_combo.blockSignals(True)
        self.sam_combo.clear()
        self.sam_combo.addItem("-- select model --")
        self.sam_combo.addItems([name for name, _ in self.sam_models])
        self.sam_combo.blockSignals(False)
        if not self.sam_models:
            self.statusBar().showMessage(f"No sam*.pt weights found in {folder}")

    def sam_model_selected(self, index):
        if index < 1:
            self.sam_check.setChecked(False)
            self.sam_check.setEnabled(False)
            self.player.refresh()
            return
        name, path = self.sam_models[index - 1]
        self.statusBar().showMessage(f"Loading {name}... this takes a few seconds")
        QApplication.processEvents()
        self.sam.load(path)
        self.sam.set_prompt(self.sam_prompt.text())
        self.sam_check.setEnabled(True)
        self.opacity_slider.setEnabled(True)
        self.sam_check.setChecked(bool(self.sam.texts))
        self.statusBar().showMessage(f"Loaded model {name}")
        self.player.refresh()

    def sam_prompt_changed(self):
        self.sam.set_prompt(self.sam_prompt.text())
        if self.sam_check.isChecked():
            self.player.refresh()

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
            for widget in (
                self.detect_check, self.mask_check, self.box_check, self.opacity_slider
            ):
                widget.setEnabled(False)
            self.player.refresh()
            return
        name, path = self.models[index - 1]
        self.detector.load(path)
        for widget in (
            self.detect_check, self.mask_check, self.box_check, self.opacity_slider
        ):
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
                show_boxes=self.box_check.isChecked(),
                show_masks=self.mask_check.isChecked(),
                opacity=self.opacity_slider.value() / 100.0,
            )
        if self.sam.predictor is not None and self.sam_check.isChecked():
            frame = self.sam.predict(frame, opacity=self.opacity_slider.value() / 100.0)
        if self.rvm.model is not None and self.rvm_check.isChecked():
            boxes = None
            if self.rvm_crops.isChecked():
                boxes = self.detector.boxes(frame) if self.detector.model else []
            frame = self.rvm.predict(
                frame,
                frame_index=self.player.current_frame,
                opacity=self.opacity_slider.value() / 100.0,
                view=self.rvm_view.currentText(),
                boxes=boxes,
            )
        if self.det_detector.model is not None and self.det_check.isChecked():
            frame = self.det_detector.predict(
                frame, show_labels=self.det_label_check.isChecked()
            )
        return frame

    def open_selected(self, row):
        if 0 <= row < len(self.videos):
            name, path = self.videos[row]
            self.current_video = path
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
