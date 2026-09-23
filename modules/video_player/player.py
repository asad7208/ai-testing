"""Video preview widget: show a video with play/pause, seek slider and go-to-frame."""

import cv2
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class VideoPlayer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cap = None
        self.frame_count = 0
        self.fps = 25.0
        self.current_frame = 0
        self._current_pixmap = None
        self._raw_frame = None
        self.processed_frame = None
        self.processor = None  # optional callable: frame -> frame
        self.apply_processor = True

        self.display = QLabel("No video loaded")
        self.display.setAlignment(Qt.AlignCenter)
        self.display.setMinimumSize(480, 270)
        self.display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.display.setStyleSheet("background: #111; color: #888;")

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop)

        self.step_buttons = []
        for text, delta in (("<< 10", -10), ("< 1", -1), ("1 >", 1), ("10 >>", 10)):
            button = QPushButton(text)
            button.setFixedWidth(60)
            button.clicked.connect(lambda _, d=delta: self.step(d))
            self.step_buttons.append(button)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderMoved.connect(self.seek)
        self.slider.valueChanged.connect(self._slider_changed)

        self.frame_box = QSpinBox()
        self.frame_box.setEnabled(False)
        self.frame_box.setKeyboardTracking(False)
        self.frame_box.valueChanged.connect(self.seek)

        self.info_label = QLabel("0 / 0    00:00 / 00:00")

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)

        controls = QHBoxLayout()
        controls.addWidget(self.step_buttons[0])
        controls.addWidget(self.step_buttons[1])
        controls.addWidget(self.play_button)
        controls.addWidget(self.stop_button)
        controls.addWidget(self.step_buttons[2])
        controls.addWidget(self.step_buttons[3])
        controls.addWidget(self.slider, 1)
        controls.addWidget(QLabel("Frame:"))
        controls.addWidget(self.frame_box)

        layout = QVBoxLayout(self)
        layout.addWidget(self.display, 1)
        layout.addLayout(controls)
        layout.addWidget(self.info_label)

    # --- loading -----------------------------------------------------
    def show_video(self, path):
        """Open a video file and display its first frame."""
        self.stop()
        if self.cap is not None:
            self.cap.release()
        self.apply_processor = True
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.cap = None
            self.display.setText("Could not open video")
            return False

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 0 else 25.0

        last = max(self.frame_count - 1, 0)
        for widget in (self.slider, self.frame_box):
            widget.blockSignals(True)
            widget.setEnabled(True)
            widget.setRange(0, last)
            widget.setValue(0)
            widget.blockSignals(False)

        self.seek(0)
        return True

    def show_image(self, image):
        """Display a still image (annotation review); playback controls go idle."""
        self.pause()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.frame_count = 0
        self.current_frame = 0
        self.apply_processor = False
        for widget in (self.slider, self.frame_box):
            widget.setEnabled(False)
        self._raw_frame = image
        self._draw_frame()
        self.info_label.setText(f"still image  {image.shape[1]}x{image.shape[0]}")

    # --- playback ----------------------------------------------------
    def toggle_play(self):
        if self.cap is None:
            return
        if self.timer.isActive():
            self.pause()
        else:
            self.play()

    def play(self):
        if self.cap is None:
            return
        if self.current_frame >= self.frame_count - 1:
            self.seek(0)
        self.timer.start(int(1000 / self.fps))
        self.play_button.setText("Pause")

    def pause(self):
        self.timer.stop()
        self.play_button.setText("Play")

    def stop(self):
        self.pause()
        if self.cap is not None:
            self.seek(0)

    def next_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            self.pause()
            return
        self.current_frame += 1
        self._set_frame(frame)
        self._sync_controls()

    def step(self, delta):
        """Move delta frames from the current position (pauses playback)."""
        if self.cap is None:
            return
        self.pause()
        self.seek(self.current_frame + delta)

    def seek(self, frame_index):
        """Jump to an exact frame number."""
        if self.cap is None:
            return
        frame_index = max(0, min(int(frame_index), max(self.frame_count - 1, 0)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.cap.read()
        if not ok:
            return
        self.current_frame = frame_index
        self._set_frame(frame)
        self._sync_controls()

    def _slider_changed(self, value):
        # Covers clicks on the groove / arrow keys, not just dragging.
        if not self.slider.isSliderDown() and value != self.current_frame:
            self.seek(value)

    # --- display -----------------------------------------------------
    def _set_frame(self, frame):
        self._raw_frame = frame
        self._draw_frame()

    def refresh(self):
        """Re-render the current frame (e.g. after processing options change)."""
        if self._raw_frame is not None:
            self._draw_frame()

    def _draw_frame(self):
        frame = self._raw_frame
        if self.processor is not None and self.apply_processor:
            frame = self.processor(frame)
        self.processed_frame = frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self._current_pixmap = QPixmap.fromImage(image)
        self._repaint()

    def _repaint(self):
        if self._current_pixmap is None:
            return
        self.display.setPixmap(
            self._current_pixmap.scaled(
                self.display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._repaint()

    def _sync_controls(self):
        for widget in (self.slider, self.frame_box):
            widget.blockSignals(True)
            widget.setValue(self.current_frame)
            widget.blockSignals(False)
        self.info_label.setText(
            f"{self.current_frame} / {max(self.frame_count - 1, 0)}    "
            f"{_timestamp(self.current_frame / self.fps)} / "
            f"{_timestamp(self.frame_count / self.fps)}    {self.fps:.2f} fps"
        )

    def closeEvent(self, event):
        self.pause()
        if self.cap is not None:
            self.cap.release()
        super().closeEvent(event)


def _timestamp(seconds):
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"
