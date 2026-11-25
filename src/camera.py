
import cv2

from src.config import (
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
)


class Camera:
    def __init__(self, camera_index=CAMERA_INDEX):
        self.camera_index = camera_index
        self.capture = None

    def start(self):
        self.capture = cv2.VideoCapture(
            self.camera_index
        )

        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            raise RuntimeError(
                f"Could not open camera {self.camera_index}"
            )

        self.capture.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            CAMERA_WIDTH,
        )

        self.capture.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            CAMERA_HEIGHT,
        )

        return self

    def read(self):
        if self.capture is None:
            raise RuntimeError("Camera is not started")

        success, frame = self.capture.read()

        if not success:
            return None

        return frame

    def stop(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None
