
import threading
import time

import cv2
import cactus

from src.config import (
    INFERENCE_WIDTH,
    CONFIDENCE_THRESHOLD,
    MAX_OBJECTS,
    INFERENCE_INTERVAL,
)


class DetectionWorker:
    def __init__(self):
        self.detection_id = 0
        self.latest_frame = None
        self.frame_id = 0
        self.processed_frame_id = -1

        self.detections = []
        self.inference_ms = 0.0
        self.status = "Loading Mirage..."
        self.error = None

        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
        )

    def start(self):
        self.thread.start()

    def update_frame(self, frame):
        with self.lock:
            self.latest_frame = frame.copy()
            self.frame_id += 1
            self.detection_id += 1

    def get_results(self):
        with self.lock:
            return (
                self.detections.copy(),
                self.inference_ms,
                self.status,
                self.error,
            )

    def _run(self):
        try:
            vision = cactus.Mirage()

            with self.lock:
                self.status = "Model ready"

            while not self.stop_event.is_set():
                with self.lock:
                    if (
                        self.latest_frame is None
                        or self.frame_id == self.processed_frame_id
                    ):
                        frame = None
                        current_frame_id = -1
                    else:
                        frame = self.latest_frame.copy()
                        current_frame_id = self.frame_id

                if frame is None:
                    time.sleep(0.01)
                    continue

                # Mark this frame as processed.
                with self.lock:
                    self.processed_frame_id = current_frame_id

                # Resize while preserving aspect ratio.
                height, width = frame.shape[:2]

                if width > INFERENCE_WIDTH:
                    new_height = int(
                        height * INFERENCE_WIDTH / width
                    )

                    frame = cv2.resize(
                        frame,
                        (INFERENCE_WIDTH, new_height),
                        interpolation=cv2.INTER_AREA,
                    )

                # Convert BGR to RGB.
                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                start_time = time.perf_counter()

                result = vision.describe(
                    rgb_frame,
                    sections=["objects"],
                    fields=["bbox", "score"],
                    conf=CONFIDENCE_THRESHOLD,
                    max_objects=MAX_OBJECTS,
                )

                inference_ms = (
                    time.perf_counter() - start_time
                ) * 1000

                objects = result.get("objects", [])

                with self.lock:
                    self.detections = objects
                    self.inference_ms = inference_ms
                    self.status = "Detecting"
                    self.error = None

                time.sleep(INFERENCE_INTERVAL)

        except Exception as error:
            with self.lock:
                self.status = "Detection error"
                self.error = str(error)

    def stop(self):
        self.stop_event.set()

        if self.thread.is_alive():
            self.thread.join(timeout=3.0)
