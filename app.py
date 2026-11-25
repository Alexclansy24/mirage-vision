
import time

import cv2

from src.camera import Camera
from src.detector import DetectionWorker
from src.renderer import draw_detections
from src.tracker import ObjectTracker

def main():
    camera = Camera(camera_index=0)
    detector = DetectionWorker()
    tracker = ObjectTracker(
    max_missed=3,
    iou_threshold=0.25,
    )

    tracked_detections = []
    last_detection_id = -1

    fps = 0.0
    previous_time = time.perf_counter()

    try:
        camera.start()
        detector.start()

        print("Mirage Vision started")
        print("Press Q to exit")

        while True:
            frame = camera.read()

            if frame is None:
                print("Could not read camera frame")
                break

            current_detection_id = detector.detection_id

            if current_detection_id != last_detection_id:
                tracked_detections = tracker.update(
                    detector.detections
                )

                last_detection_id = current_detection_id

            detector.update_frame(frame)

            detections, inference_ms, status, error = (
                detector.get_results()
            )

            # Draw the latest available detections.
            frame = draw_detections(
                frame,
                tracked_detections,
            )

            # Calculate camera FPS.
            current_time = time.perf_counter()
            delta = current_time - previous_time
            previous_time = current_time

            if delta > 0:
                current_fps = 1.0 / delta
                fps = (
                    current_fps
                    if fps == 0
                    else fps * 0.9 + current_fps * 0.1
                )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Inference: {inference_ms:.0f} ms",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Status: {status}",
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )

            if error:
                print(f"Detection error: {error}")

            cv2.imshow(
                "Mirage Vision - Real-Time Detection",
                frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except RuntimeError as error:
        print(f"Application error: {error}")

    finally:
        detector.stop()
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
