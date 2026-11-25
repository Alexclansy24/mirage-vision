
import cv2


def draw_detections(frame, detections):
    height, width = frame.shape[:2]

    for obj in detections:
        name = str(obj.get("name", "object"))
        bbox = obj.get("bbox")
        score = obj.get("score")

        if not isinstance(bbox, (list, tuple)):
            continue

        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox

        # Convert normalized coordinates to pixels.
        x1 = int(x1 * width / 1000)
        y1 = int(y1 * height / 1000)
        x2 = int(x2 * width / 1000)
        y2 = int(y2 * height / 1000)

        # Keep coordinates inside the frame.
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        
        label = name.upper()

        track_id = obj.get("track_id")

        if track_id is not None:
            label += f" #{track_id}"

        if isinstance(score, (int, float)):
            label += f" {score:.2f}"

        # Draw bounding box.
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        # Draw label background.
        (text_width, text_height), baseline = (
            cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                2,
            )
        )

        label_y = max(y1, text_height + 10)

        cv2.rectangle(
            frame,
            (x1, label_y - text_height - 8),
            (x1 + text_width + 8, label_y + baseline),
            (0, 255, 0),
            cv2.FILLED,
        )

        cv2.putText(
            frame,
            label,
            (x1 + 4, label_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return frame
