
class ObjectTracker:
    def __init__(self, max_missed=3, iou_threshold=0.25):
        self.max_missed = max_missed
        self.iou_threshold = iou_threshold

        self.next_id = 1
        self.tracks = {}

    @staticmethod
    def _iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        intersection_x1 = max(ax1, bx1)
        intersection_y1 = max(ay1, by1)
        intersection_x2 = min(ax2, bx2)
        intersection_y2 = min(ay2, by2)

        intersection_width = max(
            0, intersection_x2 - intersection_x1
        )
        intersection_height = max(
            0, intersection_y2 - intersection_y1
        )

        intersection = intersection_width * intersection_height

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

        union = area_a + area_b - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    def update(self, detections):
        detections = [dict(item) for item in detections]

        # Age existing tracks by one detection update.
        for track in self.tracks.values():
            track["missed"] += 1

        candidates = []

        # Match detections with existing tracks.
        for detection_index, detection in enumerate(detections):
            bbox = detection.get("bbox")
            label = detection.get("name", "object")

            if not bbox or len(bbox) != 4:
                continue

            for track_id, track in self.tracks.items():
                if track["name"] != label:
                    continue

                overlap = self._iou(bbox, track["bbox"])

                if overlap >= self.iou_threshold:
                    candidates.append(
                        (overlap, detection_index, track_id)
                    )

        # Highest-overlap matches are assigned first.
        candidates.sort(reverse=True)

        matched_detections = set()
        matched_tracks = set()

        for _, detection_index, track_id in candidates:
            if detection_index in matched_detections:
                continue

            if track_id in matched_tracks:
                continue

            detection = detections[detection_index]

            self.tracks[track_id]["bbox"] = detection["bbox"]
            self.tracks[track_id]["missed"] = 0

            detection["track_id"] = track_id

            matched_detections.add(detection_index)
            matched_tracks.add(track_id)

        # Create tracks for unmatched detections.
        for index, detection in enumerate(detections):
            if index in matched_detections:
                continue

            bbox = detection.get("bbox")

            if not bbox or len(bbox) != 4:
                continue

            track_id = self.next_id
            self.next_id += 1

            self.tracks[track_id] = {
                "bbox": bbox,
                "name": detection.get("name", "object"),
                "missed": 0,
            }

            detection["track_id"] = track_id

        # Remove tracks that have been missing too long.
        expired_tracks = [
            track_id
            for track_id, track in self.tracks.items()
            if track["missed"] > self.max_missed
        ]

        for track_id in expired_tracks:
            del self.tracks[track_id]

        return detections
