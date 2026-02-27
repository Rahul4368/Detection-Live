import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import supervision as sv
from datetime import datetime

# ---------------------------
# CONFIG
# ---------------------------

MODEL_PATH = "yolov8n.pt"   # Replace with custom bag model if trained
CONFIDENCE = 0.4

# Define counting line (adjust per scenario)
LINE_START = (100, 400)
LINE_END = (1000, 400)

# ---------------------------
# INITIALIZE
# ---------------------------

model = YOLO(MODEL_PATH)
tracker = sv.ByteTrack()

box_annotator = sv.BoxAnnotator()
line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
line_annotator = sv.LineZoneAnnotator()

results_log = []

# ---------------------------
# FUNCTION TO PROCESS VIDEO
# ---------------------------

def process_video(video_path, output_path):

    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ---------------------------
        # YOLO Detection
        # ---------------------------

        result = model(frame, conf=CONFIDENCE)[0]
        detections = sv.Detections.from_ultralytics(result)

        # OPTIONAL: filter only bag class if custom model trained
        # detections = detections[detections.class_id == BAG_CLASS_ID]

        # ---------------------------
        # Tracking
        # ---------------------------

        detections = tracker.update_with_detections(detections)

        # ---------------------------
        # Line Counting
        # ---------------------------

        line_counter.trigger(detections)

        # ---------------------------
        # Annotate
        # ---------------------------

        labels = [
            f"ID {tracker_id}"
            for tracker_id in detections.tracker_id
        ]

        frame = box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )

        frame = line_annotator.annotate(
            frame=frame,
            line_counter=line_counter
        )

        # Add count text
        cv2.putText(
            frame,
            f"Total Count: {line_counter.in_count}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )

        out.write(frame)

    cap.release()
    out.release()

    # Save result
    results_log.append({
        "video": video_path,
        "count": line_counter.in_count,
        "timestamp": datetime.now()
    })


# ---------------------------
# RUN ALL SCENARIOS
# ---------------------------

if __name__ == "__main__":

    process_video(
        "Problem Statement Scenario1.mp4",
        "output_s1.mp4"
    )

    process_video(
        "Problem Statement Scenario2.mp4",
        "output_s2.mp4"
    )

    process_video(
        "Problem Statement Scenario3.mp4",
        "output_s3.mp4"
    )

    # Save CSV
    df = pd.DataFrame(results_log)
    df.to_csv("counts.csv", index=False)

    print("Processing Complete ✅")