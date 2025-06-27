import cv2
import numpy as np
import pyttsx3
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize TTS
engine = pyttsx3.init()
engine.setProperty('rate', 160)

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # You can try 'yolov8s.pt' or 'yolov8m.pt' if needed

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)
IGNORED_LABELS = {"sofa", "suitcase", "tvmonitor", "remote","train"}
# Video source (webcam = 0 or replace with video path)
cap = cv2.VideoCapture('Walk_Video.mp4')

# Keep track of spoken object IDs
spoken_ids = set()
CONFIDENCE_THRESHOLD = 0.5
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]
    detections = []

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, cls_id = result
        if score < CONFIDENCE_THRESHOLD:
            continue  # Skip low-confidence detections
        label = model.names[int(cls_id)]
        detections.append(([x1, y1, x2 - x1, y2 - y1], score, label))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        label = track.get_det_class()
        if label in IGNORED_LABELS:
            continue  # skip irrelevant labels


        track_id = track.track_id
        l, t, w, h = track.to_ltrb()
        label = track.get_det_class()

        # Draw box
        cv2.rectangle(frame, (int(l), int(t)), (int(l + w), int(t + h)), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} #{track_id}", (int(l), int(t) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Speak only if new track_id
        if track_id not in spoken_ids:
            spoken_ids.add(track_id)
            engine.say(f"{label} ahead")
            engine.runAndWait()

    # Show frame
    cv2.imshow("WalkBuddy YOLOv8 + DeepSORT", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
