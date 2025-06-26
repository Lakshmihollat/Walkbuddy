import cv2
import os
import numpy as np

# Get path to the models folder
model_path = os.path.join(os.path.dirname(__file__), 'models')

# Load the pre-trained MobileNet SSD model
net = cv2.dnn.readNetFromCaffe(
    os.path.join(model_path, 'MobileNetSSD_deploy.prototxt'),
    os.path.join(model_path, 'MobileNetSSD_deploy.caffemodel')
)

# Classes the model can detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Make window fullscreen
    cv2.namedWindow('WalkBuddy - Camera Feed', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('WalkBuddy - Camera Feed', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Prepare image for detection
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Loop through all detected objects
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]

            # Calculate box coordinates
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0],
                                                       frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw box and label
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show updated frame
    cv2.imshow('WalkBuddy - Camera Feed', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
