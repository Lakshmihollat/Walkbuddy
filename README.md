#  WalkBuddy – An AI Companion for Visually Aware Navigation

**WalkBuddy** is an assistive AI system designed to help **visually impaired individuals** navigate environments more safely by detecting and tracking nearby objects in real time. It leverages state-of-the-art object detection and tracking to provide spoken alerts about potential obstacles or people in the path.

---

## What It Does

-  Detects real-world objects such as **people, bicycles, vehicles, animals** from live camera or video.
-  Uses **Text-to-Speech (TTS)** to audibly alert the user about nearby objects.
-  Assigns unique IDs to each detected object using **DeepSORT**, preventing repeated alerts.
-  Filters out irrelevant or misclassified labels (e.g., `"sofa"`, `"suitcase"`, `"tvmonitor"`).
-  Supports both **webcam input** and **pre-recorded walking videos** for testing or demonstration.

---

##  Evolution of the Project

###  Phase 1 – MobileNet SSD (OpenCV dnn)
- Used OpenCV’s DNN module with MobileNet SSD.
- Pros: Lightweight and easy to set up.
- Cons: Prone to **false positives**, e.g., mislabeling walls as sofas.

###  Phase 2 – YOLOv8 + DeepSORT
- Switched to **YOLOv8n** via the `ultralytics` library for more robust detection.
- Integrated **DeepSORT** for real-time multi-object tracking.
- Result:Improved accuracy and reduced repetitive audio alerts.

---

## Tech Stack

- Python
- YOLOv8 (Ultralytics)
- DeepSORT
- OpenCV
- pyttsx3 (offline TTS engine)

---

##  Use Cases

- Navigation assistance for **visually impaired pedestrians**
- Smart surveillance or pedestrian alert systems
- Real-time safety companion for walkers or joggers

---

## How to Run

1. Clone the repo
2. Install dependencies  
   ```bash
   pip install -r requirements.txt
