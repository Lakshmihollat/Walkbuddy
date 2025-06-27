# WalkBuddy -An AI Companion for Visually Aware Navigation

**WalkBuddy** is an intelligent object detection and tracking assistant designed to aid pedestrian awareness in real-time or simulated walking environments. Originally built using **MobileNet SSD**, it evolved to adopt **YOLOv8 + DeepSORT** for more accurate detection and multi-object tracking.

---

##  What It Does

-  Detects objects (like people, bicycles, cars, dogs) in a live feed or walking video.
-  Uses Text-to-Speech to alert users of potential objects ahead.
-  Tracks each detected object using unique IDs to prevent repetitive alerts.
-  Filters out irrelevant or noisy labels like "sofa", "suitcase", "tvmonitor", etc.
-  Supports webcam or recorded videos as input sources.

---

## Evolution

### Phase 1 – MobileNet SSD:
- Used OpenCV’s `dnn` module with MobileNet SSD.
- Quick and lightweight but prone to false positives (e.g., detecting a wall as a sofa).

### Phase 2 – YOLOv8 + DeepSORT:
- Adopted **YOLOv8n** (via `ultralytics`) for higher accuracy.
- Integrated **DeepSORT** for real-time multi-object tracking.
- Significantly reduced repetitive audio alerts and improved overall robustness.

