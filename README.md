# Conveyor Sentinel AI: Autonomous Industrial Conveyor Belt Inspection System

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://docs.ultralytics.com/)
[![Computer Vision](https://img.shields.io/badge/OpenCV-Computer%20Vision-red.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end industrial computer vision and deep learning monitoring system for real-time defect detection on conveyor belts. Conveyor Sentinel identifies punctures, holes, longitudinal tears, cracks, and foreign objects while reporting live operational telemetry and safety metrics through an interactive web dashboard.

---

## Key Features

- **Multi-Defect AI Detection**:
  - **Holes & Punctures** (Through-belt aperture detection with 5mm–500mm estimation)
  - **Cracks & Longitudinal Tears** (Fissure tracing and splice rip warnings)
  - **Foreign Objects** (Debris, rocks, metal scrap, tools)
- **Hybrid AI Engine**:
  - **Primary**: Lightweight, high-speed YOLOv8 object detector fine-tuned for industrial belt surfaces.
  - **Fallback**: Real-time adaptive OpenCV optical contour and luminance analyzer for low-compute or offline scenarios.
- **Real-Time Sentinel Dashboard**:
  - Live video stream HUD with dynamic defect reticles and bounding boxes.
  - **5-Point Safety Health Checklist** (Operating Tension, Belt Thickness, Surface Cracks, Splice Breaks, Holes).
  - Dual HUD gauges (Thickness Scale & Tension Scale).
  - Diagnostic Snapshot tool (one-click PNG export with timestamped diagnostic overlays).
  - Continuous simulation mode + Live webcam mode + Live YOLO AI mode.
- **Model Training Pipelines**:
  - Built-in synthetic conveyor generator for balanced dataset creation (300 labeled samples).
  - User image pipeline with automated augmentation (flips, contrast, luminance shifts, blur) for rapid transfer learning.

---

## System Architecture

```text
[ Industrial Belt / Camera Feed ]
               │
               ▼
   [ ConveyorDetector (YOLOv8 + OpenCV Fallback) ]
               │
               ├─────────────────────────┐
               ▼                         ▼
   [ Live MJPEG Stream (/stream) ]    [ Real-Time JSON Telemetry (/telemetry) ]
               │                         │
               └───────────┬─────────────┘
                           ▼
          [ Conveyor Sentinel Web Dashboard ]
        (5-Point Checklist, HUD, Telemetry & Logs)
```

---

## Installation

### 1. Prerequisites
- Python 3.10 or 3.11
- Webcam (or built-in camera)

### 2. Clone the Repository
```bash
git clone https://github.com/DhigveerrajuG/mining-conveyor-belt-monitoring-system.git
cd mining-conveyor-belt-monitoring-system
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Quick Start

### Launch the Complete Dashboard & AI Server
Run the batch file:
```bat
run_live_sentinel_ai.bat
```
Or start via command line:
```bash
python yolo_sentinel_server.py
```
Open your browser at: **`http://localhost:8000/`**

### Standalone Webcam Detection
To inspect defects directly through an OpenCV window:
```bat
run_webcam_detection.bat
```

---

## Model Training

### Option A: Retrain with Synthetic Data
Generates 300 synthetic damaged belt images with labels, runs 35 epochs of YOLOv8 transfer learning, and exports weights:
```bat
train_new_model.bat
```
Or run:
```bash
python train_perfect_conveyor_model.py
```

### Option B: Train with Custom Photos
1. Place your photos into the appropriate folders under `user_images/`:
   - `user_images/holes/`
   - `user_images/foreign_objects/`
   - `user_images/clean/`
2. Run the training script:
```bat
train_with_my_images.bat
```
Or run:
```bash
python train_from_user_images.py
```
The trained weights will automatically export to `weights/best.pt`.

---

## HTTP API Endpoints

| Endpoint | Method | Content-Type | Description |
| :--- | :---: | :--- | :--- |
| `/` | `GET` | `text/html` | Serves the interactive Sentinel dashboard |
| `/telemetry` | `GET` | `application/json` | Real-time belt health score, defect list, FPS, and sensor metrics |
| `/stream` | `GET` | `multipart/x-mixed-replace` | Real-time processed video stream with HUD overlays |

---

## Repository Structure

```text
├── weights/
│   └── best.pt                     # Pre-trained YOLOv8 defect detection weights
├── user_images/                    # Custom dataset directory
│   ├── holes/
│   ├── foreign_objects/
│   └── clean/
├── conveyor-sentinel-dashboard.html # Single-page Sentinel monitoring dashboard
├── yolo_sentinel_server.py         # Multi-threaded HTTP & MJPEG AI server
├── train_perfect_conveyor_model.py # Synthetic data generator & YOLO trainer
├── train_from_user_images.py       # Custom image augmentor & trainer
├── create_user_labels.py           # Ground-truth label builder
├── verify_annotations.py           # Annotation visual verification overlay tool
├── run_live_sentinel_ai.bat        # 1-click launcher for server & dashboard
├── run_webcam_detection.bat        # 1-click launcher for webcam YOLO detection
├── train_new_model.bat             # 1-click launcher for synthetic retraining
├── train_with_my_images.bat        # 1-click launcher for custom training
├── requirements.txt                # Python package requirements
├── .gitignore                      # Git ignore rules
└── LICENSE                         # MIT License
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
