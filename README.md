# Vortex ROV — MATE 2024 CV Pipeline

**3rd Place — MATE ROV Egypt Regional 2024**
Explorer Class | Computer Vision Pipeline

---

## Team

| Name | Role |
|------|------|
| Mohamed Ashraf | CV Lead, System Integration |
| Ahmed Khaled | Underwater Imaging, White Balance |
| Sara Hassan | Coral Health Analysis, Grid Classification |
| Omar Youssef | Object Detection, Box Localization |
| Nour El-Din | Photomosaic, Feature Matching |
| Rana Mahmoud | Telemetry, Dashboard, Logging |
| Yusuf Tarek | Navigation, Guidance, POI Mapping |
| Layla Ibrahim | Calibration, Distortion Correction |

---

## Competition Tasks

The MATE ROV 2024 competition challenges teams to operate an ROV in a simulated
underwater environment modeled after real ocean missions. The CV pipeline supports
four primary tasks:

### Task 1 — Transect Line Following
The ROV autonomously follows a colored guide rope laid across the pool floor.
The system detects the rope via HSV segmentation, computes the lateral offset
and angular heading error using Hough line detection, and runs a PID controller
to issue real-time steering corrections.

### Task 2 — Point of Interest Mapping
Colored markers are placed at various locations on the pool floor and walls.
The pipeline detects markers, maps their pixel positions to spatial coordinates
using camera intrinsics, and merges duplicate sightings to build a confirmed
POI map. Output is exported as JSON and visualized as a 2D overhead plot.

### Task 3 — Coral Health Assessment
Before-and-after images of a simulated coral reef grid are compared to classify
changes in each cell. Each 4x4 grid cell is analyzed using HSV color segmentation
(healthy green, bleached white, dead brown/grey), histogram correlation, and SSIM
structural comparison. The system generates a severity-scored health report and
annotated output image.

### Task 4 — Photomosaic Construction
Sequential overlapping images of the pool floor are stitched into a composite
photomosaic. ORB keypoints are matched with a ratio test, RANSAC homography is
computed, and frames are warped and blended with feathering. A grid-tiling
fallback is used when texture is insufficient for feature matching.

---

## System Architecture

```
Camera Input (V4L2 / USB)
        |
        v
White Balance + CLAHE Preprocessing
        |
        v
   [Parallel Pipelines]
   /       |        \         \
Transect  Marker   Box        YOLO
Follower  Detect   Detect     Inference
   |       |        \         /
   v       v         v       v
  PID    POI Map   Kalman  Detections
Heading  Spatial   Track
Control  Mapping      |
   |       |          v
   +-------+----> Guidance Layer
                       |
                       v
              Telemetry Dashboard + CSV Logger
```

### Key Modules

- `src/camera/` — V4L2 capture, checkerboard calibration, underwater white balance
- `src/detection/` — HSV color segmentation, box detection, marker detection, YOLOv8
- `src/tracking/` — Centroid tracker (Hungarian matching), Kalman filter wrapper
- `src/analysis/` — Coral health grid, transect PID, photomosaic stitching
- `src/navigation/` — Directional guidance, monocular distance, POI coordinate mapping
- `src/telemetry/` — tkinter live dashboard, CSV telemetry logger

---

## Setup

### Prerequisites

- Python 3.9+
- USB camera or V4L2-compatible device
- (Optional) NVIDIA GPU for YOLOv8 inference

### Install

```bash
git clone https://github.com/vortex-robotics/rov-2024.git
cd rov-2024
pip install -e ".[dev]"
```

### Docker

```bash
docker build -t vortex-rov:2024 .
docker run --device=/dev/video0 vortex-rov:2024 python scripts/run_competition.py --task transect
```

---

## Usage

### Camera Calibration

Run calibration with a printed checkerboard (9x6 inner corners):

```bash
python scripts/run_calibration.py --camera 0 --rows 9 --cols 6 --output config/calibration.yaml
```

### Competition Tasks

```bash
# Transect following (live camera)
python scripts/run_competition.py --task transect --camera 0

# POI mapping (live camera)
python scripts/run_competition.py --task poi --camera 0

# Box detection and guidance
python scripts/run_competition.py --task boxes --camera 0

# All tasks simultaneously (dashboard mode)
python scripts/run_competition.py --task all --camera 0
```

### Coral Health Analysis

```bash
python scripts/run_coral_analysis.py --before data/coral_before.jpg --after data/coral_after.jpg --grid 4 --output results/
```

### Photomosaic

```bash
# From image files
python scripts/run_mosaic.py --images data/frames/*.jpg --output results/

# Live capture
python scripts/run_mosaic.py --camera 0 --num 8 --output results/
```

---

## Configuration

All tuneable parameters are in `config/`:

- `camera.yaml` — resolution, FPS, camera index, intrinsics path
- `detection.yaml` — HSV ranges for all colors, morphology kernel sizes, area thresholds
- `competition.yaml` — PID gains, distance references, grid size, feature matching params

Edit these files to adapt the pipeline to different lighting and pool conditions
without touching source code.

---

## Model Weights

YOLOv8 fine-tuned weights are stored in `models/`. Download from the team's
shared drive or retrain with:

```bash
yolo train model=yolov8n.pt data=data/dataset.yaml epochs=100 imgsz=640
```

Place the resulting `best.pt` at `models/rov_yolov8.pt`.

---

## Results — MATE ROV Egypt Regional 2024

| Task | Score | Notes |
|------|-------|-------|
| Transect Following | 47 / 50 | Missed 1 waypoint due to turbulence |
| POI Mapping | 38 / 40 | All 5 POIs confirmed |
| Coral Health | 44 / 50 | Grid offset on run 2 |
| Photomosaic | 36 / 40 | Minor blending artifact on frame 6 |
| **Total** | **165 / 180** | **3rd Place** |

---

## Repository Structure

```
vortex-rov-new/
├── config/               # YAML configuration files
├── src/                  # Production Python source
│   ├── camera/           # Capture, calibration, white balance
│   ├── detection/        # Color segmentation, box/marker/YOLO detectors
│   ├── tracking/         # Centroid and Kalman trackers
│   ├── analysis/         # Coral health, transect PID, photomosaic
│   ├── navigation/       # Guidance arrows, POI coordinate mapping
│   ├── telemetry/        # Dashboard, CSV logger
│   └── utils/            # Image ops, geometry helpers
├── scripts/              # CLI entry points
└── tests/                # Unit tests
```

---

## License

MIT License. See LICENSE for details.
