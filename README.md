# Football Analysis with Computer Vision

End-to-end football match analytics pipeline built with Python and deep learning.

This project detects and tracks players, referees, and the ball from broadcast video, then derives higher-level match insights such as:
- team-wise ball possession
- per-player speed (km/h)
- distance covered (meters)

It combines object detection, multi-object tracking, team color clustering, camera motion compensation, and perspective transformation in one production-style pipeline.

![Project Output](output_videos/screenshot.png)

## Project Highlights
This repository demonstrates practical ML engineering for sports video analytics:
- solving a real-world problem end-to-end
- modular pipeline design across multiple components
- feature engineering from raw detections to tactical metrics
- handling noisy detections with interpolation and tracking
- generating clear visual outputs for easier analysis

## Core Features
- `YOLO`-based detection for `player`, `referee`, and `ball`
- `ByteTrack` multi-object tracking for identity persistence
- team assignment via jersey color clustering (`KMeans`)
- ball possession estimation by nearest-player assignment
- camera motion estimation using Lucas-Kanade optical flow
- perspective transformation from pixel space to field coordinates
- speed and cumulative distance estimation per tracked player
- annotated output video rendering with overlays

## Pipeline Overview
1. Read input video frames.
2. Detect and track objects (players/referees/ball).
3. Add object positions (foot point for players, center for ball).
4. Estimate camera motion and compensate object positions.
5. Transform compensated positions to top-view field coordinates.
6. Interpolate missing ball positions.
7. Estimate speed and total distance for each tracked player.
8. Cluster player jersey colors and assign team IDs.
9. Assign ball carrier frame-by-frame and aggregate possession.
10. Render all analytics to output video.

## Tech Stack
- Python
- Ultralytics YOLO
- Supervision (ByteTrack)
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- FilterPy

## Project Structure
```text
Football_Analysis/
|- main.py
|- requirements.txt
|- trackers/
|- team_assigner/
|- player_ball_assigner/
|- camera_movement_estimator/
|- view_transformer/
|- speed_and_distance_estimator/
|- utils/
|- stubs/
|- development_and_analysis/
|- output_videos/
```

## Setup
### 1) Clone and install dependencies
```bash
git clone <your-repo-url>
cd Football_Analysis
pip install -r requirements.txt
```

### 2) Add model weights and input video
By default, `main.py` expects:
- model: `models/best.pt`
- input: `test/test (19).mp4`

If these are not in your repo, place them at those paths or update the paths in `main.py`.

Reference links:
- YOLO weights: [Google Drive](https://drive.google.com/file/d/1DC2kCygbBWUKheQ_9cFziCsYVSRw6axK/view?usp=sharing)
- Sample video: [Google Drive](https://drive.google.com/file/d/1t6agoqggZKx6thamUuPAIdN_1zR9v9S_/view?usp=sharing)

### 3) Run
```bash
python main.py
```

Output video is saved to:
- `output_videos/output_video.avi`

## Notes on Reproducibility
Current `main.py` runs with cached stub files enabled:
- `stubs/track_stubs.pkl`
- `stubs/camera_movement_stub.pkl`

For a fresh end-to-end run, set `read_from_stub=False` in `main.py` for:
- `tracker.get_object_tracks(...)`
- `camera_movement_estimator.get_camera_movement(...)`

## Implementation Highlights
- Robustness: missing ball detections are interpolated using Pandas interpolation + forward/backward fill.
- Efficiency: detections run in frame batches (`batch_size=20`).
- Practical CV design: camera compensation + homography is used before motion analytics.
- Explainability: output overlays show object IDs, possession percentages, camera shift, speed, and distance.

## Current Limitations
- input/output paths are hardcoded in `main.py`
- perspective transform points are fixed for one camera view
- team-color clustering initializes from the first frame only
- possession assignment uses a simple nearest-distance heuristic
- frame rate is currently fixed at `24 FPS` for speed estimation
