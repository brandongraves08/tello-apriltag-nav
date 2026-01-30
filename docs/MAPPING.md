# Mapping (Foundations)

This repo currently supports two complementary "mapping" tracks:

1) **AprilTag map (cheap & reliable indoors)**
   - You place a few printed AprilTags around the house.
   - We estimate the drone pose relative to a tag when seen.
   - We can build a simple map of tag poses + the drone trajectory *in tag-map coordinates*.

2) **Video-based mapping (offline)**
   - Record a flight video.
   - Run an offline tool (COLMAP / ORB-SLAM) on a PC to reconstruct a trajectory / 3D model.

This file documents what’s implemented today and the next steps.

---

## AprilTag Map (what we implement first)

### Why
- Tello has no lidar.
- Pure monocular SLAM indoors is fragile.
- AprilTags give you hard landmarks that are cheap and work.

### Requirement: camera intrinsics
To get **metric** pose (meters), AprilTag pose estimation needs approximate camera intrinsics.

We provide an **OpenCV calibration script** to estimate intrinsics using a printed checkerboard.

---

## Workflow

### 1) Calibrate camera (once)
Run:
```bash
python tools/camera_calibrate.py --source tello --rows 6 --cols 9 --square-size-mm 25
```
This saves `calibration.json` containing fx, fy, cx, cy.

### 2) Build/extend a tag map
Run:
```bash
python -m src.tag_mapper --tag-size-m 0.12 --calibration calibration.json --out tagmap.json
```
This detects tags and stores observations. (Early version: writes detections and pose estimates. Later: optimizes a global map.)

### 3) Navigate using the tag map (future)
- Use `tagmap.json` to plan routes between known tags.
- Add a simple controller to go: tag A → tag B.

---

## Offline mapping (COLMAP)
If you want a 3D model of the house, do:
1) `python -m src.tello_record --out recordings/flight1` to record frames.
2) Use COLMAP to reconstruct.

We’ll add a helper script that extracts frames and writes a COLMAP-friendly folder layout.

---

## Roadmap
- [ ] Record frames + timestamps + battery + (optional) tag detections
- [ ] Camera calibration tool
- [ ] AprilTag pose estimation with calibrated intrinsics
- [ ] Simple tag map JSON schema
- [ ] Basic graph optimization (tag poses + drone poses)
- [ ] Route navigation between tags
