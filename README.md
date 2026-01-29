# tello-apriltag-nav

AprilTag-based “good enough” indoor navigation experiments for **DJI Tello EDU**, controlled from a **Windows laptop**.

Goal: get a fun, cheap autonomy loop working:
- connect to the Tello EDU over Wi‑Fi
- read its front camera stream
- detect AprilTags (tag36h11)
- do a simple control loop: yaw to center + forward/back to hold distance
- safety: hover/search behavior, emergency land

## Hardware
- DJI **Tello EDU** (front-facing camera)
- A Windows laptop (this is the compute)
- Printed AprilTags (recommend **tag36h11**, ~10–15cm squares)

## Setup (Windows)
1) Install Python 3.10+.
2) Connect your laptop to the Tello EDU Wi‑Fi.
3) Install deps:

```bash
pip install -r requirements.txt
```

## Run
```bash
python -m src.tello_apriltag_follow
```

### Controls
- `t` takeoff
- `l` land
- `q` quit (lands if flying)

## Notes / Caveats
- This is *not* obstacle-avoidance. Fly slow. Keep a hand on “land”.
- For v1 we use a **pixel-based distance proxy** (tag appears larger = closer). It avoids needing camera calibration.
- If you want metric pose (meters), we can add camera calibration and use the tag pose estimate.

## License
MIT
