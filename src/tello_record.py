"""Tello video/telemetry recorder.

Records:
- frames (JPEG)
- a metadata JSONL log with timestamps + battery + (optional) AprilTag detections

This is the foundation for offline mapping (COLMAP/SLAM) and for tuning the tag follower.

Usage:
  python -m src.tello_record --out recordings/flight1 --fps 15 --detect-tags

Keys:
  r start/stop recording
  q quit

Notes:
- Connect to Tello Wi‑Fi first.
- This script does *not* takeoff/land; it only records.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
from djitellopy import Tello
from pupil_apriltags import Detector


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Record Tello stream to frames + jsonl")
    ap.add_argument("--out", type=str, required=True, help="output directory")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--detect-tags", action="store_true")
    ap.add_argument("--family", type=str, default="tag36h11")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "log.jsonl"

    detector = None
    if args.detect_tags:
        detector = Detector(families=args.family, nthreads=2, quad_decimate=1.5)

    tello = Tello()
    print("Connecting to Tello...")
    tello.connect()
    try:
        print("Battery:", tello.get_battery(), "%")
    except Exception:
        pass

    tello.streamon()
    frame_read = tello.get_frame_read()

    recording = False
    i = 0
    dt = 1.0 / float(args.fps)
    next_t = time.time()

    print("Press 'r' to start/stop recording. 'q' to quit.")

    with log_path.open("a", encoding="utf-8") as f:
        try:
            while True:
                frame = frame_read.frame
                if frame is None:
                    time.sleep(0.01)
                    continue

                vis = frame
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    recording = not recording
                    print("Recording:", recording)

                now = time.time()
                if recording and now >= next_t:
                    next_t = now + dt

                    fname = f"{i:06d}.jpg"
                    fpath = frames_dir / fname
                    cv2.imwrite(str(fpath), frame)

                    entry = {
                        "t": now,
                        "frame": str(fpath.relative_to(out_dir)).replace("\\", "/"),
                    }
                    try:
                        entry["battery"] = int(tello.get_battery())
                    except Exception:
                        pass

                    if detector is not None:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        dets = detector.detect(gray)
                        entry["tags"] = [
                            {"id": int(d.tag_id), "center": [float(d.center[0]), float(d.center[1])], "area": float(d.area)}
                            for d in dets
                        ]

                    f.write(json.dumps(entry) + "\n")
                    f.flush()
                    i += 1

                cv2.putText(vis, f"rec={recording} frames={i}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("tello-record", vis)

        finally:
            cv2.destroyAllWindows()
            try:
                tello.streamoff()
            except Exception:
                pass
            try:
                tello.end()
            except Exception:
                pass


if __name__ == "__main__":
    main()
