"""Camera calibration helper (Windows-friendly).

Goal: estimate camera intrinsics (fx, fy, cx, cy) so we can do metric AprilTag pose.

This supports two sources:
- --source webcam (default OpenCV capture index 0)
- --source tello  (connects to Tello, uses its stream)

Usage examples:
  python tools/camera_calibrate.py --source webcam --rows 6 --cols 9 --square-size-mm 25
  python tools/camera_calibrate.py --source tello  --rows 6 --cols 9 --square-size-mm 25

Notes:
- rows/cols are *inner corners*.
- Use a printed checkerboard and hold it at different angles/distances.
- Press 'c' to capture a sample when corners are detected.
- Press 'q' to calibrate and save calibration.json.

This is intentionally minimal (no distortion model fancy stuff).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["webcam", "tello"], default="webcam")
    ap.add_argument("--camera", type=int, default=0, help="webcam index when --source webcam")
    ap.add_argument("--rows", type=int, required=True, help="checkerboard inner corners rows")
    ap.add_argument("--cols", type=int, required=True, help="checkerboard inner corners cols")
    ap.add_argument("--square-size-mm", type=float, required=True, help="checker square size in mm")
    ap.add_argument("--out", type=str, default="calibration.json")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    pattern_size = (args.cols, args.rows)
    square = float(args.square_size_mm) / 1000.0

    # Prepare object points (0,0,0), (1,0,0), ... scaled by square size.
    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0 : args.cols, 0 : args.rows].T.reshape(-1, 2)
    objp *= square

    objpoints = []
    imgpoints = []

    tello = None
    frame_read = None

    if args.source == "tello":
        from djitellopy import Tello

        tello = Tello()
        print("Connecting to Tello...")
        tello.connect()
        try:
            print("Battery:", tello.get_battery(), "%")
        except Exception:
            pass
        tello.streamon()
        frame_read = tello.get_frame_read()

    else:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise SystemExit(f"Failed to open webcam index {args.camera}")

    print("Press 'c' to capture when corners look good. Press 'q' to calibrate+save.")

    try:
        while True:
            if args.source == "tello":
                frame = frame_read.frame
                if frame is None:
                    time.sleep(0.01)
                    continue
            else:
                ok, frame = cap.read()
                if not ok:
                    continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, pattern_size)

            vis = frame.copy()
            if found:
                corners2 = cv2.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
                )
                cv2.drawChessboardCorners(vis, pattern_size, corners2, found)
                cv2.putText(vis, "corners: found", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(vis, "corners: none", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(vis, f"samples={len(objpoints)}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("calibrate", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c") and found:
                objpoints.append(objp)
                imgpoints.append(corners2)
                print(f"Captured sample {len(objpoints)}")

        if len(objpoints) < 8:
            raise SystemExit("Need at least ~8 good samples. Capture more angles/distances.")

        h, w = gray.shape[:2]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

        fx = float(mtx[0, 0])
        fy = float(mtx[1, 1])
        cx = float(mtx[0, 2])
        cy = float(mtx[1, 2])

        out = {
            "image_size": {"width": int(w), "height": int(h)},
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "dist": [float(x) for x in dist.reshape(-1).tolist()],
            "reprojection_error": float(ret),
        }

        Path(args.out).write_text(json.dumps(out, indent=2))
        print("Wrote", args.out)
        print(out)

    finally:
        cv2.destroyAllWindows()
        try:
            if args.source == "tello" and tello is not None:
                tello.streamoff()
                tello.end()
        except Exception:
            pass
        try:
            if args.source == "webcam":
                cap.release()
        except Exception:
            pass


if __name__ == "__main__":
    main()
