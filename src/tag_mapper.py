"""AprilTag mapping (early foundation).

This script connects to a Tello, detects tags, and (optionally) estimates tag pose
in meters using camera calibration.

Output JSON is currently an *observation log* (not yet a globally optimized map).
That's still useful: you can review detections, tag sizes, timestamps, etc.

Usage:
  python -m src.tag_mapper --tag-size-m 0.12 --calibration calibration.json --out tag_observations.json

Keys:
  q quit
  s save snapshot immediately

Planned next:
- build a global tag map (graph optimization)
- output trajectory in map frame
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from djitellopy import Tello
from pupil_apriltags import Detector


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Tello AprilTag mapper (observations)")
    ap.add_argument("--tag-size-m", type=float, required=True, help="Printed tag size in meters (e.g. 0.12)")
    ap.add_argument("--calibration", type=str, default=None, help="calibration.json from tools/camera_calibrate.py")
    ap.add_argument("--out", type=str, default="tag_observations.json")
    ap.add_argument("--family", type=str, default="tag36h11")
    ap.add_argument("--limit", type=int, default=2000, help="max observations before auto-save+exit")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    cam_params = None
    if args.calibration:
        cal = json.loads(Path(args.calibration).read_text())
        cam_params = (float(cal["fx"]), float(cal["fy"]), float(cal["cx"]), float(cal["cy"]))
        print("Loaded calibration:", cam_params)
    else:
        print("No calibration loaded: will log 2D detections only (no metric pose)")

    detector = Detector(
        families=args.family,
        nthreads=2,
        quad_decimate=1.5,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )

    tello = Tello()
    print("Connecting to Tello...")
    tello.connect()
    try:
        print("Battery:", tello.get_battery(), "%")
    except Exception:
        pass

    tello.streamon()
    frame_read = tello.get_frame_read()

    obs = []
    started = time.time()

    def save() -> None:
        out = {
            "schema": "tello-apriltag-nav/observations-v1",
            "created_at": time.time(),
            "duration_s": time.time() - started,
            "tag_family": args.family,
            "tag_size_m": args.tag_size_m,
            "camera_params": None if cam_params is None else {"fx": cam_params[0], "fy": cam_params[1], "cx": cam_params[2], "cy": cam_params[3]},
            "observations": obs,
        }
        Path(args.out).write_text(json.dumps(out, indent=2))
        print("Saved", args.out, "(n=", len(obs), ")")

    try:
        while True:
            frame = frame_read.frame
            if frame is None:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if cam_params is None:
                dets = detector.detect(gray)
            else:
                dets = detector.detect(gray, estimate_tag_pose=True, camera_params=cam_params, tag_size=args.tag_size_m)

            vis = frame.copy()
            for d in dets:
                corners = np.int32(d.corners)
                cv2.polylines(vis, [corners], True, (0, 255, 0), 2)
                tcx, tcy = d.center
                cv2.circle(vis, (int(tcx), int(tcy)), 4, (0, 255, 0), -1)

                entry = {
                    "t": time.time(),
                    "tag_id": int(d.tag_id),
                    "center": [float(tcx), float(tcy)],
                    "area": float(d.area),
                    "corners": [[float(x), float(y)] for x, y in d.corners],
                }

                if cam_params is not None and getattr(d, "pose_t", None) is not None:
                    # pose_t is tag translation in camera frame (meters)
                    tvec = np.array(d.pose_t).reshape(3)
                    # pose_R is 3x3 rotation matrix
                    R = np.array(d.pose_R).reshape(3, 3)
                    entry["pose_t_m"] = [float(x) for x in tvec.tolist()]
                    entry["pose_R"] = [[float(x) for x in row.tolist()] for row in R]

                    cv2.putText(
                        vis,
                        f"id={d.tag_id} z={tvec[2]:.2f}m",
                        (int(tcx) + 10, int(tcy)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                else:
                    cv2.putText(
                        vis,
                        f"id={d.tag_id}",
                        (int(tcx) + 10, int(tcy)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                obs.append(entry)

            cv2.putText(vis, f"obs={len(obs)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("tag-mapper", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                save()

            if len(obs) >= args.limit:
                print("Observation limit reached; saving and exiting")
                break

        save()

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
