"""Tello EDU + AprilTag follower

- Windows laptop connects to Tello Wi-Fi
- Reads video stream
- Detects AprilTags (tag36h11)
- Simple, conservative control:
  * yaw to center tag
  * forward/back based on apparent tag size (pixel-based distance proxy)

Keys:
  t = takeoff
  l = land
  q = quit (lands if flying)

This is intentionally minimal and safety-biased.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
from djitellopy import Tello
from pupil_apriltags import Detector


@dataclass
class Params:
    tag_family: str = "tag36h11"
    tag_id: int | None = None  # set to an int to follow only one tag

    # Control loop
    loop_hz: float = 20.0

    # Yaw control (pixels -> yaw rate)
    yaw_deadband_px: int = 25
    yaw_k: float = 0.20  # deg/s per pixel error (scaled + clamped)
    yaw_max: int = 40  # tello rc yaw is -100..100

    # Forward/back control using tag pixel area as proxy for distance
    area_target: float = 9000.0
    area_deadband: float = 1500.0
    fb_k: float = 0.0015
    fb_max: int = 25

    # Altitude hold using tag center (optional; off by default)
    use_ud: bool = False
    ud_deadband_px: int = 35
    ud_k: float = 0.18
    ud_max: int = 20

    # Search behavior when tag lost
    search_yaw: int = 15
    lost_timeout_s: float = 0.8

    # Safety
    max_rc: int = 35


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def main() -> None:
    p = Params()

    detector = Detector(
        families=p.tag_family,
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
    print("Battery:", tello.get_battery(), "%")

    tello.streamon()
    frame_read = tello.get_frame_read()

    flying = False
    last_seen = 0.0

    dt_target = 1.0 / p.loop_hz

    try:
        while True:
            t0 = time.time()
            frame = frame_read.frame
            if frame is None:
                time.sleep(0.01)
                continue

            # Tello frames are BGR already
            h, w = frame.shape[:2]
            cx, cy = w / 2.0, h / 2.0

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = detector.detect(gray)

            chosen = None
            if detections:
                if p.tag_id is None:
                    # pick largest tag by area
                    chosen = max(detections, key=lambda d: float(d.area))
                else:
                    for d in detections:
                        if int(d.tag_id) == int(p.tag_id):
                            chosen = d
                            break

            # Default commands: hover
            lr = 0
            fb = 0
            ud = 0
            yaw = 0

            if chosen is not None:
                last_seen = time.time()

                # Draw
                corners = np.int32(chosen.corners)
                cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
                tcx, tcy = chosen.center
                cv2.circle(frame, (int(tcx), int(tcy)), 4, (0, 255, 0), -1)
                cv2.putText(
                    frame,
                    f"tag {chosen.tag_id} area={chosen.area:.0f}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                # Yaw: center tag horizontally
                xerr = float(tcx - cx)
                if abs(xerr) > p.yaw_deadband_px:
                    yaw = int(clamp(p.yaw_k * xerr, -p.yaw_max, p.yaw_max))

                # Forward/back: hold target area
                aerr = float(p.area_target - float(chosen.area))
                if abs(aerr) > p.area_deadband:
                    fb = int(clamp(p.fb_k * aerr, -p.fb_max, p.fb_max))

                # Up/down (optional)
                if p.use_ud:
                    yerr = float(cy - tcy)
                    if abs(yerr) > p.ud_deadband_px:
                        ud = int(clamp(p.ud_k * yerr, -p.ud_max, p.ud_max))

            else:
                # Tag lost: search yaw after a short timeout
                if flying and (time.time() - last_seen) > p.lost_timeout_s:
                    yaw = p.search_yaw

                cv2.putText(
                    frame,
                    "tag: none",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            # Clamp overall RC for safety
            lr = int(clamp(lr, -p.max_rc, p.max_rc))
            fb = int(clamp(fb, -p.max_rc, p.max_rc))
            ud = int(clamp(ud, -p.max_rc, p.max_rc))
            yaw = int(clamp(yaw, -p.max_rc, p.max_rc))

            if flying:
                tello.send_rc_control(lr, fb, ud, yaw)

            # UI
            cv2.line(frame, (int(cx), 0), (int(cx), h), (255, 255, 255), 1)
            cv2.line(frame, (0, int(cy)), (w, int(cy)), (255, 255, 255), 1)
            cv2.putText(
                frame,
                f"flying={flying} rc fb={fb} yaw={yaw} ud={ud}",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("tello-apriltag-nav", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("t") and not flying:
                print("TAKEOFF")
                tello.takeoff()
                flying = True
                last_seen = time.time()
            if key == ord("l") and flying:
                print("LAND")
                tello.land()
                flying = False

            # pace loop
            dt = time.time() - t0
            if dt < dt_target:
                time.sleep(dt_target - dt)

    finally:
        try:
            if flying:
                tello.land()
        except Exception:
            pass
        try:
            tello.send_rc_control(0, 0, 0, 0)
        except Exception:
            pass
        try:
            tello.streamoff()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
