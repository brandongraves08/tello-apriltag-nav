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

import argparse
import time
from dataclasses import dataclass

import cv2
import numpy as np
from djitellopy import Tello
from djitellopy.tello import TelloException
from pupil_apriltags import Detector


@dataclass
class Params:
    tag_family: str = "tag36h11"
    tag_id: int | None = None  # set to an int to follow only one tag

    # Safety gates
    min_battery_to_fly: int = 25  # percent
    max_flight_time_s: float = 5 * 60  # auto-land after this many seconds

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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Tello EDU AprilTag follower")
    ap.add_argument("--tag-id", type=int, default=None, help="Only follow this tag id (default: largest tag)")
    ap.add_argument("--area-target", type=float, default=None, help="Target tag pixel area (distance proxy)")
    ap.add_argument("--min-battery", type=int, default=None, help="Refuse takeoff if battery below this percent")
    ap.add_argument("--max-flight-time", type=float, default=None, help="Auto-land after N seconds")
    ap.add_argument("--use-ud", action="store_true", help="Enable up/down centering")
    ap.add_argument("--no-search", action="store_true", help="Disable search yaw when tag lost")
    ap.add_argument("--debug", action="store_true", help="Verbose prints")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    p = Params()

    # Apply CLI overrides
    if args.tag_id is not None:
        p.tag_id = args.tag_id
    if args.area_target is not None:
        p.area_target = float(args.area_target)
    if args.min_battery is not None:
        p.min_battery_to_fly = int(args.min_battery)
    if args.max_flight_time is not None:
        p.max_flight_time_s = float(args.max_flight_time)
    if args.use_ud:
        p.use_ud = True

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

    # Basic sanity info (helps debug 'error' responses)
    try:
        print("Battery:", tello.get_battery(), "%")
    except Exception:
        print("Battery: (failed to read)")
    try:
        print("Temp:", tello.get_temperature(), "C")
    except Exception:
        pass
    try:
        print("Height:", tello.get_height(), "cm")
    except Exception:
        pass
    print("Tip: If takeoff returns 'error', close the Tello app/controller, ensure battery >20%, and set Windows Firewall to allow Python on Private networks.")

    tello.streamon()
    frame_read = tello.get_frame_read()

    flying = False
    autonomy_enabled = True  # can be paused (panic hover)
    last_seen = 0.0
    takeoff_time = None  # type: float | None
    last_battery_check = 0.0
    battery_pct = None  # type: int | None
    was_tracking = False

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
                was_tracking = True

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
                # Tag lost: stop motion immediately; optionally search by yawing slowly.
                if was_tracking and flying:
                    try:
                        tello.send_rc_control(0, 0, 0, 0)
                    except Exception:
                        pass
                    was_tracking = False

                if flying and not args.no_search and (time.time() - last_seen) > p.lost_timeout_s:
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
                # Auto-land after max flight time as a simple safety net.
                if takeoff_time is not None and (time.time() - takeoff_time) > p.max_flight_time_s:
                    print(f"Max flight time reached ({p.max_flight_time_s}s). Landing...")
                    try:
                        tello.land()
                    except Exception:
                        pass
                    flying = False
                    autonomy_enabled = True
                    takeoff_time = None
                else:
                    if autonomy_enabled:
                        tello.send_rc_control(lr, fb, ud, yaw)
                    else:
                        # paused (panic hover)
                        tello.send_rc_control(0, 0, 0, 0)

            # UI
            cv2.line(frame, (int(cx), 0), (int(cx), h), (255, 255, 255), 1)
            cv2.line(frame, (0, int(cy)), (w, int(cy)), (255, 255, 255), 1)
            batt_txt = "?" if battery_pct is None else str(battery_pct)
            cv2.putText(
                frame,
                f"flying={flying} batt={batt_txt}% rc fb={fb} yaw={yaw} ud={ud}",
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

            if key == ord("h"):
                autonomy_enabled = not autonomy_enabled
                print("Autonomy:", "ON" if autonomy_enabled else "PAUSED (hover)")
            # Periodic battery check (don’t spam)
            now = time.time()
            if (now - last_battery_check) > 5.0:
                last_battery_check = now
                try:
                    battery_pct = int(tello.get_battery())
                except Exception:
                    pass

            if key == ord("t") and not flying:
                print("TAKEOFF")

                if battery_pct is not None and battery_pct < p.min_battery_to_fly:
                    print(
                        f"Refusing takeoff: battery {battery_pct}% < {p.min_battery_to_fly}%. Charge first."
                    )
                else:
                    try:
                        tello.takeoff()
                        flying = True
                        autonomy_enabled = True
                        takeoff_time = time.time()
                        last_seen = time.time()
                        was_tracking = False
                    except TelloException as e:
                        print("Takeoff failed:", e)
                        print(
                            "Common causes: already connected in DJI app, low battery, not on level surface, or firewall/VPN blocking UDP. "
                            "Try: close Tello app, disconnect other devices from drone Wi-Fi, disable VPN, allow Python through Windows Firewall (Private)."
                        )
                    except Exception as e:
                        print("Takeoff failed (unexpected):", repr(e))

            if key == ord("l"):
                print("LAND")
                try:
                    tello.land()
                    flying = False
                    takeoff_time = None
                except TelloException as e:
                    # Tello can respond 'error' even if it is already landing/landed.
                    print("Land returned error:", e)
                    flying = False
                    takeoff_time = None
                except Exception as e:
                    print("Land failed (unexpected):", repr(e))
                    flying = False
                    takeoff_time = None

            # pace loop
            dt = time.time() - t0
            if dt < dt_target:
                time.sleep(dt_target - dt)

    except KeyboardInterrupt:
        print("KeyboardInterrupt: exiting...")

    finally:
        # Be defensive here: Windows Ctrl+C can interrupt socket calls and
        # djitellopy's __del__ can throw if cleanup isn't graceful.
        try:
            tello.send_rc_control(0, 0, 0, 0)
        except Exception:
            pass

        try:
            if flying:
                # Land can respond 'error' if already on ground; ignore.
                tello.land()
        except Exception:
            pass

        try:
            tello.streamoff()
        except Exception:
            pass

        try:
            tello.end()
        except Exception:
            pass

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
