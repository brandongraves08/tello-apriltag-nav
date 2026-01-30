"""Minimal connectivity + takeoff sanity check for DJI Tello (EDU).

Run (Windows):
  python test_takeoff.py

Notes:
- Connect your computer to the Tello's Wi‑Fi first.
- Close the DJI Tello app (and any other controller apps) while testing SDK control.
"""

from djitellopy import Tello
from djitellopy.tello import TelloException


def main() -> None:
    t = Tello()
    print("Connecting...")
    t.connect()

    try:
        print("battery", t.get_battery())
    except Exception:
        print("battery (failed)")

    try:
        print("temp", t.get_temperature())
    except Exception:
        pass

    print("takeoff...")
    try:
        t.takeoff()
        print("takeoff ok")
    except TelloException as e:
        print("takeoff failed:", e)
        return

    print("land...")
    try:
        t.land()
        print("land ok")
    except TelloException as e:
        print("land failed:", e)


if __name__ == "__main__":
    main()
