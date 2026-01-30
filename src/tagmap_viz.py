"""Visualize a tagmap.json as a simple 2D plot (top-down X/Y).

Usage:
  python -m src.tagmap_viz --in tagmap.json

Writes: tagmap_plot.png

Note: We assume the map frame is roughly "flat" for visualization; we plot X vs Y
from the homogeneous transform.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot tag map")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", default="tagmap_plot.png")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(Path(args.inp).read_text())
    tags = data.get("tags", {})

    xs, ys, labels = [], [], []
    for tid, info in tags.items():
        pos = info.get("position_m")
        if not pos:
            continue
        x, y = float(pos[0]), float(pos[1])
        xs.append(x)
        ys.append(y)
        labels.append(tid)

    plt.figure(figsize=(6, 6))
    plt.scatter(xs, ys)
    for x, y, lab in zip(xs, ys, labels):
        plt.text(x, y, lab)

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("AprilTag map (rough)")
    plt.axis("equal")
    plt.grid(True)

    Path(args.out).write_bytes(b"")
    plt.savefig(args.out, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
