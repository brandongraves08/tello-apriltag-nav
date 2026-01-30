"""Build a simple AprilTag map from observation logs.

Input: `tag_observations.json` produced by `python -m src.tag_mapper ...`
Output: `tagmap.json` containing:
- per-tag estimated pose in a global map frame
- per-observation camera pose estimates (when possible)

This is intentionally a *first pass* map builder:
- It uses the first-seen tag as the global origin.
- When multiple tags are seen with pose in the same frame, it derives relative transforms.
- It accumulates edges and computes a best-effort pose using breadth-first propagation.

This is not full graph optimization (no bundle adjustment). It's enough to:
- get a rough tag layout
- start building "go to tag X" behaviors

Usage:
  python -m src.tagmap_build --in tag_observations.json --out tagmap.json

Roadmap:
- add robust averaging of edges
- add loop-closure refinement (least squares)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build tag map from observations")
    ap.add_argument("--in", dest="inp", required=True, help="tag_observations.json")
    ap.add_argument("--out", dest="out", default="tagmap.json")
    ap.add_argument("--origin", type=int, default=None, help="tag id to use as map origin (default: first seen)")
    return ap.parse_args()


def T_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def main() -> None:
    args = parse_args()
    data = json.loads(Path(args.inp).read_text())
    obs = data.get("observations", [])

    # group detections by (approx) timestamp bucket to find multi-tag co-observations
    # tag_mapper currently appends entries sequentially per frame; timestamps are close.
    # We'll bucket by rounding to 0.1s.
    buckets = defaultdict(list)
    for o in obs:
        if "pose_t_m" not in o or "pose_R" not in o:
            continue
        key = round(float(o["t"]) * 10) / 10.0
        buckets[key].append(o)

    # Collect relative edges between tags: T_a_b (tag b pose in tag a frame)
    edges = defaultdict(list)  # (a,b) -> [T]

    seen_tags = []
    seen_set = set()

    for _, items in buckets.items():
        # convert each detection into T_cam_tag (tag pose in camera frame)
        Ts = {}
        for it in items:
            tid = int(it["tag_id"])
            R = np.array(it["pose_R"], dtype=float)
            t = np.array(it["pose_t_m"], dtype=float)
            Ts[tid] = T_from_Rt(R, t)
            if tid not in seen_set:
                seen_set.add(tid)
                seen_tags.append(tid)

        tids = list(Ts.keys())
        for i in range(len(tids)):
            for j in range(len(tids)):
                if i == j:
                    continue
                a = tids[i]
                b = tids[j]
                # T_a_b = inv(T_cam_a) * T_cam_b
                T_a_b = inv_T(Ts[a]) @ Ts[b]
                edges[(a, b)].append(T_a_b)

    if not seen_tags:
        raise SystemExit("No metric tag poses found. Run tag_mapper with --calibration and --tag-size-m.")

    origin = args.origin if args.origin is not None else seen_tags[0]

    # Average edges crudely: mean translation + nearest rotation via SVD
    def avg_T(T_list: list[np.ndarray]) -> np.ndarray:
        ts = np.stack([T[:3, 3] for T in T_list], axis=0)
        t_mean = ts.mean(axis=0)

        Rs = np.stack([T[:3, :3] for T in T_list], axis=0)
        R_mean = Rs.mean(axis=0)
        # project to SO(3)
        U, _, Vt = np.linalg.svd(R_mean)
        R_hat = U @ Vt
        if np.linalg.det(R_hat) < 0:
            U[:, -1] *= -1
            R_hat = U @ Vt
        return T_from_Rt(R_hat, t_mean)

    avg_edges = {}
    for (a, b), Ts in edges.items():
        avg_edges[(a, b)] = avg_T(Ts)

    # BFS propagate tag poses in global frame: T_origin_tag
    poses = {origin: np.eye(4)}
    q = deque([origin])

    # build adjacency
    adj = defaultdict(list)
    for (a, b) in avg_edges.keys():
        adj[a].append(b)

    while q:
        a = q.popleft()
        T_origin_a = poses[a]
        for b in adj[a]:
            if b in poses:
                continue
            T_a_b = avg_edges[(a, b)]
            poses[b] = T_origin_a @ T_a_b
            q.append(b)

    # Output
    out = {
        "schema": "tello-apriltag-nav/tagmap-v1",
        "origin_tag_id": origin,
        "tags": {
            str(tid): {
                "T_origin_tag": poses[tid].tolist(),
                "position_m": poses[tid][:3, 3].tolist(),
            }
            for tid in poses.keys()
        },
        "unresolved_tags": [tid for tid in seen_tags if tid not in poses],
        "edge_counts": {f"{a}->{b}": len(edges[(a, b)]) for (a, b) in edges.keys()},
    }

    Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
