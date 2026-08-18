"""
STAGE 6 - MAP THE 5 FPS CLUSTERS BACK ONTO THE ORIGINAL FRAMES

Two steps:
    1. EXPAND  each 5 fps label covers `stride` original frames
    2. SNAP    each boundary is moved to the true cut, found by scanning the
               original frames inside its window

Expansion alone puts every boundary at the LAST frame of the window, which
is late by 0-4 frames and strands the opening frames of a new scene at the
tail of the previous cluster. The snap tests each candidate position by the
mean absolute difference between consecutive thumbnails; it is deliberately
crude, since stage 5 already decided WHETHER there is a cut, not where.

    in   05_clusters_5fps/labels.npy, 01_original_frames/*.jpg,
         02_frames_5fps/reduce_meta.json (for the stride)
    out  06_clusters_original/  cluster folders, labels.npy, segments.txt

    python stage06_map_to_original.py --video_dir output/one
"""

import os
import json
import glob
import shutil
import argparse

import cv2
import numpy as np


def _thumb(path, cache, size=(160, 120)):
    # cached because neighbouring boundaries re-examine the same frames
    if path not in cache:
        g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        cache[path] = (cv2.resize(g, size).astype(np.float32)
                       if g is not None else None)
    return cache[path]


def map_to_original(labels_5fps, original_frames, stride, refine=True):
    """Expand each label to the frames it represents, then snap every
    boundary to the true cut."""
    T_red, T_org = len(labels_5fps), len(original_frames)

    idx = np.minimum(np.arange(T_org) // stride, T_red - 1)
    labels = labels_5fps[idx]

    moves = []
    if refine:
        cuts = [i + 1 for i in range(T_org - 1) if labels[i] != labels[i + 1]]
        cache, refined = {}, []
        for c in cuts:
            best, best_d = c, -1.0
            for cand in range(max(1, c - stride + 1), min(T_org, c + 1)):
                a = _thumb(original_frames[cand - 1], cache)
                b = _thumb(original_frames[cand], cache)
                if a is None or b is None:
                    continue
                diff = float(np.abs(b - a).mean())
                if diff > best_d:
                    best_d, best = diff, cand
            refined.append(best)
            if best != c:
                moves.append(best - c)

        labels = np.zeros(T_org, dtype=int)
        for j, c in enumerate(sorted(set(refined))):
            labels[c:] = j + 1

    print(f"  [6] {T_red} labels -> {T_org} original frames (x{stride}); "
          f"{len(moves)} boundary(s) snapped to the true cut"
          + (f", max shift {max(abs(m) for m in moves)} frames" if moves else ""))
    return labels


def export_clusters(frame_paths, labels, out_dir):
    """Write one folder per cluster, plus labels.npy and segments.txt.

    stage05_cluster_onm.py carries the same function, because each stage
    file is meant to run on its own. Change the format in both."""
    os.makedirs(out_dir, exist_ok=True)

    # clear a previous run, or frames from an older partition survive
    for d in glob.glob(os.path.join(out_dir, "cluster_*")):
        if os.path.isdir(d) and os.path.basename(d)[8:].isdigit():
            shutil.rmtree(d)

    for p, lab in zip(frame_paths, labels):
        d = os.path.join(out_dir, f"cluster_{lab}")
        os.makedirs(d, exist_ok=True)
        shutil.copy(p, os.path.join(d, os.path.basename(p)))

    with open(os.path.join(out_dir, "segments.txt"), "w") as f:
        start, cur = 0, labels[0]
        for i in range(1, len(labels) + 1):
            if i == len(labels) or labels[i] != cur:
                f.write(f"frames {start:6d} - {i-1:6d}  ->  cluster {cur}\n")
                if i < len(labels):
                    start, cur = i, labels[i]

    np.save(os.path.join(out_dir, "labels.npy"), labels)


def run(video_dir, stride=None, refine=True):
    """Read stage 5's labels and stage 1's frames, write 06_clusters_original."""
    lab_path = os.path.join(video_dir, "05_clusters_5fps", "labels.npy")
    if not os.path.isfile(lab_path):
        raise SystemExit(f"{lab_path} missing - run stage 5 first")
    labels5 = np.load(lab_path)

    if stride is None:
        meta = os.path.join(video_dir, "02_frames_5fps", "reduce_meta.json")
        if not os.path.isfile(meta):
            raise SystemExit(f"{meta} missing - run stage 2 first, "
                             f"or pass --stride")
        with open(meta) as f:
            stride = json.load(f)["stride"]

    frames_org = sorted(glob.glob(os.path.join(video_dir,
                                               "01_original_frames", "*.jpg")))
    if not frames_org:
        raise SystemExit(f"no frames in {video_dir}/01_original_frames "
                         f"- run stage 1 first")

    labels_org = map_to_original(labels5, frames_org, stride, refine)
    export_clusters(frames_org, labels_org,
                    os.path.join(video_dir, "06_clusters_original"))
    return labels_org


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--video_dir", required=True,
                    help="per-video folder holding 01_original_frames/ "
                         "and 05_clusters_5fps/")
    ap.add_argument("--stride", type=int, default=None,
                    help="override the stride recorded by stage 2")
    ap.add_argument("--no_refine", action="store_true",
                    help="plain expansion only; leaves every boundary "
                         "0-4 frames late - see the header")
    args = ap.parse_args()

    run(args.video_dir, args.stride, refine=not args.no_refine)
    d6 = os.path.join(args.video_dir, "06_clusters_original")
    print(f"      -> {os.path.abspath(d6)}")


if __name__ == "__main__":
    main()
