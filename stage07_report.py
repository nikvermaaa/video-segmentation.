"""
STAGE 7 - METADATA REPORT, BEFORE AND AFTER REDUCTION

Reads and formats what stages 1, 2 and 5 wrote to disk; computes nothing
new, so the report can never disagree with the pipeline that produced it.

    in   01_original_frames/video_meta.json, 02_frames_5fps/reduce_meta.json,
         05_clusters_5fps/labels.npy, onm_report.txt
    out  video_info/<name>_info.txt        one per video
         video_info/all_videos_info.txt    summary of every video
         video_info/onm_summary.txt        N, A(C), D(C), T, CC, SL

--only refreshes the per-video file and leaves the two summaries alone,
since regenerating them from a single video would replace a complete table
with a one-line one.

    python stage07_report.py --out output
    python stage07_report.py --out output --only one
"""

import os
import re
import json
import glob
import argparse

import numpy as np


def read_video_state(video_dir):
    """Gather everything the earlier stages left on disk for one video.
    Returns None if the video was never processed."""
    vm = os.path.join(video_dir, "01_original_frames", "video_meta.json")
    rm = os.path.join(video_dir, "02_frames_5fps", "reduce_meta.json")
    lb = os.path.join(video_dir, "05_clusters_5fps", "labels.npy")
    if not (os.path.isfile(vm) and os.path.isfile(rm) and os.path.isfile(lb)):
        return None

    with open(vm) as f:
        orig = json.load(f)
    with open(rm) as f:
        red = json.load(f)
    labels = np.load(lb)

    state = {"name": os.path.basename(os.path.normpath(video_dir)),
             "orig": orig, "red": red,
             "clusters": int(labels.max()) + 1,
             "onm": parse_onm_report(os.path.join(video_dir,
                                                  "onm_report.txt"))}
    return state


def parse_onm_report(path):
    """Pull T, CC, SL, N and A(C)/D(C) back out of stage 5's report, rather
    than recomputing them. Returns None if the report is absent."""
    if not os.path.isfile(path):
        return None
    txt = open(path).read()

    def grab(pattern, cast=float):
        m = re.search(pattern, txt)
        return cast(m.group(1)) if m else None

    return {
        "T":  grab(r"Similarity T\s*:\s*([\d.]+)"),
        "CC": grab(r"Control centroid CC\s*:\s*(\d+)", int),
        "SL": grab(r"Similarity limit SL\s*:\s*([\d.]+)"),
        "N_centroids": grab(r"Distinct centroid objects \(N\)\s*:\s*(\d+)", int),
        "N_clusters":  grab(r"Clusters with members\s*:\s*(\d+)", int),
        "A": grab(r"A\(C\) Intra Association\s*:\s*([\d.]+)"),
        "D": grab(r"D\(C\) Intra Divergence\s*:\s*([\d.]+)"),
    }


def write_info(state, path):
    orig, red = state["orig"], state["red"]
    with open(path, "w") as f:
        f.write(f"VIDEO INFORMATION - {state['name']}\n")
        f.write("=" * 58 + "\n\n")
        f.write(f"Source file : {orig.get('source_file', '?')}\n\n")

        f.write("BEFORE - original video\n")
        f.write("-" * 58 + "\n")
        f.write(f"  Resolution        : {orig['width']} x {orig['height']}\n")
        f.write(f"  FPS               : {orig['fps']:.2f}\n")
        f.write(f"  Frames (reported) : {orig['reported_frames']}\n")
        f.write(f"  Frames (extracted): {orig['extracted_frames']}\n")
        f.write(f"  Duration          : {orig['duration']:.2f} seconds\n\n")

        f.write(f"AFTER - reduced to {red['target_fps']:g} fps\n")
        f.write("-" * 58 + "\n")
        f.write(f"  Resolution        : {orig['width']} x {orig['height']} "
                f"(unchanged)\n")
        f.write(f"  FPS               : {red['target_fps']:g}\n")
        f.write(f"  Frames            : {red['frames']}\n")
        f.write(f"  Duration          : {red['duration']:.2f} seconds\n")
        f.write(f"  Sampling stride   : every {red['stride']}th frame\n")
        f.write(f"  Reduction         : {orig['extracted_frames']} -> "
                f"{red['frames']} frames "
                f"({100*red['frames']/max(orig['extracted_frames'],1):.1f}% kept)\n\n")

        f.write("RESULT\n")
        f.write("-" * 58 + "\n")
        f.write(f"  Scenes found      : {state['clusters']}\n")
        f.write(f"  Mapped back onto  : {orig['extracted_frames']} "
                f"original frames\n")

        o = state["onm"]
        if o:
            f.write("\nONM / ECVM\n")
            f.write("-" * 58 + "\n")
            f.write(f"  Similarity T      : {o['T']:.4f}\n")
            f.write(f"  Control centroid  : {o['CC']}\n")
            f.write(f"  Similarity limit  : {o['SL']:.4f}\n")
            f.write(f"  SDCO centroids N  : {o['N_centroids']}\n")
            f.write(f"  A(C) association  : {o['A']:.2f} %\n")
            f.write(f"  D(C) divergence   : {o['D']:.2f} %\n")
    print(f"  [7] metadata -> video_info/{os.path.basename(path)}")


def write_summary(states, path):
    with open(path, "w") as f:
        f.write("ALL VIDEOS - SUMMARY\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"{'Video':8s} {'Resolution':12s} {'FPS':>6s} "
                f"{'Frames':>8s} {'5fps':>7s} {'Dur(s)':>8s} {'Scenes':>7s}\n")
        f.write("-" * 78 + "\n")
        for s in states:
            o = s["orig"]
            f.write(f"{s['name']:8s} "
                    f"{o['width']}x{o['height']:<7d} "
                    f"{o['fps']:6.2f} "
                    f"{o['extracted_frames']:8d} "
                    f"{s['red']['frames']:7d} "
                    f"{o['duration']:8.2f} "
                    f"{s['clusters']:7d}\n")
    print(f"[summary] -> {path}")


def write_onm_summary(states, path):
    """The ECVM side of the summary. 'SDCO N' and 'clusters' are both shown
    on purpose: SDCO N is what the paper's count produces, clusters is what
    survives stage 5's post-processing passes."""
    with open(path, "w") as f:
        f.write("ONM CLUSTERING - ALL VIDEOS\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"{'Video':8s} {'Frames':>7} {'5fps':>6} {'SDCO N':>7} "
                f"{'Clust':>6} {'A(C)%':>8} {'D(C)%':>8} {'T':>9} "
                f"{'CC':>5} {'SL':>8}\n")
        f.write("-" * 78 + "\n")
        for s in states:
            o = s["onm"]
            if not o:
                continue
            f.write(f"{s['name']:8s} {s['orig']['extracted_frames']:7d} "
                    f"{s['red']['frames']:6d} {o['N_centroids']:7d} "
                    f"{o['N_clusters']:6d} "
                    f"{o['A']:8.2f} {o['D']:8.2f} {o['T']:9.3f} "
                    f"{o['CC']:5d} {o['SL']:8.2f}\n")
    print(f"[onm summary] -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", required=True,
                    help="root output folder holding the per-video folders")
    ap.add_argument("--only", default=None,
                    help="report on one video only; leaves the two "
                         "summary files untouched")
    args = ap.parse_args()

    info_dir = os.path.join(args.out, "video_info")
    os.makedirs(info_dir, exist_ok=True)

    dirs = sorted(d for d in glob.glob(os.path.join(args.out, "*"))
                  if os.path.isdir(d)
                  and os.path.basename(d) != "video_info")
    if args.only:
        dirs = [d for d in dirs if os.path.basename(d) == args.only]
        if not dirs:
            raise SystemExit(f"no folder {args.out}/{args.only}")

    states = []
    for d in dirs:
        st = read_video_state(d)
        if st is None:
            print(f"  [skip] {os.path.basename(d)} - stages 1/2/5 incomplete")
            continue
        write_info(st, os.path.join(info_dir, f"{st['name']}_info.txt"))
        states.append(st)

    if not states:
        raise SystemExit("nothing to report on")

    if args.only:
        print(f"\n[note] --only {args.only}: per-video file updated; "
              f"all_videos_info.txt and onm_summary.txt left untouched "
              f"(re-run without --only to refresh them)")
    else:
        write_summary(states, os.path.join(info_dir, "all_videos_info.txt"))
        write_onm_summary(states, os.path.join(info_dir, "onm_summary.txt"))


if __name__ == "__main__":
    main()
