"""
RUN ALL - drive every video through all seven stages

Calls stage01 ... stage07 in order for each video in VIDEOS and prints a
summary. The stages hand work to each other through FILES, not function
arguments, so any stage can be re-run alone with --from_stage:

    1  video file           -> 01_original_frames/ + video_meta.json
    2  01_original_frames/  -> 02_frames_5fps/     + reduce_meta.json
    3  02_frames_5fps/      -> 03_gray_enhanced/
    4  03_gray_enhanced/    -> 04_feature_vectors/all_features.npy
    5  all_features.npy     -> 05_clusters_5fps/labels.npy + report
    6  labels.npy + stage 1 -> 06_clusters_original/
    7  the json + labels    -> video_info/*.txt

    python run_all.py                 # all videos
    python run_all.py --only three    # just that one
    python run_all.py --keep          # do not wipe output/ first
    python run_all.py --from_stage 5  # reuse frames and features
"""

import os
import sys
import time
import shutil
import argparse

# the stage modules live beside this file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import stage01_extract_frames as s1
import stage02_reduce_fps as s2
import stage03_enhance as s3
import stage04_features as s4
import stage05_cluster_onm as s5
import stage06_map_to_original as s6
import stage07_report as s7


# ======================================================================
# CONFIGURATION
# ======================================================================
# The key names the output folder and labels the report. It selects nothing:
# stage 5 gives every clip the same parameters, so adding, renaming or
# swapping an input here cannot change how any other video is clustered.
VIDEOS = [
    ("input1", "input1.mp4"),
    ("input2", "input2.mp4"),
    ("input3", "input3.mp4"),
    ("input4", "input4.mp4"),
    ("input5", "input5.mp4"),
]

TARGET_FPS = 5


def run_video(name, video_path, out_root, target_fps, from_stage,
              fix_gamma, save_reduced):
    """Drive one video through stages 1-6. Stage 7 runs once at the end
    for all videos, because two of its three outputs are summaries."""
    video_dir = os.path.join(out_root, name)
    print(f"\n{'='*66}\n{name.upper()}  -  {os.path.basename(video_path)}"
          f"\n{'='*66}")

    if from_stage <= 1:
        if not os.path.isfile(video_path):
            print(f"  [skip] '{video_path}' not found")
            return False
        info = s1.extract_frames(video_path,
                                 os.path.join(video_dir, "01_original_frames"))
        if info is None or info["extracted_frames"] == 0:
            return False

    if from_stage <= 2:
        import json
        with open(os.path.join(video_dir, "01_original_frames",
                               "video_meta.json")) as f:
            vm = json.load(f)
        s2.reduce_fps(os.path.join(video_dir, "01_original_frames"),
                      os.path.join(video_dir, "02_frames_5fps"),
                      vm["fps"], target_fps, (vm["width"], vm["height"]),
                      save_video_to=(os.path.join(
                          video_dir, f"reduced_{target_fps:g}fps.mp4")
                          if save_reduced else None))

    if from_stage <= 3:
        s3.enhance_dir(os.path.join(video_dir, "02_frames_5fps"),
                       os.path.join(video_dir, "03_gray_enhanced"),
                       fix_gamma)

    if from_stage <= 4:
        s4.build_features(os.path.join(video_dir, "03_gray_enhanced"),
                          os.path.join(video_dir, "04_feature_vectors"))

    if from_stage <= 5:
        # every clip gets the same parameters; nothing is keyed by name
        if s5.cluster_video(video_dir, name, s5.resolve_params()) is None:
            return False

    if from_stage <= 6:
        s6.run(video_dir)

    return True


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--videos_dir", default=os.path.dirname(here),
                    help="folder holding the input .mp4 files "
                         "(default: the parent of this folder)")
    ap.add_argument("--out", default=os.path.join(here, "output"),
                    help="root output folder (default: ./output)")
    ap.add_argument("--only", default=None,
                    help=f"process one video only: "
                         f"{' | '.join(n for n, _ in VIDEOS)}")
    ap.add_argument("--target_fps", type=float, default=TARGET_FPS,
                    help=f"reduced frame rate (default: {TARGET_FPS})")
    ap.add_argument("--from_stage", type=int, default=1, choices=range(1, 8),
                    help="skip earlier stages and reuse what is on disk")
    ap.add_argument("--fix_gamma", action="store_true",
                    help="stage 3: apply the CORRECT gamma exponent; this "
                         "changes the features, so stage 5 needs re-tuning")
    ap.add_argument("--save_reduced_video", action="store_true",
                    help="stage 2: also write the reduced .mp4")
    ap.add_argument("--keep", action="store_true",
                    help="do not wipe the output folder before running")
    args = ap.parse_args()

    jobs = [(n, v) for n, v in VIDEOS if args.only in (None, n)]
    if not jobs:
        raise SystemExit(f"--only must be one of: "
                         f"{', '.join(n for n, _ in VIDEOS)}")

    # a stale run leaves frames behind that the globs would pick up; skipped
    # for --from_stage, which exists precisely to reuse earlier output
    if os.path.isdir(args.out) and not args.keep and args.from_stage == 1:
        if args.only:
            target = os.path.join(args.out, args.only)
            if os.path.isdir(target):
                shutil.rmtree(target)
                print(f"[clean] removed {target}/")
        else:
            shutil.rmtree(args.out)
            print(f"[clean] removed {args.out}/ and everything in it")
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    done = []
    for name, video in jobs:
        if run_video(name, os.path.join(args.videos_dir, video), args.out,
                     args.target_fps, args.from_stage,
                     args.fix_gamma, args.save_reduced_video):
            done.append(name)

    if not done:
        print("\nNothing was processed.")
        return

    # stage 7 runs once, over everything, because two of its three outputs
    # are all-video summaries
    print(f"\n{'='*66}\nSTAGE 7 - reports\n{'='*66}")
    sys.argv = ["stage07_report.py", "--out", args.out]
    if args.only:
        sys.argv += ["--only", args.only]
    s7.main()

    states = [s for s in (s7.read_video_state(os.path.join(args.out, n))
                          for n in done) if s]
    print(f"\n{'='*66}\nDONE  ({time.time()-t0:.0f}s)\n{'='*66}")
    for s in states:
        o = s["onm"]
        line = (f"  {s['name']:6s} {s['orig']['extracted_frames']:5d} frames "
                f"-> {s['red']['frames']:4d} @ {args.target_fps:g}fps "
                f"-> {s['clusters']:2d} clusters")
        if o:
            line += f"  A(C)={o['A']:6.2f}%  D(C)={o['D']:5.2f}%"
        print(line)
    print(f"\n  output: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
