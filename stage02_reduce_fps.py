"""
STAGE 2 - REDUCE THE FRAME RATE TO 5 FPS

Keeps every stride-th frame of stage 1's output, where
stride = round(original_fps / target_fps), renumbered from zero.
Sampling the frames avoids the lossy mp4 re-encode that would otherwise
add compression artefacts to the texture stage 4 measures.

    in   01_original_frames/frame_0000.jpg ... + video_meta.json
    out  02_frames_5fps/frame_0000.jpg ...  the kept frames, renumbered
         02_frames_5fps/reduce_meta.json   stride, target_fps, count

    python stage02_reduce_fps.py --video_dir output/one --target_fps 5
"""

import os
import json
import glob
import argparse

import cv2


TARGET_FPS = 5


def reduce_fps(src_dir, out_dir, original_fps, target_fps, size,
               save_video_to=None):
    """Keep every stride-th frame of src_dir. Returns the reduction
    metadata dict."""
    stride = max(1, round(original_fps / target_fps))
    frames = sorted(glob.glob(os.path.join(src_dir, "*.jpg")))
    if not frames:
        raise SystemExit(f"no frames in {src_dir} - run stage 1 first")

    os.makedirs(out_dir, exist_ok=True)
    for p in glob.glob(os.path.join(out_dir, "*.jpg")):
        os.remove(p)                       # clear a previous run

    writer = None
    if save_video_to:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_video_to, fourcc, target_fps, size)

    kept = 0
    for i, p in enumerate(frames):
        if i % stride:
            continue
        img = cv2.imread(p)
        cv2.imwrite(os.path.join(out_dir, f"frame_{kept:04d}.jpg"), img)
        if writer is not None:
            writer.write(img)
        kept += 1
    if writer is not None:
        writer.release()
        print(f"      also wrote {save_video_to}")

    print(f"  [2] {len(frames)} -> {kept} frames "
          f"(every {stride}th, {target_fps} fps) "
          f"-> {os.path.basename(out_dir)}/")

    meta = {"stride": stride, "target_fps": target_fps, "frames": kept,
            "source_frames": len(frames),
            "duration": kept / target_fps if target_fps else 0.0}
    with open(os.path.join(out_dir, "reduce_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--video_dir", required=True,
                    help="per-video folder holding 01_original_frames/")
    ap.add_argument("--target_fps", type=float, default=TARGET_FPS,
                    help=f"reduced frame rate (default: {TARGET_FPS})")
    ap.add_argument("--source_fps", type=float, default=None,
                    help="override the fps recorded by stage 1")
    ap.add_argument("--save_video", action="store_true",
                    help="also write reduced_<fps>fps.mp4 (not used "
                         "downstream - re-encoded, so lossy)")
    args = ap.parse_args()

    d1 = os.path.join(args.video_dir, "01_original_frames")
    d2 = os.path.join(args.video_dir, "02_frames_5fps")

    meta_path = os.path.join(d1, "video_meta.json")
    if args.source_fps is None:
        if not os.path.isfile(meta_path):
            raise SystemExit(f"{meta_path} missing - run stage 1 first, "
                             f"or pass --source_fps")
        with open(meta_path) as f:
            vm = json.load(f)
        source_fps, size = vm["fps"], (vm["width"], vm["height"])
    else:
        source_fps, size = args.source_fps, (0, 0)

    save_to = (os.path.join(args.video_dir,
                            f"reduced_{args.target_fps:g}fps.mp4")
               if args.save_video else None)
    reduce_fps(d1, d2, source_fps, args.target_fps, size, save_to)
    print(f"      -> {os.path.abspath(d2)}")


if __name__ == "__main__":
    main()
