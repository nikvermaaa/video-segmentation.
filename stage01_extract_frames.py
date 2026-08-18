"""
STAGE 1 - EXTRACT EVERY FRAME AT THE ORIGINAL FPS

Decodes one video and writes every frame to disk, plus the video's
properties so later stages need not re-open the video.

    in   a video file
    out  01_original_frames/frame_0000.jpg ...  every decoded frame
         01_original_frames/video_meta.json     fps, width, height, counts

    python stage01_extract_frames.py --video ../Input_video33sec.mp4 --out output/one
"""

import os
import json
import glob
import argparse

import cv2


def extract_frames(video_path, out_dir):
    """Decode video_path into out_dir. Returns the metadata dict, or None
    if the file cannot be opened."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [error] cannot open {video_path}")
        return None

    info = {
        "source_file": os.path.basename(video_path),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "reported_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration"] = (info["reported_frames"] / info["fps"]
                        if info["fps"] > 0 else 0.0)

    os.makedirs(out_dir, exist_ok=True)

    # clear a previous run, or its frames get mixed into this one
    for p in glob.glob(os.path.join(out_dir, "*.jpg")):
        os.remove(p)

    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(os.path.join(out_dir, f"frame_{saved:04d}.jpg"), frame)
        saved += 1
        if saved % 250 == 0:
            print(f"    {saved} frames ...")
    cap.release()

    # the decoded count is authoritative; a container header can lie
    info["extracted_frames"] = saved
    print(f"  [1] {saved} frames @ {info['fps']:.2f} fps "
          f"({info['width']}x{info['height']}) -> {os.path.basename(out_dir)}/")
    if info["reported_frames"] and saved != info["reported_frames"]:
        print(f"      note: container header said {info['reported_frames']}; "
              f"the decoded count is authoritative")

    with open(os.path.join(out_dir, "video_meta.json"), "w") as f:
        json.dump(info, f, indent=2)
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--video", required=True, help="path to the input video")
    ap.add_argument("--out", required=True,
                    help="per-video output folder; frames go in its "
                         "01_original_frames/ subfolder")
    args = ap.parse_args()

    if not os.path.isfile(args.video):
        raise SystemExit(f"no such video: {args.video}")

    d = os.path.join(args.out, "01_original_frames")
    if extract_frames(args.video, d) is None:
        raise SystemExit(1)
    print(f"      -> {os.path.abspath(d)}")


if __name__ == "__main__":
    main()
