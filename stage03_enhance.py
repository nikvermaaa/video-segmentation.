"""
STAGE 3 - GREYSCALE CONVERSION + CLAHE ENHANCEMENT

Per frame: resize to 320x240 -> 3x3 Gaussian blur -> CLAHE (clip 2.0,
8x8 tiles) -> auto gamma. CLAHE is the real work: it equalises local
contrast so two frames of one shot at different exposures still look
alike to stage 4.

    in   02_frames_5fps/frame_0000.jpg ...
    out  03_gray_enhanced/frame_0000.jpg ...  single-channel, 320x240

    python stage03_enhance.py --video_dir output/one
"""

import os
import glob
import argparse

import cv2
import numpy as np


ENHANCE_SIZE = (320, 240)     # frames are resized before enhancement
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)


def enhance_frame(gray, clip_limit=CLAHE_CLIP, tile_grid=CLAHE_GRID,
                  resize_to=ENHANCE_SIZE, auto_gamma=True, fix_gamma=False):
    """resize -> denoise -> CLAHE -> auto gamma, tuned for feature
    extraction."""
    if resize_to is not None:
        gray = cv2.resize(gray, resize_to, interpolation=cv2.INTER_AREA)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)      # so CLAHE doesn't amplify noise
    gray = cv2.createCLAHE(clipLimit=clip_limit,
                           tileGridSize=tile_grid).apply(gray)

    if auto_gamma:
        mean = max(gray.mean(), 1e-6)
        gamma = float(np.clip(np.log(0.5) / np.log(mean / 255.0), 0.5, 2.0))
        # --fix_gamma applies the correct one.
        # always apply fix gamma the results, more acurate at times and alos been tested, withoput it even works, give nearly acurate one, but still fix gamma stays true to the logic
        exponent = gamma if fix_gamma else 1.0 / gamma
        lut = np.array([((i / 255.0) ** exponent) * 255
                        for i in range(256)]).astype("uint8")
        gray = cv2.LUT(gray, lut)
    return gray


def enhance_dir(src_dir, out_dir, fix_gamma=False):
    """Enhance every jpg in src_dir. Returns the list of written paths."""
    frames = sorted(glob.glob(os.path.join(src_dir, "*.jpg")))
    if not frames:
        raise SystemExit(f"no frames in {src_dir} - run stage 2 first")

    os.makedirs(out_dir, exist_ok=True)
    for p in glob.glob(os.path.join(out_dir, "*.jpg")):
        os.remove(p)                       # clear a previous run

    paths = []
    for p in frames:
        gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)     # greyscale conversion
        if gray is None:
            continue
        out = os.path.join(out_dir, os.path.basename(p))
        cv2.imwrite(out, enhance_frame(gray, fix_gamma=fix_gamma))
        paths.append(out)

    print(f"  [3] {len(paths)} greyscale+enhanced frames "
          f"-> {os.path.basename(out_dir)}/"
          + ("   [--fix_gamma active]" if fix_gamma else ""))
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--video_dir", required=True,
                    help="per-video folder holding 02_frames_5fps/")
    ap.add_argument("--fix_gamma", action="store_true",
                    help="apply the CORRECT gamma exponent; changes the "
                         "features, so stage 5 would need re-tuning")
    args = ap.parse_args()

    d2 = os.path.join(args.video_dir, "02_frames_5fps")
    d3 = os.path.join(args.video_dir, "03_gray_enhanced")
    enhance_dir(d2, d3, args.fix_gamma)
    print(f"      -> {os.path.abspath(d3)}")


if __name__ == "__main__":
    main()
