"""
STAGE 4 - ONE FEATURE VECTOR PER FRAME  (379 raw -> PCA)

379 measurements per frame - 9 statistics, 40 GLCM texture, 32 histogram
bins, 288 HOG, 10 Canny edge densities - then z-scored and PCA-reduced to
95% of the variance, which lands at roughly 19-48 dims depending on the
video. No re-standardisation after PCA: the component ordering carries
real information and flattening it would give trailing noise equal weight.

    in   03_gray_enhanced/frame_0000.jpg ...
    out  04_feature_vectors/frame_0000.npy ...   one vector per frame
         04_feature_vectors/all_features.npy     the stacked (T x dims) matrix
         04_feature_vectors/frame_paths.txt      row order, for traceability

    python stage04_features.py --video_dir output/one
"""

import os
import glob
import argparse

import cv2
import numpy as np
from scipy.stats import skew, kurtosis, entropy
from skimage.feature import graycomatrix, graycoprops, hog
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PCA_VARIANCE = 0.95


def extract_features(gray, hist_bins=32):
    """379 features: 9 statistical + 40 GLCM + 32 histogram + 288 HOG + 10 edge."""
    feats = []
    pix = gray.astype(np.float64).ravel()

    # 1) first-order statistics (9)
    # skew and kurtosis divide by the standard deviation, so scipy returns
    # NaN for a constant frame (a fade to black) and PCA below then dies.
    # Both are 0 by convention for a degenerate distribution.
    sd = float(pix.std())
    feats += [pix.mean(), sd, pix.var()]
    feats += [skew(pix), kurtosis(pix)] if sd > 0 else [0.0, 0.0]
    hist_full, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
    feats += [entropy(hist_full + 1e-12), np.sum(hist_full ** 2),
              np.median(pix), pix.max() - pix.min()]

    # 2) GLCM / Haralick texture (40)
    q = (gray // 8).astype(np.uint8)                   # 32 grey levels
    glcm = graycomatrix(q, distances=[1, 2],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=32, symmetric=True, normed=True)
    for prop in ("contrast", "correlation", "energy",
                 "homogeneity", "dissimilarity"):
        feats.extend(graycoprops(glcm, prop).ravel())

    # 3) intensity histogram (32)
    hist32, _ = np.histogram(gray, bins=hist_bins, range=(0, 256), density=True)
    feats.extend(hist32)

    # 4) HOG - shape/structure, what separates humans from animals (288)
    small = cv2.GaussianBlur(cv2.resize(gray, (160, 120),
                                        interpolation=cv2.INTER_AREA), (5, 5), 0)
    feats.extend(hog(small, orientations=9, pixels_per_cell=(32, 32),
                     cells_per_block=(2, 2), block_norm="L2-Hys",
                     feature_vector=True))

    # 5) Canny edge density, global + 3x3 grid (10)
    edges = cv2.Canny(gray, 100, 200)
    feats.append(edges.mean() / 255.0)
    h, w = edges.shape
    for r in range(3):
        for c in range(3):
            feats.append(edges[r*h//3:(r+1)*h//3, c*w//3:(c+1)*w//3].mean()/255.0)

    return np.array(feats, dtype=np.float64)


def build_features(src_dir, out_dir):
    """Extract, standardise and PCA-reduce every frame in src_dir.
    Returns the (T x dims) matrix."""
    enhanced_paths = sorted(glob.glob(os.path.join(src_dir, "*.jpg")))
    if not enhanced_paths:
        raise SystemExit(f"no frames in {src_dir} - run stage 3 first")

    os.makedirs(out_dir, exist_ok=True)
    for p in glob.glob(os.path.join(out_dir, "*.npy")):
        os.remove(p)                       # clear a previous run

    raw = []
    for i, p in enumerate(enhanced_paths):
        raw.append(extract_features(cv2.imread(p, cv2.IMREAD_GRAYSCALE)))
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(enhanced_paths)} frames ...")
    raw = np.vstack(raw)

    # safety net behind the guard in extract_features; warns loudly because
    # a silent 0 is a feature that stopped measuring anything
    

    # one file per frame, as well as the stacked matrix
    for p, vec in zip(enhanced_paths, X):
        stem = os.path.splitext(os.path.basename(p))[0]
        np.save(os.path.join(out_dir, f"{stem}.npy"), vec)
    np.save(os.path.join(out_dir, "all_features.npy"), X)
    with open(os.path.join(out_dir, "frame_paths.txt"), "w") as f:
        f.write("\n".join(enhanced_paths))

    print(f"  [4] {raw.shape[1]} raw features -> {X.shape[1]} PCA dims, "
          f"{len(X)} per-frame vectors -> {os.path.basename(out_dir)}/")
    return X


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--video_dir", required=True,
                    help="per-video folder holding 03_gray_enhanced/")
    args = ap.parse_args()

    d3 = os.path.join(args.video_dir, "03_gray_enhanced")
    d4 = os.path.join(args.video_dir, "04_feature_vectors")
    build_features(d3, d4)
    print(f"      -> {os.path.abspath(os.path.join(d4, 'all_features.npy'))}")


if __name__ == "__main__":
    main()
