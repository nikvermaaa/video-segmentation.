# Feature Vector Extraction

Covers **stage 4** — `stage04_features.py`, where the pipeline stops working
on pixels and starts working on numbers.

---

## 1. What this stage does

Stage 5 needs to answer *"how similar are these two frames?"*. It cannot do
that on images. This stage turns each frame into a single vector, so that
question becomes an ordinary Euclidean distance.

```
03_gray_enhanced/frame_0000.jpg ...        the enhanced greyscale frames
  -> 379 raw measurements per frame
  -> z-score every dimension
  -> PCA to 95% of the variance
  -> 04_feature_vectors/all_features.npy   (frames x dims)
```

---

## 2. The 379 raw features

Five groups, chosen so that a blind spot in one is covered by another.

```
  9  first-order statistics   mean, std, var, skew, kurtosis,
                              entropy, energy, median, range
 40  GLCM / Haralick texture  contrast, correlation, energy,
                              homogeneity, dissimilarity,
                              over 2 distances x 4 angles
 32  intensity histogram      32-bin, density-normalised
288  HOG                      9 orientations, 32x32 cells,
                              2x2 block normalisation
 10  Canny edge density       whole frame + a 3x3 spatial grid
---
379  total
```

### Group 1 — first-order statistics (9)

Mean, standard deviation, variance, skew, kurtosis, Shannon entropy, energy
(`sum of p^2`), median, and range (`max - min`), over the flat pixel array.

These describe the *brightness distribution* and nothing about layout. Cheap,
and they separate a bright outdoor shot from a dim indoor one instantly.

Skew and kurtosis divide by the standard deviation, so they are undefined for
a constant frame — a fade to black — and scipy returns NaN rather than
raising. Both are set to 0 there, by the convention for a degenerate
distribution; without that guard the PCA below fails.

### Group 2 — GLCM / Haralick texture (40)

The grey-level co-occurrence matrix counts how often a pixel of value *i* sits
at a given offset from a pixel of value *j*. Five properties are read off it —
contrast, correlation, energy, homogeneity, dissimilarity — at 2 distances
(1, 2 px) × 4 angles (0°, 45°, 90°, 135°). 5 × 2 × 4 = 40.

**Quantised to 32 grey levels first** (`gray // 8`): a full 256-level GLCM is
a 256×256 matrix per angle per distance. At 320×240 = 76800 pixels most of
those 65536 cells would hold 0 or 1, so the properties would be estimated from
almost no samples. A 32×32 matrix gives ~75 samples per cell and stable
properties.

Texture is what distinguishes grass from water from fur at the same average
brightness, which group 1 cannot see. **Multiple angles matter** because
texture is often directional — reeds, fur, a fence — and a single angle would
call a vertical texture and a horizontal one identical.

### Group 3 — intensity histogram (32)

A 32-bin density-normalised histogram: the *shape* of the brightness
distribution rather than its summary statistics.

Group 1 already has the mean and std, but two very different frames can share
both — a bimodal frame (dark subject against bright sky) and a flat mid-grey
frame do. The histogram sees the difference. Density-normalised, so it is
comparable regardless of frame size.

### Group 4 — HOG (288)

Histogram of Oriented Gradients: 9 orientation bins, 32×32 pixel cells, 2×2
cell blocks, L2-Hys block normalisation, computed on a 160×120 downscale that
has been Gaussian blurred (5×5) first.

This is **288 of the 379 features — 76% of the raw vector**. HOG describes
*shape and spatial structure*: where the edges are and which way they point.
It is what separates a human figure from a four-legged animal from a
landscape, the distinction brightness and texture cannot make.

**Blurred and downscaled again**, even though stage 3 already resized to
320×240: HOG at the full working resolution produces thousands of features
dominated by fine detail that changes every frame within one static shot
(leaves moving, sensor noise). The extra downscale plus blur targets HOG at
*composition* — the large-scale arrangement of the frame, which is stable
within a shot and changes at a cut.

**L2-Hys normalisation** makes each block invariant to local contrast changes,
a second line of defence behind stage 3's CLAHE.

### Group 5 — Canny edge density (10)

Canny edges (thresholds 100/200), then the mean edge density of the whole
frame plus each cell of a 3×3 grid. 1 + 9 = 10.

The grid is the point. A global edge density says *how much* detail there is;
the grid says **where** it is, so a shot with a busy left half and an empty
right half is distinguishable from its mirror image and from a shot with
detail spread evenly at the same total density.

### Why five groups

| group | blind to |
|-------|----------|
| statistics | layout, texture, shape |
| GLCM | where in the frame, overall brightness |
| histogram | layout, texture |
| HOG | absolute brightness, fine texture |
| edge grid | what the content actually is |

Two shots that fool all five simultaneously are rare. Two that fool one are
common.

---

## 3. Standardisation

```python
X = StandardScaler().fit_transform(raw)
```

The 379 raw features live on wildly different scales — variance of pixel
values runs into the thousands, Canny edge density is bounded in [0, 1]. In a
Euclidean distance a feature's influence is proportional to its scale, so
without z-scoring the variance feature alone would outweigh all 10 edge
features and all 288 HOG features combined, purely because its units are
bigger.

Z-scoring to mean 0, std 1 makes influence proportional to *how much a feature
varies across this video*, which is the useful criterion.

It must happen **before** PCA, because PCA maximises variance and would
otherwise simply discover the large-scale features and report them as the
principal components.

---

## 4. PCA to 95% variance

```python
X = PCA(n_components=0.95).fit_transform(X)
```

379 features, but nowhere near 379 independent facts. GLCM contrast at 0° and
at 45° are highly correlated, adjacent HOG cells overlap by construction, mean
and median track each other. PCA finds the directions that actually vary.

| video | frames | 379 → PCA dims |
|-------|--------|----------------|
| input1 | 167 | 48 |
| input2 | 197 | 14 |
| input3 | 290 | 26 |

A ~90% reduction with 95% of the variance kept.

### Why the dimension differs per video

The target is a **variance fraction**, not a fixed count, so a video of many
visually varied scenes needs more components to reach 95% than a video of a
few similar ones.

This matters downstream, because **stage 5's similarity rule is stated as "all
but 2 of the m dimensions must agree"**, so m changing per video changes how
strict that rule is:

| video | m | what "all but 2" demands |
|-------|---|--------------------------|
| input3 | 26 | 92.3% agreement |
| input2 | 14 | 85.7% agreement |
| input1 | 48 | 95.8% agreement |

It also means **no threshold downstream can be an absolute distance** — the
scale and dimensionality of the feature space differ for every video. That is
why stage 5's T is a percentile and its SL is `T × sqrt(dims)`.

### No re-standardisation after PCA

PCA components come out **ordered by how much variance they explain**, and
that ordering is meaningful: component 1 carries more of what distinguishes
frames than component 20. Every distance in stage 5 is a plain Euclidean norm,
so leaving the natural variance weighting in place is what keeps dominant
components dominant.

Re-standardising afterwards would flatten all components to equal weight and
hand the noisiest trailing component the same say as the principal one. The
z-scoring in section 3 is correct *because* the raw features' scales are
arbitrary units; the PCA components' scales are not arbitrary — they are the
signal.

---

## 5. What gets written

```
04_feature_vectors/
    frame_0000.npy ...      one vector per frame, for inspection
    all_features.npy        the stacked (frames x dims) matrix
    frame_paths.txt         row order, for traceability
```

`all_features.npy` is what stage 5 loads, and it is the entire handover — from
here on the pipeline has no idea it is looking at a video. The per-frame files
and `frame_paths.txt` exist so a single frame's vector can be examined without
re-running anything.

`frame_paths.txt` stores **absolute** paths, so it goes stale if the project
folder is moved or renamed; re-running stage 4 fixes it.

Because the file is a clean handover, the clustering can be re-run as often as
you like without touching a pixel:

```bash
python run_all.py --from_stage 5
python stage05_cluster_onm.py --video_dir output/input1 --cc 8
```

---

## 6. Parameters

| name | default | meaning |
|------|---------|---------|
| `PCA_VARIANCE` | 0.95 | Fraction of variance kept. Lower gives fewer dimensions and more aggressive denoising, at the cost of blunter distinctions. Changing it changes m, which changes how strict stage 5's m−2 rule is. |
| `hist_bins` | 32 | Histogram resolution, in `extract_features`. |

The GLCM distances and angles, the HOG geometry and the Canny thresholds are
inline in `extract_features()` rather than exposed as constants, because
changing any of them changes the length of the raw vector and invalidates
every fitted parameter in stage 5.

