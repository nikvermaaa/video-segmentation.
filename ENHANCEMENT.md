# Frame Enhancement

Covers **stage 3** — `stage03_enhance.py`. Greyscale conversion followed by a
four-step enhancement, applied to every 5 fps frame before any feature is
measured.

---

## 1. What this stage is for

This stage is **not** about making frames look good to a human. Nothing
downstream is ever viewed — `03_gray_enhanced/` exists only to be measured by
stage 4.

The goal is to make the 379 features describe **scene content** rather than
**camera exposure**. Two frames of the same shot, one slightly darker because
the auto-exposure drifted, should produce nearly identical feature vectors. If
they do not, stage 5 sees a large frame-to-frame jump, mistakes it for a cut,
and splits one shot into two clusters.

```
greyscale -> resize 320x240 -> blur 3x3 -> CLAHE -> auto gamma
```

---

## 2. Greyscale conversion

Done at load time:

```python
cv2.imread(path, cv2.IMREAD_GRAYSCALE)
```

Colour is discarded entirely, for three reasons:

* The texture and shape features that matter — GLCM, HOG, Canny — are all
  defined on a single channel anyway.
* It cuts the data by 3×, and stage 4 is the slow stage.
* Colour is the channel most corrupted by white-balance drift, which is
  exactly the nuisance variation this stage exists to suppress.

The cost: a cut between two shots differing *only* in colour, at identical
luminance and texture, is invisible to the pipeline.

---

## 3. Resize to 320×240

```python
cv2.resize(gray, (320, 240), interpolation=cv2.INTER_AREA)
```

From 1280×720, a 4× reduction on each axis and ~12× less data through the
expensive feature stage. It also suppresses sensor noise and compression
blocking, which are per-frame random and would otherwise register as texture
that changes every frame *within a single static shot*.

**`INTER_AREA` is the correct choice for downscaling.** It *averages* over the
source pixels being collapsed, which is a genuine low-pass filter. The faster
alternatives (`INTER_NEAREST`, `INTER_LINEAR`) *sample* instead of averaging
and alias high-frequency detail into false low-frequency patterns, which the
GLCM features would then measure as if it were real content

The resize does not preserve aspect ratio — 1280×720 is 16:9, 320×240 is 4:3 —
so frames are slightly squashed horizontally. Harmless here, because every
frame is squashed identically and all comparisons are frame-to-frame.

---

## 4. Gaussian blur 3×3

```python
cv2.GaussianBlur(gray, (3, 3), 0)
```

A light denoise, and **its position in the sequence is the point**: it must
come *before* CLAHE.

CLAHE is a local contrast amplifier and cannot distinguish faint detail from
sensor noise, so any noise still present when it runs gets amplified along
with everything else — and amplified noise is exactly the kind of per-frame
random variation that produces false scene boundaries.

The kernel is deliberately small. A 3×3 with sigma derived from the kernel
size kills single-pixel noise while leaving the edges Canny and HOG depend on
intact.

---

## 5. CLAHE

```python
cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
```

*Contrast Limited Adaptive Histogram Equalisation.* **This is the core of the
stage** — the step that delivers the exposure invariance section 1 asked for.

Plain global histogram equalisation uses ONE mapping for the whole frame, so a
bright window in the corner flattens everything else. CLAHE instead:

1. Divides the frame into an 8×8 grid of tiles (each 40×30 px here).
2. Equalises each tile against its **own** local histogram.
3. Interpolates bilinearly between neighbouring tiles so no seams appear.

**The "contrast limited" part is the `clipLimit`.** Within each tile any
histogram bin taller than the limit is clipped and the excess redistributed
evenly. Without it a tile of nearly uniform brightness — a patch of sky, a
plain wall — would have its tiny variations stretched across the full 0–255
range, turning invisible gradients into loud fake texture. `clipLimit = 2.0`
is a moderate setting; higher gives more local contrast and more amplified
noise.

CLAHE is inherently normalising: a shot that is globally dim and the same shot
correctly exposed converge to similar output, because each tile is rescaled
against its own local statistics. That is the exposure invariance the
clustering needs.

---

## 6. Auto gamma

```python
mean  = gray.mean()
gamma = clip( log(0.5) / log(mean/255), 0.5, 2.0 )
lut   = ((i/255) ** (1/gamma)) * 255   for i in 0..255
gray  = cv2.LUT(gray, lut)
```

A final per-frame brightness normalisation. The exponent is **solved for**
rather than fixed: to move a normalised mean m to 0.5 you need the power p
satisfying `m^p = 0.5`, hence `p = log(0.5)/log(m)`. The clip to [0.5, 2.0]
bounds how hard the correction can push, so a nearly black or nearly white
frame is not stretched violently.

A LUT is used because there are only 256 possible input values. The power is
evaluated 256 times and applied by table lookup rather than per pixel — for
320×240 that is 76800 pixels, roughly a 300× saving.

**Note the exponent actually applied.** `gamma` is already the exponent that
moves the mean to mid-grey, and the code applies `1.0 / gamma` — the
reciprocal — so brightness is pushed away from mid-grey rather than toward it.
It is deterministic and identical for every frame, so results stay
reproducible, but it works *against* the purpose stated in section 1: two
frames of one shot at different exposures are pushed further apart in
brightness rather than closer, which degrades the 9 statistical and 32
histogram features — 41 of the 379.

It is left as the default rather than silently changed, because switching it
alters every feature vector and therefore every clustering result.
`--fix_gamma` applies `gamma` directly; stage 5's thresholds would want
re-checking against the new features if you use it.

In the most cases go with fix gamma

---

## 7. Why this order

```
greyscale -> resize -> blur -> CLAHE -> gamma
```

| ordering | why |
|----------|-----|
| **resize before blur** | Blurring 1280×720 then discarding 15/16 of the pixels wastes the work. `INTER_AREA` is itself an averaging filter, so some denoising already happens in the resize. |
| **blur before CLAHE** | Mandatory. CLAHE amplifies whatever it is given, noise included. |
| **CLAHE before gamma** | CLAHE is local and redistributes brightness within tiles, so it changes the frame mean. The gamma step must see the *final* mean to target it. |

---

## 8. Parameters

| name | default | meaning |
|------|---------|---------|
| `ENHANCE_SIZE` | (320, 240) | Working resolution. Larger keeps fine texture and costs time; smaller denoises harder and may erase real detail. |
| `CLAHE_CLIP` | 2.0 | Contrast amplification limit. Higher gives punchier local contrast and more amplified noise. Above ~4.0, flat regions start generating fake texture. |
| `CLAHE_GRID` | (8, 8) | Tile grid. Finer means more local adaptation and less tolerance of large smooth areas. |
| `auto_gamma` | True | Per-frame brightness normalisation. |
| `--fix_gamma` | off | Applies `gamma` instead of `1/gamma`. Changes every feature vector; stage 5 would need re-tuning. |

---

## 9. What this stage feeds

`enhance_dir()` writes `03_gray_enhanced/`, which goes straight into stage 4.
**Every one of the 379 features is measured on these frames, never on the
originals:**

```
  9  statistics       depend on the brightness normalisation
 40  GLCM texture     benefits from blur + CLAHE
 32  histogram        depend on the brightness normalisation
288  HOG              benefits from CLAHE edge clarity
 10  Canny edges      benefits from blur (fewer noise edges)
```

The original-resolution colour frames in `01_original_frames/` are used again
only in **stage 6**, for boundary snapping — where raw pixel differences are
wanted precisely *because* they are unprocessed.
