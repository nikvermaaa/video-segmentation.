## How it works

The core idea is that **N is a count, not a search result**. Rather than
trying several values of k and scoring them, ONM counts how many *distinct*
objects in the dataset *repeat* at least CC times. Those recurring objects are
the centroids, and how many there are is the number of clusters.

Here an object is one video frame, represented by a PCA feature vector. So the
pipeline's job is to turn frames into vectors that are close within a shot and
far apart across a cut, then let ONM count the recurring ones.

```
video -> every frame -> 5 fps -> greyscale + CLAHE -> 379 features -> PCA
      -> SDCO counts distinct recurring frames  -> N centroids
      -> assign each frame to its nearest centroid
      -> smooth in time, drop weak boundaries
      -> map the labels back onto the original frame rate
```

`ALGORITHM.md` covers the counting and clustering in full.

---

## The seven stages

Each stage is one file. They communicate through **files on disk**, not
function calls — so any stage can be re-run on its own, and re-tuning the
clustering does not mean re-extracting thousands of frames.

| # | File | Reads | Writes |
|---|------|-------|--------|
| 1 | `stage01_extract_frames.py` | a `.mp4` | `01_original_frames/` + `video_meta.json` |
| 2 | `stage02_reduce_fps.py` | `01_original_frames/` | `02_frames_5fps/` + `reduce_meta.json` |
| 3 | `stage03_enhance.py` | `02_frames_5fps/` | `03_gray_enhanced/` |
| 4 | `stage04_features.py` | `03_gray_enhanced/` | `04_feature_vectors/all_features.npy` |
| 5 | `stage05_cluster_onm.py` | `all_features.npy` | `05_clusters_5fps/labels.npy`, `onm_report.txt` |
| 6 | `stage06_map_to_original.py` | `labels.npy` + `01_original_frames/` | `06_clusters_original/` |
| 7 | `stage07_report.py` | the `.json` files + `labels.npy` | `video_info/*.txt` |

`run_all.py` calls all seven in order.

**Stage 2** keeps every stride-th frame, where
`stride = round(original_fps / 5)`. It samples the frames rather than writing
and re-reading a 5 fps `.mp4`, which would add compression artefacts to the
exact texture and edge detail stage 4 goes on to measure.

**Stage 6** expands each 5 fps label back over `stride` original frames, then
snaps each boundary to the true cut by scanning the original frames inside its
window for the largest frame-to-frame difference. Plain expansion puts every
boundary at the last frame of the window, which strands the opening frames of
a new scene at the tail of the previous cluster.

**Stage 7** only reads and formats what stages 1, 2 and 5 wrote, so the
reports can never disagree with the run that produced them.

---

## Documentation

| document | covers |
|----------|--------|
| [SETUP.md](SETUP.md) | installing and running on a fresh machine (conda or venv), disk needs, troubleshooting |
| [ALGORITHM.md](ALGORITHM.md) | stage 5 — SDCO, clustering, ECVM, and the temporal smoothing built on top |
| [FEATURE_EXTRACTION.md](FEATURE_EXTRACTION.md) | stage 4 — the 379 features, standardisation, and PCA |
| [ENHANCEMENT.md](ENHANCEMENT.md) | stage 3 — greyscale, resize, blur, CLAHE, gamma, and why in that order |

Stages 1, 2, 6 and 7 are mechanical and documented in their own file headers.

---

## Running it

Full installation instructions, including conda, are in
**[SETUP.md](SETUP.md)**. The short version:

```bash
conda env create -f environment.yml     # or: pip install -r requirements.txt
conda activate onm

python run_all.py                  # all videos
python run_all.py --only input2    # just one
python run_all.py --from_stage 5   # reuse frames + features, redo clustering
```

A full run writes roughly **600 MB per minute of 720p video** — the three
sample clips produce 1.7 GB. See SETUP.md §7 before running a long video.

The `.mp4` files are input data, not source code, so they are not copied into
this folder — `--videos_dir` defaults to the parent folder:

```bash
python run_all.py --videos_dir /path/to/videos
```

Add a clip by appending it to `VIDEOS` in `run_all.py`. The key names the
output folder and labels the report; it selects nothing, so adding, renaming
or swapping an input cannot change how any other video is clustered.

Any stage also runs alone:

```bash
python stage01_extract_frames.py --video ../input1.mp4 --out output/input1
python stage05_cluster_onm.py --video_dir output/input1 --cc 8
python stage07_report.py --out output
```

To run the published algorithm with none of the additions:

```bash
python stage05_cluster_onm.py --video_dir output/input1 \
    --time_weight 0 --min_run 0 --cc 1 --sl_merge 0 --spike 0 --min_jump 0
```

---

## Output layout

```
output/
    video_info/
        input1_info.txt ... input3_info.txt   before/after tables per video
        all_videos_info.txt                   summary of every video
        onm_summary.txt                       SDCO N, clusters, A(C), D(C), T, CC, SL
    input1/
        01_original_frames/     frame_0000.jpg ...
        02_frames_5fps/         frame_0000.jpg ...
        03_gray_enhanced/       frame_0000.jpg ...
        04_feature_vectors/     frame_0000.npy ... + all_features.npy
        05_clusters_5fps/       cluster_0/ ... + segments.txt + labels.npy
        06_clusters_original/   cluster_0/ ... + segments.txt + labels.npy
        onm_report.txt          T, CC, SL, per-cluster IT/IS, A(C), D(C)
    input2/ input3/
```

---

## Results

```
Video   Frames  5fps  PCA dims  SDCO N  Clusters  A(C)%   D(C)%      T   CC     SL
input1     833   167        48      20         7  94.09    5.91  2.376    1  16.46
input2    1177   197        14      11        13 100.00    0.00  6.387    5  23.90
input3    1449   290        26      18        16  99.61    0.39  3.586    5  18.29
```

`SDCO N` and `Clusters` are different numbers by design: N is what the count
in stage 5 produces, and `Clusters` is what remains after the temporal
smoothing, which merges runs and renumbers each surviving run as its own
cluster. `onm_summary.txt` prints both.

A(C) is only meaningful alongside the SL that produced it — raising SL raises
A(C) mechanically — so the two are always quoted together.

---

## Known limitations

* **Clusters are time-contiguous intervals, not sets.** A camera angle used at
  0:05 and again at 0:40 gets two cluster ids, because `_renumber` gives every
  temporal run a fresh one. The published algorithm would put them in one
  cluster. See ALGORITHM.md §4.
* **Colour is discarded** in stage 3, so two shots that differ only in colour —
  same composition, same luminance, same texture — are indistinguishable to the
  pipeline.
* **Gradual transitions are invisible.** `drop_flat_boundaries` deliberately
  suppresses gradual change to kill lighting-drift false positives, which means
  a slow dissolve or fade is exactly what it discards. Hard cuts only.
* **N remains sensitive to T and CC.** ONM converts "how many clusters?" into
  "what are T and CC?" rather than answering it absolutely. That is a property
  of the published method.
* **The clustering does not iterate** — one assignment pass, one centroid
  update, faithful to the paper, so a poor initial centroid is never corrected.

