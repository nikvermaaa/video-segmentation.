# Setup and Run

Linux, with conda. Verified against conda 25.5.1 and Python 3.13.

What the pipeline actually does is in [README.md](README.md); this is only how
to run it.

---

## Run it

```bash
# 1. install  (from inside this folder)
conda env create -f environment.yml
conda activate onm

# 2. check OpenCV can decode video - this is the one thing worth verifying
python -c "import cv2; print(cv2.VideoCapture('../input1.mp4').isOpened())"

# 3. run
python run_all.py

# 4. look at the result
cat output/video_info/onm_summary.txt
```

Step 2 must print `True`. Nothing else needs checking — if the imports were
broken it would fail there too.

---

## Where the videos go

`run_all.py` looks in its **parent folder** by default:

```
some-folder/
    input1.mp4  input2.mp4  input3.mp4      <- videos here
    final major project/                    <- this folder
```

Anywhere else works with `--videos_dir /path/to/videos`.

The filenames must match the `VIDEOS` list at the top of `run_all.py`. To use
your own footage, edit that list — one line per file. A listed-but-missing
clip is skipped with a message, not a crash.

---

## Options worth knowing

```bash
python run_all.py --only input2         # one video
python run_all.py --from_stage 5        # reuse frames + features, recluster
python run_all.py --keep                # do not wipe output/ first
```

`--from_stage 5` is the one you will use most: stages 1 and 4 (decoding, then
379 measurements per frame) are the slow ones, so changing a clustering
parameter and re-running everything wastes minutes. With it, seconds.

**Disk: about 600 MB per minute of 720p video.** The three sample clips
produce 1.7 GB, because every frame is written at full rate, again at 5 fps,
again enhanced, and once more into its cluster folder. A 10-minute video needs
roughly 6 GB. `python stage05_cluster_onm.py --video_dir output/input1
--no_copy` skips the cluster-folder copies if you only need the labels.

Runtime is roughly 45 s per 1000 original frames on a laptop CPU. No GPU, ~2 GB
RAM, single process.

---

## Expected output

```
Video     Frames   5fps  SDCO N  Clust    A(C)%    D(C)%         T    CC       SL
input1       833    167      20      7    94.09     5.91     2.376     1    16.46
input2      1449    290      18     16    99.61     0.39     3.586     5    18.29
input3      1440    288      21     19   100.00     0.00     4.546     5    21.80
```

Then the groupings themselves:

```
output/input1/05_clusters_5fps/cluster_0/    the frames of each cluster
output/input1/05_clusters_5fps/segments.txt  frame ranges, readable
output/input1/onm_report.txt                 T, CC, SL, per-cluster IT/IS
```

There are no random seeds anywhere, so the same videos and package versions
give identical labels every run. Small differences across OpenCV or
scikit-image versions are normal — JPEG encoding and HOG are not bit-identical
between releases. Large differences are not.

---

## If something breaks

| symptom | fix |
|---|---|
| step 2 prints `False` | wrong path, or a codec OpenCV lacks. `pip install --force-reinstall opencv-python-headless` — that wheel bundles FFmpeg. |
| `ImportError: libGL.so.1` | you have plain `opencv-python`. `pip uninstall -y opencv-python && pip install opencv-python-headless`. |
| `No module named 'cv2'` | wrong interpreter. `python -c "import sys; print(sys.executable)"` must point inside the `onm` env. |
| `all_features.npy missing - run stage 4 first` | you used `--from_stage 5` before that clip ever had features. Run `--from_stage 1` once. |
| conda solve hangs or conflicts | skip the YAML: `conda create -n onm python=3.13 && conda activate onm && pip install -r requirements.txt` |

---

## Other platforms

The same steps work on Windows and macOS unchanged — `opencv-python-headless`
is published for all three, and this project never opens a window. Without
conda, substitute:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Copying the project

The folder is self-contained; no module imports anything outside it. Two
things do not travel with it: the `.mp4` files, and `output/`, which is
regenerated — no need to copy gigabytes of frames.
