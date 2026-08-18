# The ONM Algorithm

Covers **stage 5** — `stage05_cluster_onm.py`, the file that decides how many
scenes a video has and which frames belong to each.

> Sreedhar Kumar S and Madheswaran M, *"An Improved Partitioned Clustering
> Technique for Identifying Optimum Number of Dissimilar Groups in Multimedia
> Dataset"*, European Journal of Scientific Research, Vol. 151 No. 1,
> December 2018, pp. 5–21.

---

## 1. The central idea

**N is a count, not a search result.**

k-means makes you supply `k`. Methods that "find the best k" — elbow,
silhouette, gap statistic, X-means — answer by trying several values and
scoring them. ONM never evaluates a candidate N. It asks:

> How many **distinct** objects in this dataset **repeat** at least CC times?

Count them. Those recurring objects **are** the centroids, and how many there
are **is** the number of clusters.

---

## 2. The three stages

```
Input : the object set, control centroid threshold CC
Output: clusters {c_1 ... c_N}

  Stage 1  SDCO        find the distinct centroid objects  -> N
  Stage 2  Clustering  assign objects, update centroids
  Stage 3  ECVM        validate with Intra Thickness / Separation
```

Complexity per the paper: SDCO is `O(n - N)`, clustering is `O(nN)`, overall
`O((n - N) + nN)`.

### Stage 1 — SDCO (Search Distinct Centroid Objects)

**Equation (1) — rate of repetition.** For a pair of objects i and j (j ≠ i):

```
per dimension f:   1 if |x_if - x_jf| < T
                   0 otherwise

objects similar:   1 if  SUM_f (that indicator)  >=  m - 2
                   0 otherwise

rr(x_i) = the number of j for which the pair is similar
```

Two objects count as "the same" when all but at most **two** of their m
dimensions agree to within T, the similarity threshold.

**Equation (2) — distinct centroid objects.**

```
Y = { x_i : rr_i >= CC },   N = |Y|
```

CC is the *control centroid threshold*, and the paper is explicit that it is
the main lever: it "is acted as a major key factor … it could directly affect
the performance".

`sdco()` implements Eq (2) with an added **distinctness rule**: candidates are
taken in descending rr order and one is accepted only if it is not similar to
a centroid already accepted. Eq (2) as written selects every object whose rr
clears CC, which for a 16384-object image returns thousands; the paper reports
N=58 for Lena, so the distinctness rule is needed to reproduce its numbers.

### Stage 2 — Clustering

Equations (3), (4), (5) — assignment:

```
d(x_i, y_l) = sqrt( SUM_f (x_if - y_lf)^2 )
c_l         = argmin over l of d(x_i, y_l)
```

Every object joins the nearest centroid **object**. At this point the
centroids are *real data objects*, not computed means — one of the things that
distinguishes ONM from k-means. It is also why the result is deterministic:
there is no random initialisation anywhere.

Equation (6) — centroid update: `y'_l` is the arithmetic mean of the members
of cluster l.

**It does not iterate.** The published algorithm performs ONE assignment pass
followed by ONE centroid update — steps 4–11 of its pseudocode are the nested
assignment loop, and there is no outer loop back to the start. This project
follows that. The mean update therefore feeds only the validation below, never
a second round of assignment.

### Stage 3 — ECVM validation

*Effective Cluster Validation Measure.* SL is the similarity limit.

```
Eq (8)   IT(c_l) = % of members with |c_lj - y'_l| < SL
Eq (7)   A(C)    = mean of IT over the N clusters     "Intra Association"

Eq (10)  IS(c_l) = % of members with |c_lj - y'_l| > SL
Eq (9)   D(C)    = mean of IS over the N clusters     "Intra Divergence"
```

IT and IS are complements, so `IT + IS = 100` for every cluster and
`A(C) + D(C) = 100`. Higher A(C) is better — members sit close to their
centroid. **A(C) is always quoted with the SL that produced it**, since raising
SL raises A(C) mechanically; the paper leaves SL to the user and never states
the value it used. Here `SL = SL_FACTOR × T × sqrt(dims)`, so validation and
clustering share one scale.

---

## 3. Applying it to video

The paper segments **one still image**. This project partitions **a video**.

### An object is a frame, not a pixel block

The algorithm is untouched; only what an object *is* changes.

| | paper | this project |
|---|---|---|
| what an object is | a 2×2 or 3×3 pixel block | one video frame |
| n (objects) | 16384 for a 256×256 image | 167 – 290 |
| m (dimensions) | 4 or 9 | 23 – 48 (PCA, varies per video) |
| the dataset | one image | one video |

As published, ONM clusters are **not time-contiguous**: a cluster is a *set*
of similar objects wherever they occur, so a camera angle used at 0:05 and
again at 0:40 lands in one cluster. Section 4 changes that.

### T is a percentile

The paper's T compares raw pixels on a fixed 0–255 scale, so a constant is
meaningful. PCA features have no fixed scale and the dimensionality differs
per video, so `choose_T()` sets T to a percentile (default p70) of the
per-dimension absolute differences actually present in the data. Across the
three clips T lands between 2.376 and 4.546.

T is computed on the **feature matrix only**. The time column of section 4 is
on a different scale by construction and would drag the percentile.

### CC is a percentile of the observed rr

CC is a threshold on rr, not on n. In the paper's images a flat block — sky, a
wall — recurs hundreds of times, so rr values are large and CC=75 is a genuine
filter. Video frames recur only a handful of times:

| video | rr min | rr median | rr max | CC from p35 | CC after the cap |
|-------|--------|-----------|--------|-------------|------------------|
| input1 | 0 | 1  | 19 | 1  | 1 |
| input2 | 0 | 15 | 26 | 14 | 5 |
| input3 | 2 | 14 | 42 | 12 | 5 |

So CC is taken as a percentile of the observed rr — the same intent on the
scale that exists. `--cc <int>` accepts the paper's absolute form.

### CC is capped at `min_run − 1`

This is the one place the two controls are forced to agree, and it fixes a
real defect.

A scene of L frames can only produce a centroid if **CC < L**, because a frame
has at most L−1 similar partners inside its own shot. So CC is a hidden
minimum-scene-length filter — and `min_run` is an openly declared one. Left
independent they contradict each other, and the stricter wins with no mention
in any output. Measured on input3 with the percentile unconstrained:

```
frames 0-9     L=10 -> at most  9 partners; max rr seen = 10   below CC=12, no centroid
frames 10-34   L=25 -> at most 24 partners; max rr seen = 24   gets a centroid
frames 35-42   L= 8 -> at most  7 partners; max rr seen =  7   below CC=12, no centroid
```

`CC = 12` forbade every scene shorter than 13 frames while `min_run = 10`
declared 10 frames acceptable. Two real shots — at cuts jumping 0.91 SL and
1.23 SL, against 0.06 and 0.38 *within* those shots — got no representative at
all and were swallowed by the long neighbour between them, which is why one
cluster spanned frames 0–42.

`sdco(..., cc_cap=min_run-1)` makes the two agree by construction. **This cap
is not in the paper**; the paper has no `min_run` and so no conflict to
resolve.

### The m−2 tolerance in high dimensions

In the paper m is 4 or 9, so "all but 2 must agree" means 50% or 78% of
dimensions. Here m is 23–48, so the same literal rule demands **91–96%
agreement**:

| video | m | what "all but 2" demands |
|-------|---|--------------------------|
| input3 | 23 | 91.3% |
| input2 | 26 | 92.3% |
| input1 | 48 | 95.8% |

`tol=2` is the default because that is what Eq (1) literally says; `--tol 10`
gives the paper's own 3×3 proportion instead. The choice is exposed rather
than silently made.

---

## 4. The additions

Four steps that are not in the paper, plus the CC cap of section 3. Each is
flagged at its definition in the source, and each can be switched off.

### Time weight

A normalised frame-position column is appended, scaled by the mean feature
magnitude so the weight means the same thing at any dimensionality. It enters
the algorithm twice:

* **Eq (4) distance** — pulls frames toward temporally near centroids.
* **Eq (1) similarity** — the time column must match within T and is
  deliberately **not** covered by the m−2 tolerance, or a lone time mismatch
  would simply be absorbed by the tolerance and the weight would do nothing.

Keep it small; above about 1 the clock starts overruling the picture and the
video is cut into near-uniform temporal slabs.

### `merge_short_runs`

ONM assigns every object independently, so nothing stops a frame mid-shot
jumping to another centroid and back. This absorbs any temporal run shorter
than `min_run` into the adjacent run whose mean feature vector is closer,
repeating until none remain. Merging whole runs never moves a surviving
boundary, because it only ever deletes them — which a mode filter does not
manage.

`min_run` is a physical statement — a scene lasts at least this long. At 5 fps
the default of 6 frames is 1.2 seconds. **It is now the only
minimum-scene-length control**, since CC follows from it.

### `merge_similar_neighbours`

Merges the closest pair of adjacent runs while their centroids are nearer than
`sl_merge × SL`, repeating until none are. This reuses the paper's own
criterion: Eq (8) and (10) define SL as the similarity limit, so two adjacent
clusters within SL of each other are not two clusters by that definition —
SDCO produced two only because a slow pan inside one shot generated two
distinct centroid objects.

### `drop_flat_boundaries`

Removes boundaries that fail either of two tests.

**The spike test** (`--spike`). A cut is one large jump against quiet
neighbours; a pan or a lighting change is gradual and unremarkable against its
own neighbours. `local_contrast()` measures each gap as
`frame-to-frame jump / median jump in a ±6 frame window`, and a boundary must
reach `SPIKE_MIN` times the local median. Centroid distance alone cannot do
this: a subject walking from silhouette into full light moves the centroid as
far as a real cut does.

**The absolute floor** (`--min_jump`). A ratio has no sense of scale, so it
goes blind in a quiet shot — in a near-static scene a slight camera wobble can
read as twice the local median. The floor requires the jump to be at least
`MIN_JUMP × SL` in absolute distance. SL is the right unit because Eq (8)
calls two objects closer than SL the same object, so a "boundary" whose two
sides are the same object by that definition is not a boundary.

A boundary must pass **both**. They fail differently — the ratio catches a
large gradual drift, the floor catches a small sharp wobble — so neither
subsumes the other.

### `_renumber` — sets become intervals

Every merge is followed by `_renumber()`, which gives each temporal run a
fresh id. This converts ONM's clusters from **sets** into **intervals**: the
reported cluster count becomes the number of runs after smoothing, and a
camera angle used twice no longer rejoins one cluster.

This is a real departure from the paper, and it is the largest one. It is
applied because the deliverable is a scene partition of a timeline, which is
an interval partition by definition. Stage 6 does the same thing again when it
re-lays labels over the snapped boundaries. Because of it the final cluster
count is not SDCO's N, so `onm_summary.txt` prints both columns side by side.

### Restoring the published algorithm

```bash
python stage05_cluster_onm.py --video_dir output/input1 \
    --time_weight 0 --min_run 0 --cc 1 --sl_merge 0 --spike 0 --min_jump 0
```

`--min_run 0` also removes the CC cap, since there is no stated scene length
left to enforce.

---

## 5. End to end

`onm()` runs the whole chain:

```
X_raw                          (frames x dims) from stage 4
  -> add_time_feature          append the scaled position column
  -> choose_T                  T from the feature matrix only
  -> sdco                      Eq (1)(2), CC capped at min_run-1  -> N centroids
  -> assign                    Eq (3)(4)(5)
  -> drop empty centroids, renumber
  -> merge_short_runs          min_run
  -> merge_similar_neighbours  sl_merge x SL
  -> drop_flat_boundaries      spike ratio AND min_jump x SL
  -> update_centroids          Eq (6), in feature space
  -> ecvm                      Eq (7)-(10), in feature space
```

Eq (6) and ECVM both run on `X_raw`, not the time-augmented matrix: the paper
validates feature similarity, and including the time column would let a
cluster score badly merely for being long.

Then `cluster_video()` writes the outputs:

```
05_clusters_5fps/cluster_0/ ...       the frames of each cluster
05_clusters_5fps/labels.npy           one cluster id per frame
05_clusters_5fps/segments.txt         frame ranges, human readable
05_clusters_5fps/cluster_members.txt  frame indices per cluster
onm_report.txt                        T, CC, SL, N, per-cluster IT/IS, A(C), D(C)
```

---

## 6. Parameters

| name | default | meaning |
|------|---------|---------|
| `T_PERCENTILE` | 70.0 | Percentile setting the similarity threshold T. |
| `CC_PERCENTILE` | 35.0 | CC as a percentile of the observed rr, then capped at `MIN_RUN-1`. `--cc <int>` gives the paper's absolute form. |
| `TOL` | 2 | Equation (1)'s m−2. |
| `TIME_WEIGHT` | 0.5 | Temporal pull. 0 = the paper exactly. |
| `SL_FACTOR` | 1.0 | `SL = SL_FACTOR × T × sqrt(dims)`. |
| `MIN_RUN` | 6 | Shortest allowed scene, in frames. 1.2 s at 5 fps. Also caps CC. 0 = the paper exactly. |
| `SL_MERGE` | 0.8 | Merge adjacent clusters closer than this × SL. 0 disables. |
| `SPIKE_MIN` | 1.5 | Boundary must be this × the local median jump. 0 disables. |
| `MIN_JUMP` | 0.55 | Boundary must **also** jump this × SL in absolute distance. 0 disables. |

Precedence is **CLI flag > module default**, resolved by `resolve_params()`.
There is no per-video table — see README.md for why it was removed.

---

## 7. Results

```
Video   Frames  5fps  PCA dims  SDCO N  Clusters  A(C)%   D(C)%      T   CC     SL
input1     833   167        48      20         7  94.09    5.91  2.376    1  16.46
input2    1449   290        26      18        16  99.61    0.39  3.586    5  18.29
input3    1440   288        23      21        19 100.00    0.00  4.546    5  21.80
```

Worked example, input1:

```
833 original frames @ 25 fps, 1280x720
  -> stride 5, so 167 frames at 5 fps
  -> 379 raw features per frame, PCA to 48 dims (95% variance)
  -> T = 2.376 (p70 of per-dimension differences)
  -> rr: min 0, median 1, max 19  ->  CC = 1  (p35 gives 1; the cap of 5 is not binding)
  -> SDCO: 20 distinct centroid objects
  -> assign, then merge_short_runs / merge_similar_neighbours / drop_flat
  -> 7 clusters
  -> SL = 16.46, A(C) = 94.09%, D(C) = 5.91%
  -> expanded to 833 frames; 4 boundaries snapped, max shift 4
```

Note how the cap binds differently per clip: on input1 the percentile already
gives CC=1, so the cap changes nothing; on input2 and input3 it pulls CC from
14 and 12 down to 5, which is what lets their short shots earn a centroid.
