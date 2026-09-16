"""
STAGE 5 - CLUSTERING WITH ONM (Optimum N-Means)

Reads stage 4's feature matrix and decides which frames belong to the same
scene without being told how many scenes there are. N is a COUNT, not a
search: the number of distinct objects that repeat at least CC times IS the
number of clusters.

    SDCO        Eq (1)(2)     count repetitions, keep distinct centroids
    Clustering  Eq (3)-(6)    assign to nearest centroid, one mean update
    ECVM        Eq (7)-(10)   IT/IS per cluster, averaged to A(C) and D(C)

Sreedhar Kumar S and Madheswaran M, "An Improved Partitioned Clustering
Technique for Identifying Optimum Number of Dissimilar Groups in Multimedia
Dataset", European Journal of Scientific Research, 151(1), 2018, pp. 5-21.

The time weight, merge_short_runs, merge_similar_neighbours and
drop_flat_boundaries are ADDITIONS, not in the paper. Run with
--time_weight 0 --min_run 0 --cc 1 --sl_merge 0 --spike 0 --min_jump 0
to restore the published algorithm exactly.

    in   04_feature_vectors/all_features.npy, 02_frames_5fps/*.jpg
    out  05_clusters_5fps/  cluster folders, labels.npy, segments.txt,
                            cluster_members.txt
         onm_report.txt     T, CC, SL, N, per-cluster IT/IS, A(C), D(C)

    python stage05_cluster_onm.py --video_dir output/one --name one
"""

import os
import glob
import shutil
import argparse

import numpy as np


# ======================================================================
# TIME WEIGHT  (an ADDITION - not in the paper)
# ======================================================================
def add_time_feature(X, time_weight):
    """Append normalised frame position, scaled by the mean feature
    magnitude so the weight means the same thing at any dimensionality.
    Returns the augmented matrix and the index of the time column.

    Keep the weight below about 1, or the clock starts overruling the
    picture. 0 disables it and restores the paper."""
    if time_weight <= 0:
        return X, None
    T = len(X)
    t = np.linspace(0.0, 1.0, T).reshape(-1, 1)
    scale = time_weight * np.mean(np.linalg.norm(X, axis=1))
    return np.hstack([X, t * scale]), X.shape[1]


# ======================================================================
# SDCO STAGE  -  Eq (1) and Eq (2)
# ======================================================================
def choose_T(X, percentile):
    """Read T off the distribution of the differences the Eq (1) test
    actually sees, since PCA features have no fixed 0-255 scale.

    Must be given the FEATURE matrix only - the time column is on a
    different scale and would drag the percentile."""
    n = len(X)
    idx = np.arange(n)
    if n > 400:                                  # sample for large n
        idx = np.random.RandomState(0).choice(n, 400, replace=False)
    diffs = np.abs(X[idx][:, None, :] - X[idx][None, :, :])
    iu = np.triu_indices(len(idx), k=1)
    return float(np.percentile(diffs[iu[0], iu[1], :], percentile))


def similarity_matrix(X, T, tol, time_col=None):
    """Eq (1) inner test. Objects i and j are similar when at least
    (m - tol) of their m dimensions differ by less than T.

    A time column is held out of the tolerance and must match on its own,
    or one time mismatch is simply absorbed and the weight has no bite.

    tol=2 is the paper's literal rule, written for m of 4 or 9. Here m is
    19-48, so it demands 92-96% agreement; --tol 10 gives the paper's own
    3x3 proportion instead."""
    n, m = X.shape
    S = np.zeros((n, n), dtype=bool)
    for i in range(n):
        agree = np.abs(X - X[i]) < T                  # (n, m) booleans
        if time_col is None:
            S[i] = agree.sum(axis=1) >= (m - tol)
        else:
            feat = np.delete(agree, time_col, axis=1)
            S[i] = ((feat.sum(axis=1) >= (feat.shape[1] - tol))
                    & agree[:, time_col])
    np.fill_diagonal(S, False)                        # j != i in Eq (1)
    return S


def sdco(X, T, tol, time_col=None, cc=None, cc_pct=None, cc_cap=None):
    """Search Distinct Centroid Objects.

    Eq (1)  rr(x_i) = how many other objects are similar to x_i
    Eq (2)  Y = { x_i : rr_i >= CC }, and they must be DISTINCT

    CC is taken as a percentile of the observed rr, not a fraction of n:
    it is a threshold on rr, and video frames recur only a handful of
    times where the paper's image blocks recur hundreds.

    cc_cap KEEPS CC HONEST ABOUT SCENE LENGTH. A scene of L frames can only
    produce a centroid if CC < L, because a frame has at most L-1 similar
    partners inside its own shot. So CC is a minimum-scene-length filter -
    and so is min_run, which states the same thing openly. Left independent
    the two contradict each other silently, and the stricter one wins with
    no mention in any output: on a 60 s clip, CC=12 from the percentile
    forbade every scene shorter than 13 frames while min_run=10 declared 10
    frames acceptable, so two real shots (10 and 8 frames long, at cuts of
    0.91 and 1.23 SL) got no representative and were swallowed by their
    neighbour. Capping CC at min_run-1 makes the two agree by construction.

    The distinctness rule is not in Eq (2) as written, but is needed to
    reproduce the paper's numbers: candidates are taken in descending rr
    order and accepted only if not similar to one already accepted."""
    S = similarity_matrix(X, T, tol, time_col)
    rr = S.sum(axis=1)

    if cc is None:
        cc = max(1, int(np.percentile(rr, cc_pct)))
    if cc_cap is not None:
        cc = max(1, min(cc, int(cc_cap)))

    order = np.argsort(-rr, kind="stable")
    centroids = []
    for i in order:
        if rr[i] < cc:
            break                                     # rr is descending
        if not any(S[i, c] for c in centroids):
            centroids.append(int(i))
    return np.array(centroids, dtype=int), rr, cc


# ======================================================================
# CLUSTERING STAGE  -  Eq (3)(4)(5)(6)
# ======================================================================
def assign(X, Y):
    """Eq (3)(4)(5): every object joins its nearest centroid object.
    The centroids are REAL data objects, not computed means - one of the
    things that distinguishes ONM from k-means."""
    d = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2)
    return np.argmin(d, axis=1)


def update_centroids(X, labels, N):
    """Eq (6): centroid becomes the arithmetic mean of its members.
    ONE pass, not iterated to convergence. Empty clusters keep their seed
    position and are dropped later."""
    Y = np.zeros((N, X.shape[1]))
    for l in range(N):
        m = X[labels == l]
        Y[l] = m.mean(axis=0) if len(m) else np.nan
    return Y


# ======================================================================
# TEMPORAL SMOOTHING  (ADDITIONS - not in the paper)
# ======================================================================
def _runs(lab):
    b = [0] + list(np.where(np.diff(lab) != 0)[0] + 1) + [len(lab)]
    return [(b[i], b[i + 1]) for i in range(len(b) - 1)]


def _renumber(lab):
    """Give every temporal RUN a fresh id.

    This converts ONM's clusters from SETS into INTERVALS: the reported
    cluster count becomes the number of runs after smoothing, and a camera
    angle used twice no longer rejoins one cluster."""
    new = np.zeros(len(lab), dtype=int)
    k = 0
    for i in range(1, len(lab)):
        if lab[i] != lab[i - 1]:
            k += 1
        new[i] = k
    return new


def local_contrast(X, half=6):
    """Per-gap ratio: frame-to-frame jump / typical jump around it. A hard
    cut is a SPIKE; camera drift inside one shot is sustained moderate
    motion with no spike. The ratio is comparable across videos, absolute
    jump size is not."""
    d = np.linalg.norm(np.diff(X, axis=0), axis=1)
    out = np.zeros(len(d))
    for i in range(len(d)):
        lo, hi = max(0, i - half), min(len(d), i + half + 1)
        nb = np.concatenate([d[lo:i], d[i + 1:hi]])
        out[i] = d[i] / max(np.median(nb), 1e-9) if len(nb) else 1.0
    return d, out


def merge_short_runs(labels, X, min_len):
    """Absorb any temporal run shorter than min_len into the adjacent run
    whose mean feature vector is closer, repeating until none remain.

    ONM assigns every object independently, so nothing stops a frame
    mid-shot jumping to another centroid and back. Merging whole runs
    never moves a surviving boundary, because it only ever deletes them -
    which a mode filter does not manage.

    min_len is a physical statement: a scene lasts at least this long.
    --min_run 0 turns it off and restores the paper's behaviour."""
    if min_len <= 1:
        return labels
    lab = labels.copy()
    while True:
        rs = _runs(lab)
        if len(rs) < 2:
            break
        short = [i for i, (s, e) in enumerate(rs) if e - s < min_len]
        if not short:
            break
        i = min(short, key=lambda k: rs[k][1] - rs[k][0])   # shortest first
        s, e = rs[i]
        cand = ([rs[i - 1]] if i > 0 else []) + \
               ([rs[i + 1]] if i < len(rs) - 1 else [])
        me = X[s:e].mean(axis=0)
        best = min(cand,
                   key=lambda r: np.linalg.norm(X[r[0]:r[1]].mean(axis=0) - me))
        lab[s:e] = lab[best[0]]
        lab = _renumber(lab)
    return lab


def merge_similar_neighbours(labels, X, sl):
    """Merge the closest pair of ADJACENT runs while their centroids are
    nearer than sl, repeating until none are.

    This reuses the paper's own criterion: Eq (8) and (10) define SL as the
    similarity limit, so two adjacent clusters within SL of each other are
    not two clusters by that definition - SDCO produced two only because a
    slow pan inside one shot generated two distinct centroid objects."""
    if sl <= 0:
        return labels
    lab = labels.copy()
    while True:
        rs = _runs(lab)
        if len(rs) < 2:
            break
        cents = [X[s:e].mean(axis=0) for s, e in rs]
        d, i = min((np.linalg.norm(cents[k + 1] - cents[k]), k)
                   for k in range(len(rs) - 1))
        if d >= sl:
            break
        lab[rs[i + 1][0]:rs[i + 1][1]] = lab[rs[i][0]]
        lab = _renumber(lab)
    return lab


def drop_flat_boundaries(labels, X, min_contrast, min_jump=0.0):
    """Remove boundaries that are not a SPIKE in frame-to-frame distance,
    and boundaries whose jump is too SMALL in absolute terms.

    Centroid distance alone cannot separate the two populations, because a
    subject walking from silhouette into full light moves the centroid as
    far as a real cut does. What separates them is the SHAPE of the change:
    a cut is one large jump against quiet neighbours, a pan or a lighting
    change is gradual and unremarkable against its own neighbours.

    A ratio has no sense of scale, so it is blind in a QUIET shot - a slight
    camera wobble in a near-static scene clears it. The absolute floor,
    measured in SL, catches those. A boundary must pass BOTH tests; they
    fail differently and neither subsumes the other.

        --spike 0      disables the ratio test.
        --min_jump 0   disables the absolute floor.
    """
    if min_contrast <= 0 and min_jump <= 0:
        return labels
    d, contrast = local_contrast(X)
    lab = labels.copy()
    while True:
        rs = _runs(lab)
        if len(rs) < 2:
            break
        # score each boundary by how far it falls short of the test it fails
        # WORST, normalised so the two are comparable: below 1 is a failure
        # either way, and the weakest boundary overall goes first
        weak = []
        for i in range(len(rs) - 1):
            g = rs[i + 1][0] - 1                  # gap the boundary sits on
            score = np.inf
            if min_contrast > 0:
                score = min(score, contrast[g] / min_contrast)
            if min_jump > 0:
                score = min(score, d[g] / min_jump)
            if score < 1.0:
                weak.append((score, i))
        if not weak:
            break
        _, i = min(weak)
        lab[rs[i + 1][0]:rs[i + 1][1]] = lab[rs[i][0]]
        lab = _renumber(lab)
    return lab


# ======================================================================
# ECVM VALIDATION  -  Eq (7)(8)(9)(10)
# ======================================================================
def ecvm(X, labels, Y, SL):
    """Intra Thickness  IT(c_l) = % of members within  SL of the centroid
    Intra Separation IS(c_l) = % of members beyond SL of the centroid
    A(C) and D(C) are their means over the clusters, and sum to 100.
    Higher A(C) is better - members sit close to their centroid.

    Must be given the FEATURE matrix only; including the time column would
    let a cluster score badly merely for being long.

    A(C) can be raised arbitrarily by raising SL, so quote it with the SL
    that produced it - which onm_report.txt always does."""
    IT, IS, sizes = [], [], []
    for l in range(len(Y)):
        m = X[labels == l]
        if not len(m):
            continue
        dist = np.linalg.norm(m - Y[l], axis=1)
        near = float((dist < SL).sum()) / len(m) * 100.0
        IT.append(near)
        IS.append(100.0 - near)
        sizes.append(len(m))
    return (np.array(IT), np.array(IS), np.array(sizes),
            float(np.mean(IT)) if IT else 0.0,
            float(np.mean(IS)) if IS else 0.0)


# ======================================================================
# THE ALGORITHM, END TO END
# ======================================================================
def onm(X_raw, t_pct=T_PERCENTILE, cc_pct=CC_PERCENTILE, tol=TOL,
        time_weight=TIME_WEIGHT, sl_factor=SL_FACTOR, cc=None,
        min_run=MIN_RUN, sl_merge=SL_MERGE, spike=SPIKE_MIN,
        min_jump=MIN_JUMP, verbose=True):
    X, time_col = add_time_feature(X_raw, time_weight)

    T = choose_T(X_raw, t_pct)          # features only - see choose_T

    # min_run is the stated shortest scene, so a scene that long must be able
    # to earn a centroid: that requires CC < min_run. See sdco's cc_cap.
    cents, rr, cc = sdco(X, T, tol, time_col, cc=cc, cc_pct=cc_pct,
                         cc_cap=(min_run - 1) if min_run and min_run > 1
                         else None)
    N = len(cents)
    if N == 0:                                        # nothing repeats
        return None

    labels = assign(X, X[cents])                      # Eq (3)(4)(5)

    # drop centroids that attracted nothing, then renumber
    keep = [l for l in range(N) if (labels == l).any()]
    remap = {old: new for new, old in enumerate(keep)}
    labels = np.array([remap[l] for l in labels])
    empty = N - len(keep)

    SL = sl_factor * T * np.sqrt(X_raw.shape[1])
    labels = merge_short_runs(labels, X_raw, min_run)
    labels = merge_similar_neighbours(labels, X_raw, sl_merge * SL)
    labels = drop_flat_boundaries(labels, X_raw, spike, min_jump * SL)
    keep = list(range(int(labels.max()) + 1))

    # Eq (6) and the ECVM validation both live in FEATURE space
    Y = update_centroids(X_raw, labels, len(keep))
    IT, IS, sizes, A, D = ecvm(X_raw, labels, Y, SL)

    if verbose:
        print(f"      T={T:.3f} (p{t_pct:g})  CC={cc}  tol=m-{tol}  "
              f"SL={SL:.3f}")
        print(f"      rr: min {rr.min()} median {np.median(rr):.0f} "
              f"max {rr.max()}")
        print(f"      N centroid objects = {N}"
              + (f", {empty} attracted no members -> {len(keep)} clusters"
                 if empty else f" -> {len(keep)} clusters"))
        print(f"      A(C) intra association = {A:.2f}%   "
              f"D(C) intra divergence = {D:.2f}%")

    return {"labels": labels, "centroid_idx": cents, "rr": rr, "T": T,
            "CC": cc, "SL": SL, "N_centroids": N, "N_clusters": len(keep),
            "empty": empty, "IT": IT, "IS": IS, "sizes": sizes,
            "A": A, "D": D}


# ======================================================================
# OUTPUT
# ======================================================================
def export_clusters(frame_paths, labels, out_dir):
    """Write one folder per cluster, plus labels.npy and segments.txt.

    stage06_map_to_original.py carries the same function, because each stage
    file is meant to run on its own. Change the format in both."""
    os.makedirs(out_dir, exist_ok=True)

    # clear a previous run, or frames from an older partition survive
    for d in glob.glob(os.path.join(out_dir, "cluster_*")):
        if os.path.isdir(d) and os.path.basename(d)[8:].isdigit():
            shutil.rmtree(d)

    for p, lab in zip(frame_paths, labels):
        d = os.path.join(out_dir, f"cluster_{lab}")
        os.makedirs(d, exist_ok=True)
        shutil.copy(p, os.path.join(d, os.path.basename(p)))

    with open(os.path.join(out_dir, "segments.txt"), "w") as f:
        start, cur = 0, labels[0]
        for i in range(1, len(labels) + 1):
            if i == len(labels) or labels[i] != cur:
                f.write(f"frames {start:6d} - {i-1:6d}  ->  cluster {cur}\n")
                if i < len(labels):
                    start, cur = i, labels[i]

    np.save(os.path.join(out_dir, "labels.npy"), labels)


def write_members(labels, path):
    """List membership by frame index. Without smoothing an ONM cluster is a
    SET of similar frames wherever they occur, so a range is not enough."""
    with open(path, "w") as f:
        for l in range(int(labels.max()) + 1):
            idx = np.where(labels == l)[0]
            f.write(f"cluster {l:3d}  ({len(idx):4d} frames)  "
                    f"{', '.join(str(i) for i in idx)}\n")


def write_report(name, res, n_frames, path):
    with open(path, "w") as f:
        f.write(f"ONM RESULT - {name}\n")
        f.write("=" * 70 + "\n\n")
        f.write("PARAMETERS\n" + "-" * 70 + "\n")
        f.write(f"  Objects (frames)      : {n_frames}\n")
        f.write(f"  Similarity T          : {res['T']:.4f}\n")
        f.write(f"  Control centroid CC   : {res['CC']}\n")
        f.write(f"  Similarity limit SL   : {res['SL']:.4f}\n\n")
        f.write("SDCO STAGE\n" + "-" * 70 + "\n")
        f.write(f"  Repetition rr: min {res['rr'].min()}, "
                f"median {np.median(res['rr']):.0f}, "
                f"max {res['rr'].max()}\n")
        f.write(f"  Distinct centroid objects (N) : {res['N_centroids']}\n")
        f.write(f"  Clusters with members         : {res['N_clusters']}\n")
        if res["empty"]:
            f.write(f"  Centroids attracting nothing  : {res['empty']}\n")
        f.write("\nECVM VALIDATION\n" + "-" * 70 + "\n")
        f.write(f"  A(C) Intra Association : {res['A']:.2f} %\n")
        f.write(f"  D(C) Intra Divergence  : {res['D']:.2f} %\n\n")
        f.write(f"  {'cluster':>8} {'size':>6} {'IT %':>8} {'IS %':>8}\n")
        f.write("  " + "-" * 32 + "\n")
        for l, (it, is_, sz) in enumerate(zip(res["IT"], res["IS"],
                                              res["sizes"])):
            f.write(f"  {l:8d} {sz:6d} {it:8.2f} {is_:8.2f}\n")


def resolve_params(args=None):
    """Precedence: explicit CLI flag > module default. Nothing else.

    There is deliberately no per-video table. One previously keyed fitted
    parameters off the video's name, which was wrong twice over: it made
    every reported figure a fit rather than a prediction, and it silently
    coupled the result to a FILENAME, so renaming or replacing an input
    handed it another clip's tuning without a word of warning."""
    base = {"t_pct": T_PERCENTILE, "cc_pct": CC_PERCENTILE, "cc": None,
            "time_weight": TIME_WEIGHT, "min_run": MIN_RUN,
            "sl_merge": SL_MERGE, "spike": SPIKE_MIN,
            "min_jump": MIN_JUMP}
    params = {}
    for key, default in base.items():
        cli = getattr(args, key, None) if args is not None else None
        params[key] = default if cli is None else cli
    return params


def cluster_video(video_dir, name, params, tol=TOL, sl_factor=SL_FACTOR,
                  copy_frames=True):
    """Read stage 4's matrix, cluster it, write everything out.
    Returns the result dict, or None if nothing repeated CC times."""
    feat = os.path.join(video_dir, "04_feature_vectors", "all_features.npy")
    if not os.path.isfile(feat):
        raise SystemExit(f"{feat} missing - run stage 4 first")
    X = np.load(feat)

    res = onm(X, tol=tol, sl_factor=sl_factor, verbose=False, **params)
    if res is None:
        print("  [5] no object repeated CC times - nothing to cluster")
        return None

    print(f"  [5] ONM: T={res['T']:.3f} CC={res['CC']} SL={res['SL']:.2f}"
          f" -> {res['N_centroids']} centroid objects"
          f" -> {res['N_clusters']} clusters")
    print(f"      A(C) intra association {res['A']:.2f}%   "
          f"D(C) intra divergence {res['D']:.2f}%")

    d5 = os.path.join(video_dir, "05_clusters_5fps")
    frames5 = sorted(glob.glob(os.path.join(video_dir, "02_frames_5fps",
                                            "*.jpg")))
    if copy_frames and len(frames5) == len(res["labels"]):
        export_clusters(frames5, res["labels"], d5)
    else:
        # still emit the machine-readable outputs even with no frames
        os.makedirs(d5, exist_ok=True)
        np.save(os.path.join(d5, "labels.npy"), res["labels"])
    write_members(res["labels"], os.path.join(d5, "cluster_members.txt"))
    write_report(name, res, len(X), os.path.join(video_dir, "onm_report.txt"))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--video_dir", required=True,
                    help="per-video folder holding 04_feature_vectors/")
    ap.add_argument("--name", default=None,
                    help="label used in onm_report.txt; defaults to the "
                         "folder name. It selects nothing and changing it "
                         "cannot change the clustering.")
    # these default to None so an explicitly passed flag can be told apart
    # from an unset one, which is what lets the module default stand
    ap.add_argument("--t_pct", type=float, default=None)
    ap.add_argument("--cc_pct", type=float, default=None,
                    help="CC as a percentile of the observed rr")
    ap.add_argument("--cc", type=int, default=None,
                    help="absolute CC, overrides --cc_pct (the paper's form)")
    ap.add_argument("--tol", type=int, default=TOL,
                    help="Eq (1)'s m-2; see similarity_matrix before changing")
    ap.add_argument("--time_weight", type=float, default=None,
                    help="0 restores the paper exactly")
    ap.add_argument("--sl_factor", type=float, default=SL_FACTOR)
    ap.add_argument("--min_run", type=int, default=None,
                    help="shortest allowed scene in frames; 0 restores "
                         "the paper exactly")
    ap.add_argument("--sl_merge", type=float, default=None,
                    help="merge adjacent clusters closer than this * SL; "
                         "0 disables")
    ap.add_argument("--spike", type=float, default=None,
                    help="boundary must be this x the local median jump; "
                         "0 disables")
    ap.add_argument("--min_jump", type=float, default=None,
                    help="boundary must also jump this x SL in absolute "
                         "distance; 0 disables")
    ap.add_argument("--no_copy", action="store_true",
                    help="write labels and listings but do not copy frames")
    args = ap.parse_args()

    name = args.name or os.path.basename(os.path.normpath(args.video_dir))
    params = resolve_params(args)

    res = cluster_video(args.video_dir, name, params, tol=args.tol,
                        sl_factor=args.sl_factor,
                        copy_frames=not args.no_copy)
    if res is None:
        raise SystemExit(1)
    print(f"      -> {os.path.abspath(os.path.join(args.video_dir, '05_clusters_5fps'))}")


if __name__ == "__main__":
    main()
