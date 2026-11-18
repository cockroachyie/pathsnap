"""
PathSnap feature extraction: candidate generation + (observation, candidate) features + labels.

Problem framing (from "Models and Training Guidelines"):
    For every GPS observation we generate a set of candidate road arcs nearby. Each
    (observation, candidate) pair is one row with a feature vector. A classifier scores each row;
    the HMM/Viterbi decoder later uses those scores as emission probabilities.

Inputs  : data_clean/ written by Data Cleaning/clean_dataset.py
Outputs : data_features/
            {partition}/{trip}.csv.gz                     features only (safe for the prediction pipeline)
            labels/{partition}/{trip}.csv.gz              row labels: idx, arc_id, label
            labels/{partition}/{trip}_points.csv.gz       per-observation reference arc (from route alignment)
            candidate_summary.csv                         rows, positives, candidate miss rate per trip
            feature_params.json

Labels: the .route file is an ordered arc list with no link to individual GPS points. Each point's
reference arc is derived with a monotonic alignment (dynamic programming): points are assigned to route
arcs in route order, minimising total point-to-arc distance. Points outside the route's label span (partial
reference routes) get label -1 and must be ignored in training and scoring.

All thresholds are fixed or chosen on the train partition only.

Usage:
    python extract_features.py            # defaults: ../../data_clean -> ../../data_features
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

EARTH_R = 6371008.8
CAND_RADIUS_M = 50.0     # candidate search radius around each GPS observation
MAX_CANDS = 20           # keep at most the 20 nearest arcs (two-way roads count twice; 12 truncated dense areas)
LABEL_MAX_DIST_M = 50.0  # an aligned reference arc farther than this is an alignment failure, not a label
DENSIFY_M = 10.0         # arc sampling step for the spatial index
HEADING_K = 2            # heading/speed from smoothed points i-K .. i+K (same segment)
HEADING_MIN_DISP_M = 4.0 # below this displacement the heading is treated as unknown
CONTEXT_K = 2            # neighbour-context distances use observations i-2..i+2

FEATURES = [
    # geometry
    "dist_m", "dist_rank", "dist_minus_best_m", "dist_ratio_best", "n_cands", "n_cands_20m",
    "proj_t", "dist_to_endpoint_m", "arc_length_m",
    # direction / motion
    "heading_valid", "heading_diff_deg", "cos_heading", "speed_mps",
    # context (neighbouring observations, no labels)
    "ctx_dist_mean_m", "ctx_dist_max_m",
    "same_as_prev_nearest", "connects_from_prev_nearest", "connects_to_next_nearest",
    # topology
    "is_two_way", "in_deg_u", "out_deg_v",
    # sampling
    "dt_prev", "dt_next", "speed_flag",
]


def local_xy(lon, lat, lon0, lat0):
    return np.c_[np.radians(lon - lon0) * EARTH_R * np.cos(np.radians(lat0)),
                 np.radians(lat - lat0) * EARTH_R]


def seg_dist(P, A, B):
    """Row-wise distance from P[k] to segment A[k]-B[k]; also returns the projection parameter t."""
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-9)
    t = np.clip(((P - A) * AB).sum(1) / L2, 0, 1)
    D = P - (A + t[:, None] * AB)
    return np.sqrt((D ** 2).sum(1)), t


# ---------------------------------------------------------------- candidates
def build_arc_index(Q, au, av, arcs, bbox):
    """KD-tree over points sampled every DENSIFY_M metres along the arcs that touch bbox."""
    (x0, y0), (x1, y1) = bbox
    pu, pv = Q[au[arcs]], Q[av[arcs]]
    lo, hi = np.minimum(pu, pv), np.maximum(pu, pv)
    near = (hi[:, 0] >= x0) & (lo[:, 0] <= x1) & (hi[:, 1] >= y0) & (lo[:, 1] <= y1)
    arcs, pu, pv = arcs[near], pu[near], pv[near]
    L = np.sqrt(((pv - pu) ** 2).sum(1))
    ns = np.maximum(np.ceil(L / DENSIFY_M).astype(int), 1) + 1
    owner = np.repeat(np.arange(len(arcs)), ns)
    frac = np.concatenate([np.linspace(0, 1, n) for n in ns])
    samples = pu[owner] + frac[:, None] * (pv[owner] - pu[owner])
    return cKDTree(samples), arcs[owner]


def generate_candidates(P, tree, sample_arc, Q, au, av):
    hits = tree.query_ball_point(P, CAND_RADIUS_M + DENSIFY_M / 2)
    lens = np.fromiter((len(h) for h in hits), int, len(hits))
    pt = np.repeat(np.arange(len(P)), lens)
    arc = sample_arc[np.concatenate([np.asarray(h, int) for h in hits])] if lens.sum() else np.array([], int)
    key = np.unique(pt.astype(np.int64) * (au.size + 1) + arc)
    pt, arc = key // (au.size + 1), key % (au.size + 1)
    d, t = seg_dist(P[pt], Q[au[arc]], Q[av[arc]])
    keep = d <= CAND_RADIUS_M
    c = pd.DataFrame({"p": pt[keep], "arc_id": arc[keep], "dist_m": d[keep], "proj_t": t[keep]})
    c = c.sort_values(["p", "dist_m"], kind="mergesort")
    c["dist_rank"] = c.groupby("p").cumcount()
    return c[c.dist_rank < MAX_CANDS].reset_index(drop=True)


# ---------------------------------------------------------------- labels (route alignment)
def align_to_route(P, Q, ru, rv):
    """Monotonic DP: assign each point a route position j(i), non-decreasing in i, minimising sum of distances."""
    A, B = Q[ru], Q[rv]
    m = len(ru)
    idx = np.arange(m)

    def row(p):
        return seg_dist(np.broadcast_to(p, (m, 2)), A, B)[0]

    cost = row(P[0])
    bp = np.zeros((len(P), m), np.int32)
    for i in range(1, len(P)):
        run = np.minimum.accumulate(cost)
        arg = np.maximum.accumulate(np.where(cost <= run, idx, 0))
        cost = row(P[i]) + run
        bp[i] = arg
    j = np.empty(len(P), int)
    j[-1] = int(np.argmin(cost))
    for i in range(len(P) - 1, 0, -1):
        j[i - 1] = bp[i, j[i]]
    d = seg_dist(P, A[j], B[j])[0]
    return j, d


# ---------------------------------------------------------------- per trip
def process_trip(clean_root, part, tid, out_root):
    trk = pd.read_csv(clean_root / part / tid / "track_clean.csv")
    net = np.load(clean_root / part / tid / "network_clean.npz")
    au, av, alen = net["arc_u"], net["arc_v"], net["arc_length_m"]
    canon = np.flatnonzero(net["arc_is_canonical"])

    obs = trk[~trk.is_spike & (trk.segment_id >= 0)].reset_index(drop=True)  # observations we predict for
    lon0, lat0 = obs.lon.median(), obs.lat.median()
    P = local_xy(obs.lon.to_numpy(), obs.lat.to_numpy(), lon0, lat0)          # raw positions for distances
    Ps = local_xy(obs.lon_s.to_numpy(), obs.lat_s.to_numpy(), lon0, lat0)     # smoothed, for heading/speed
    Q = local_xy(net["node_lon"], net["node_lat"], lon0, lat0)
    seg = obs.segment_id.to_numpy()
    t = obs.t.to_numpy()
    n = len(obs)

    # ---- candidates
    pad = CAND_RADIUS_M + DENSIFY_M
    tree, sample_arc = build_arc_index(Q, au, av, canon, (P.min(0) - pad, P.max(0) + pad))
    c = generate_candidates(P, tree, sample_arc, Q, au, av)
    p, a = c.p.to_numpy(), c.arc_id.to_numpy()

    # ---- per-observation motion
    lo_i = np.arange(n)
    seg_start = pd.Series(lo_i).groupby(seg).transform("min").to_numpy()
    seg_end = pd.Series(lo_i).groupby(seg).transform("max").to_numpy()
    i0 = np.maximum(lo_i - HEADING_K, seg_start)
    i1 = np.minimum(lo_i + HEADING_K, seg_end)
    mv = Ps[i1] - Ps[i0]
    disp = np.sqrt((mv ** 2).sum(1))
    speed = disp / np.maximum(t[i1] - t[i0], 1e-9)
    heading_valid = disp >= HEADING_MIN_DISP_M
    dt_prev = np.r_[np.nan, np.diff(t)]
    dt_next = np.r_[np.diff(t), np.nan]
    dt_prev[lo_i == seg_start] = -1  # -1 = first observation of a segment
    dt_next[lo_i == seg_end] = -1

    # ---- direction agreement
    av_vec = Q[av[a]] - Q[au[a]]
    av_len = np.maximum(np.sqrt((av_vec ** 2).sum(1)), 1e-9)
    gv = mv[p]
    cosang = (av_vec * gv).sum(1) / (av_len * np.maximum(disp[p], 1e-9))
    cosang = np.clip(cosang, -1, 1)
    hv = heading_valid[p]
    c["heading_valid"] = hv.astype(int)
    c["cos_heading"] = np.where(hv, cosang, 0.0)
    c["heading_diff_deg"] = np.where(hv, np.degrees(np.arccos(cosang)), 90.0)
    c["speed_mps"] = speed[p]

    # ---- geometry
    best = c.groupby("p").dist_m.transform("min").to_numpy()
    c["dist_minus_best_m"] = c.dist_m - best
    c["dist_ratio_best"] = (best + 1) / (c.dist_m + 1)
    c["n_cands"] = c.groupby("p").arc_id.transform("size")
    c["n_cands_20m"] = (c.dist_m <= 20).groupby(c.p).transform("sum").astype(int)
    c["arc_length_m"] = alen[a]
    c["dist_to_endpoint_m"] = np.minimum(c.proj_t, 1 - c.proj_t) * alen[a]

    # ---- context: how close are neighbouring observations to this arc?
    ctx = []
    for off in [o for o in range(-CONTEXT_K, CONTEXT_K + 1) if o]:
        q = p + off
        ok = (q >= 0) & (q < n)
        qq = np.clip(q, 0, n - 1)
        ok &= seg[qq] == seg[p]
        d, _ = seg_dist(P[qq], Q[au[a]], Q[av[a]])
        ctx.append(np.where(ok, d, np.nan))
    ctx = np.vstack(ctx)
    cnt = (~np.isnan(ctx)).sum(0)
    has = cnt > 0  # isolated observation (no neighbours in segment): fall back to its own distance
    c["ctx_dist_mean_m"] = np.where(has, np.nansum(ctx, 0) / np.maximum(cnt, 1), c.dist_m)
    c["ctx_dist_max_m"] = np.where(has, np.where(np.isnan(ctx), -np.inf, ctx).max(0), c.dist_m)

    # ---- continuity with neighbours' nearest candidate (label-free)
    nearest = np.full(n, -1)
    first = c.dist_rank.to_numpy() == 0
    nearest[p[first]] = a[first]
    prev_nn = np.where((p > 0) & (seg[np.maximum(p - 1, 0)] == seg[p]), nearest[np.maximum(p - 1, 0)], -1)
    next_nn = np.where((p < n - 1) & (seg[np.minimum(p + 1, n - 1)] == seg[p]), nearest[np.minimum(p + 1, n - 1)], -1)
    c["same_as_prev_nearest"] = (a == prev_nn).astype(int)
    c["connects_from_prev_nearest"] = ((prev_nn >= 0) & ((a == prev_nn) | (au[a] == av[np.maximum(prev_nn, 0)]))).astype(int)
    c["connects_to_next_nearest"] = ((next_nn >= 0) & ((a == next_nn) | (av[a] == au[np.maximum(next_nn, 0)]))).astype(int)

    # ---- topology
    in_deg = np.bincount(av[canon], minlength=len(Q))
    out_deg = np.bincount(au[canon], minlength=len(Q))
    pairs = set(zip(au[canon].tolist(), av[canon].tolist()))
    c["is_two_way"] = np.fromiter(((v, u) in pairs for u, v in zip(au[a], av[a])), bool, len(a)).astype(int)
    c["in_deg_u"] = in_deg[au[a]]
    c["out_deg_v"] = out_deg[av[a]]

    # ---- sampling
    c["dt_prev"] = dt_prev[p]
    c["dt_next"] = dt_next[p]
    c["speed_flag"] = obs.speed_flag.to_numpy()[p].astype(int)

    c.insert(0, "idx", obs.idx.to_numpy()[p])
    c.insert(0, "trip_id", tid)
    feats = c[["trip_id", "idx", "arc_id"] + FEATURES]

    (out_root / part).mkdir(parents=True, exist_ok=True)
    feats.to_csv(out_root / part / f"{tid}.csv.gz", index=False, float_format="%.4f")

    summary = {"trip_id": tid, "partition": part, "n_obs": n, "n_rows": len(feats),
               "obs_without_candidates": int(n - c.p.nunique()),
               "mean_cands": round(len(feats) / max(n, 1), 2)}

    # ---- labels
    lab_dir = clean_root / "labels" / part / tid
    if lab_dir.exists():
        route = pd.read_csv(lab_dir / "route_clean.csv")
        qc = pd.read_csv(lab_dir / "label_qc.csv").set_index("idx")
        in_span = qc.loc[obs.idx, "in_label_span"].to_numpy()
        ref_arc = np.full(n, -1)
        ref_dist = np.full(n, np.nan)
        ref_filled = np.zeros(n, bool)
        sel = np.flatnonzero(in_span)
        if len(sel):
            ru, rv = route.u.to_numpy(), route.v.to_numpy()
            j, d = align_to_route(P[sel], Q, ru, rv)
            ref_arc[sel] = route.canonical_arc_id.to_numpy()[j]
            ref_dist[sel] = d
            ref_filled[sel] = route.is_filled.to_numpy()[j]
        cand_set = set(zip(p.tolist(), a.tolist()))
        in_cands = np.array([(i, ref_arc[i]) in cand_set for i in range(n)])
        align_outlier = (ref_arc >= 0) & (ref_dist > LABEL_MAX_DIST_M)
        labelled = (ref_arc >= 0) & ~align_outlier
        pts = pd.DataFrame({"idx": obs.idx, "ref_arc": ref_arc, "ref_dist_m": ref_dist.round(2),
                            "ref_is_filled": ref_filled, "ref_in_candidates": in_cands & labelled,
                            "labelled": labelled})
        row_label = np.where(labelled[p], (a == ref_arc[p]).astype(int), -1)
        (out_root / "labels" / part).mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"idx": feats.idx, "arc_id": a, "label": row_label}).to_csv(
            out_root / "labels" / part / f"{tid}.csv.gz", index=False)
        pts.to_csv(out_root / "labels" / part / f"{tid}_points.csv.gz", index=False)
        summary.update({
            "n_labelled_obs": int(labelled.sum()),
            "n_alignment_outliers": int(align_outlier.sum()),
            "n_positive_rows": int((row_label == 1).sum()),
            "candidate_miss_rate": round(float(1 - in_cands[labelled].mean()) if labelled.any() else np.nan, 4),
            "ref_dist_p50_m": round(float(np.nanmedian(ref_dist)), 2) if labelled.any() else np.nan,
            "ref_dist_p95_m": round(float(np.nanpercentile(ref_dist, 95)), 2) if labelled.any() else np.nan,
        })
    return summary


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", type=Path, default=here.parents[1] / "data_clean")
    ap.add_argument("--out", type=Path, default=here.parents[1] / "data_features")
    ap.add_argument("--partitions", nargs="+", default=["train", "student_test"])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for part in args.partitions:
        for d in sorted(x for x in (args.clean / part).iterdir() if x.is_dir()):
            t0 = time.time()
            rows.append(process_trip(args.clean, part, d.name, args.out))
            print(f"{part}/{d.name}: {rows[-1]['n_rows']:,} rows, "
                  f"miss={rows[-1].get('candidate_miss_rate')}  ({time.time() - t0:.1f}s)", flush=True)

    pd.DataFrame(rows).to_csv(args.out / "candidate_summary.csv", index=False)
    params = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float))}
    params["FEATURES"] = FEATURES
    (args.out / "feature_params.json").write_text(json.dumps(params, indent=2))
    print(f"\nWrote {len(rows)} trips to {args.out}")


if __name__ == "__main__":
    main()
