"""
PathSnap data cleaning.

Applies the fixes found in the train-partition discrepancy audit:

  INPUT SIDE (safe to use at prediction time)
    1. Relative time            t_rel = t - t0; dt kept per point (sampling is not always 1 Hz)
    2. Spike removal            single-point out-and-back GPS glitches (impossible speed in AND out, and far
                                off the line between neighbours). None exist in train; kept as a safeguard.
       Speed flag               steps faster than VMAX are timestamp hiccups (positions stay on the road),
                                so they are flagged, not removed and not used to split segments.
    3. Segmentation             new segment at time gaps > GAP_S
    4. Smoothing + thinning     lon_s/lat_s = rolling median; keep_thin marks smoothed points >= THIN_M apart,
                                for length/speed/heading features (raw 1 Hz jitter doubles track length)
    5. Network clean-up         merge nodes with identical coordinates, drop self-loops, map duplicate
                                (u, v) arcs to one canonical arc id, compute arc length in metres

  LABEL SIDE (evaluation / training targets only - never feed into the matcher)
    6. Canonical route          route arc ids mapped to canonical ids
    7. Route breaks             detected; filled with the shortest path for TRAINING use (flagged as filled)
    8. Label coverage QC        per-point distance to the reference route; trips where the route covers only
                                part of the track are flagged label_partial and excluded from tuning

Thresholds are fixed rules chosen from the train partition only. They are not fitted on student_test.
Raw files are never modified.

Usage:
    python clean_dataset.py                       # defaults: ../../Student folder -> ../../data_clean
    python clean_dataset.py --data DIR --out DIR
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

# ---------------------------------------------------------------- parameters (chosen from TRAIN only)
EARTH_R = 6371008.8
GAP_S = 60.0          # start a new segment when consecutive fixes are more than 60 s apart
VMAX = 70.0           # m/s (252 km/h): above any real road speed in train (Autobahn trips average 120-132 km/h)
SPIKE_DEV_M = 30.0    # a spike must also sit > 30 m off the line between its neighbours (GPS noise is ~5-15 m)
SMOOTH_WIN = 5        # rolling-median window (points) for lon_s / lat_s
THIN_M = 20.0         # keep_thin: min spacing on smoothed coords; median-5 + 20 m brings track length within ~3 % of route length
LABEL_RADIUS_M = 50.0 # a track point is "covered" by the reference route if within 50 m of it
LABEL_WINDOW = 61     # rolling window (points) to smooth coverage into a label span
PARTIAL_FRAC = 0.05   # trip flagged label_partial if > 5 % of points are not covered
FILL_LIMIT_M = 3000.0 # max shortest-path search distance when filling a route break
FILL_MAX_DETOUR = 3.0 # accept a fill only if path <= 3 x straight-line gap + FILL_SLACK_M
FILL_SLACK_M = 100.0


# ---------------------------------------------------------------- geometry helpers
def haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def local_xy(lon, lat, lon0, lat0):
    """Equirectangular projection in metres around the trip centre (error << GPS noise at trip scale)."""
    return np.c_[np.radians(lon - lon0) * EARTH_R * np.cos(np.radians(lat0)),
                 np.radians(lat - lat0) * EARTH_R]


def point_to_segments_min(P, A, B, chunk=2000):
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-9)
    out = np.empty(len(P))
    for s in range(0, len(P), chunk):
        p = P[s:s + chunk, None, :]
        t = np.clip(((p - A[None]) * AB[None]).sum(2) / L2[None], 0, 1)
        out[s:s + chunk] = np.sqrt(((p - (A[None] + t[..., None] * AB[None])) ** 2).sum(2)).min(1)
    return out


def point_line_dev(P, A, B):
    """Perpendicular distance of P from the infinite line A-B (falls back to |P-A| if A == B)."""
    AB = B - A
    L = np.sqrt((AB ** 2).sum(1))
    cross = np.abs(AB[:, 0] * (P[:, 1] - A[:, 1]) - AB[:, 1] * (P[:, 0] - A[:, 0]))
    return np.where(L > 1e-6, cross / np.maximum(L, 1e-6), np.sqrt(((P - A) ** 2).sum(1)))


# ---------------------------------------------------------------- 1-4: track
def clean_track(T):
    lon, lat, t = T[:, 0], T[:, 1], T[:, 2]
    n = len(T)
    xy = local_xy(lon, lat, np.median(lon), np.median(lat))

    # 2. spike removal: iterate so that two-point spikes are also caught
    is_spike = np.zeros(n, bool)
    for _ in range(3):
        idx = np.flatnonzero(~is_spike)
        if len(idx) < 3:
            break
        P, tt = xy[idx], t[idx]
        d = np.sqrt((np.diff(P, axis=0) ** 2).sum(1))
        v = d / np.maximum(np.diff(tt), 1e-9)
        v_in, v_out = v[:-1], v[1:]
        dev = point_line_dev(P[1:-1], P[:-2], P[2:])
        mid = (v_in > VMAX) & (v_out > VMAX) & (dev > SPIKE_DEV_M)
        new = idx[1:-1][mid]
        # endpoints: first/last fix with an impossible jump to its neighbour
        if v[0] > VMAX and d[0] > SPIKE_DEV_M:
            new = np.r_[idx[0], new]
        if v[-1] > VMAX and d[-1] > SPIKE_DEV_M:
            new = np.r_[new, idx[-1]]
        if len(new) == 0:
            break
        is_spike[new] = True

    # 3. segmentation on the spike-free sequence: time gaps only.
    #    Speed is NOT used to split: in train, every > VMAX step is a timestamp hiccup (late fix, then
    #    catch-up) with the positions still collinear along the road, so the geometry is fine.
    keep = np.flatnonzero(~is_spike)
    segment_id = np.full(n, -1)
    dt_k = np.diff(t[keep])
    step_k = np.sqrt((np.diff(xy[keep], axis=0) ** 2).sum(1))
    segment_id[keep] = np.r_[0, np.cumsum(dt_k > GAP_S)]
    speed_flag = np.zeros(n, bool)  # timing anomaly marker on the point that ends the step
    speed_flag[keep[1:]] = (step_k / np.maximum(dt_k, 1e-9) > VMAX) & (dt_k <= GAP_S)

    # 4a. smoothed coordinates: rolling median within each segment (removes ~5-10 m jitter)
    sm = pd.DataFrame({"lon": lon, "lat": lat, "seg": segment_id}).iloc[keep]
    sm = sm.groupby("seg")[["lon", "lat"]].transform(
        lambda s: s.rolling(SMOOTH_WIN, center=True, min_periods=1).median())
    lon_s, lat_s = np.full(n, np.nan), np.full(n, np.nan)
    lon_s[keep], lat_s[keep] = sm["lon"].to_numpy(), sm["lat"].to_numpy()
    xy_s = local_xy(lon_s, lat_s, np.median(lon), np.median(lat))

    # 4b. thinning flag on smoothed coordinates (greedy, restarted in every segment)
    keep_thin = np.zeros(n, bool)
    last_seg, last_xy = -1, None
    for i in keep:
        if segment_id[i] != last_seg or np.hypot(*(xy_s[i] - last_xy)) >= THIN_M:
            keep_thin[i] = True
            last_seg, last_xy = segment_id[i], xy_s[i]
    # always keep the last point of every segment
    seg_last = keep[np.r_[np.diff(segment_id[keep]) != 0, True]]
    keep_thin[seg_last] = True

    dt = np.r_[np.nan, np.diff(t)]
    df = pd.DataFrame({
        "idx": np.arange(n), "lon": lon, "lat": lat, "lon_s": lon_s, "lat_s": lat_s,
        "t": t, "t_rel": t - t[0], "dt": dt,
        "is_spike": is_spike, "speed_flag": speed_flag, "segment_id": segment_id, "keep_thin": keep_thin,
    })
    stats = {
        "n_pts": n, "n_spikes": int(is_spike.sum()), "n_speed_flags": int(speed_flag.sum()),
        "n_segments": int(segment_id.max() + 1),
        "max_gap_s": float(np.nanmax(dt)) if n > 1 else 0.0,
        "frac_dt1": float(np.mean(np.diff(t) == 1)) if n > 1 else 1.0,
        "t0": float(t[0]), "n_thin": int(keep_thin.sum()),
    }
    return df, xy, stats


# ---------------------------------------------------------------- 5: network
def clean_network(N, A):
    n_nodes = len(N)
    # merge nodes with identical coordinates -> lowest index
    key = pd.DataFrame(N, columns=["lon", "lat"])
    node_map = key.groupby(["lon", "lat"], sort=False).ngroup().to_numpy()
    first_of_group = pd.Series(np.arange(n_nodes)).groupby(node_map).transform("min").to_numpy()
    node_map = first_of_group

    u, v = node_map[A[:, 0]], node_map[A[:, 1]]
    self_loop = u == v
    # canonical arc id = first arc row with the same (u, v) after merging
    pairs = pd.DataFrame({"u": u, "v": v, "arc": np.arange(len(A))})
    first = pairs[~self_loop].groupby(["u", "v"])["arc"].transform("min")
    canonical = np.full(len(A), -1)
    canonical[first.index.to_numpy()] = first.to_numpy()
    is_canonical = canonical == np.arange(len(A))

    length_m = haversine_m(N[u, 0], N[u, 1], N[v, 0], N[v, 1])
    stats = {
        "n_nodes": n_nodes, "n_arcs": len(A),
        "nodes_merged": int((node_map != np.arange(n_nodes)).sum()),
        "self_loops_removed": int(self_loop.sum()),
        "duplicate_arcs_mapped": int((~self_loop & ~is_canonical).sum()),
        "arcs_gt2km": int((length_m > 2000).sum()),
    }
    net = {"node_lon": N[:, 0], "node_lat": N[:, 1], "node_map": node_map,
           "arc_u": u, "arc_v": v, "arc_length_m": length_m,
           "arc_canonical": canonical, "arc_is_canonical": is_canonical}
    return net, stats


# ---------------------------------------------------------------- 6-8: labels
def clean_route(route, net, track_xy, lon0, lat0):
    u, v, canon, L = net["arc_u"], net["arc_v"], net["arc_canonical"], net["arc_length_m"]
    keep = net["arc_is_canonical"]
    n_nodes = len(net["node_lon"])

    # graph of canonical arcs for gap filling
    G = csr_matrix((np.maximum(L[keep], 0.01), (u[keep], v[keep])), shape=(n_nodes, n_nodes))
    pair_to_arc = {(a, b): i for i, a, b in zip(np.flatnonzero(keep), u[keep], v[keep])}

    r = canon[route]
    r = r[r >= 0]  # a route entry that was a self-loop carries no movement
    rows, n_breaks, n_filled = [], 0, 0
    for k, arc in enumerate(r):
        if k > 0 and v[r[k - 1]] != u[arc]:
            n_breaks += 1
            src, dst = v[r[k - 1]], u[arc]
            dist, pred = dijkstra(G, indices=src, limit=FILL_LIMIT_M, return_predecessors=True)
            gap_m = haversine_m(net["node_lon"][src], net["node_lat"][src],
                                net["node_lon"][dst], net["node_lat"][dst])
            # reject detours: a 73 m break must not be "filled" with a 1 km loop around one-way streets
            if np.isfinite(dist[dst]) and dist[dst] <= FILL_MAX_DETOUR * gap_m + FILL_SLACK_M:
                path, node = [], dst
                while node != src:
                    path.append(node)
                    node = pred[node]
                path.append(src)
                path = path[::-1]
                for a, b in zip(path[:-1], path[1:]):
                    rows.append((pair_to_arc[(a, b)], True, False))
                n_filled += 1
                rows.append((int(arc), False, False))
            else:
                rows.append((int(arc), False, True))  # this arc follows a break that could not be filled
            continue
        rows.append((int(arc), False, False))
    route_df = pd.DataFrame(rows, columns=["canonical_arc_id", "is_filled", "unfilled_break_before"])
    route_df["u"] = u[route_df.canonical_arc_id]
    route_df["v"] = v[route_df.canonical_arc_id]
    route_df["length_m"] = L[route_df.canonical_arc_id]
    route_df.insert(0, "pos", np.arange(len(route_df)))

    # coverage QC against the ORIGINAL (unfilled) reference arcs
    Q = local_xy(net["node_lon"], net["node_lat"], lon0, lat0)
    d = point_to_segments_min(track_xy, Q[u[r]], Q[v[r]])
    covered = d <= LABEL_RADIUS_M
    span = pd.Series(covered.astype(float)).rolling(LABEL_WINDOW, center=True, min_periods=1).mean() >= 0.5
    qc = pd.DataFrame({"idx": np.arange(len(d)), "dist_to_route_m": d.round(2), "in_label_span": span.to_numpy()})

    uncovered = float((~covered).mean())
    stats = {
        "route_len_raw": int(len(route)), "route_breaks": n_breaks, "breaks_filled": n_filled,
        "breaks_unfilled": n_breaks - n_filled,
        "label_uncovered_frac": round(uncovered, 4),
        "label_partial": uncovered > PARTIAL_FRAC,
        "dist_to_route_p50_m": round(float(np.median(d)), 2),
        "dist_to_route_p95_m": round(float(np.percentile(d, 95)), 2),
    }
    return route_df, qc, stats


# ---------------------------------------------------------------- driver
def process_trip(trip_dir, part, out_root):
    tid = trip_dir.name
    T = np.loadtxt(trip_dir / f"{tid}.track", ndmin=2)
    N = np.loadtxt(trip_dir / f"{tid}.nodes", ndmin=2)
    A = np.loadtxt(trip_dir / f"{tid}.arcs", dtype=np.int64, ndmin=2)
    route_file = trip_dir / f"{tid}.route"

    track_df, track_xy, s_track = clean_track(T)
    net, s_net = clean_network(N, A)

    d_in = out_root / part / tid
    d_in.mkdir(parents=True, exist_ok=True)
    track_df.to_csv(d_in / "track_clean.csv", index=False)
    np.savez_compressed(d_in / "network_clean.npz", **net)

    row = {"trip_id": tid, "partition": part, **s_track, **s_net}
    if route_file.exists():
        route = np.loadtxt(route_file, dtype=np.int64, ndmin=1)
        lon0, lat0 = np.median(T[:, 0]), np.median(T[:, 1])
        route_df, qc, s_route = clean_route(route, net, track_xy, lon0, lat0)
        d_lab = out_root / "labels" / part / tid
        d_lab.mkdir(parents=True, exist_ok=True)
        route_df.to_csv(d_lab / "route_clean.csv", index=False)
        qc.to_csv(d_lab / "label_qc.csv", index=False)
        row.update(s_route)
        row["use_for_tuning"] = (part == "train") and not s_route["label_partial"]
    return row


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=here.parents[1] / "Student folder")
    ap.add_argument("--out", type=Path, default=here.parents[1] / "data_clean")
    ap.add_argument("--partitions", nargs="+", default=["train", "student_test"])
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for part in args.partitions:
        for trip_dir in sorted(p for p in (args.data / part).iterdir() if p.is_dir()):
            rows.append(process_trip(trip_dir, part, args.out))
            print(f"{part}/{trip_dir.name} done", flush=True)

    flags = pd.DataFrame(rows)
    flags.to_csv(args.out / "trip_flags.csv", index=False)
    params = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float))}
    (args.out / "cleaning_params.json").write_text(json.dumps(params, indent=2))
    print(f"\nWrote {len(flags)} trips to {args.out}")


if __name__ == "__main__":
    main()
