"""HMM / Viterbi map matching on top of per-candidate classifier probabilities.

Hidden state at observation t  = one candidate arc of that observation (+ its projection point on the arc).
Emission                       = classifier probability of that candidate.
Transition t -> t+1            = how well the network driving distance between the two candidate positions
                                 matches the straight-line distance between the two GPS fixes
                                 (Newson & Krumm 2009): log P = -|d_route - d_gc| / beta, minus a U-turn penalty.
Infeasible transition          = no forward network path within MAX_ROUTE_M -> probability 0.
Decoding                       = Viterbi in log space, restarted at time-gap segment boundaries and wherever no
                                 transition is feasible (a "break").
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

EARTH_R = 6371008.8
MAX_ROUTE_M = 2000.0   # longest network path searched between consecutive observations


def haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class TripGraph:
    """Directed road graph of one trip (canonical arcs only) with cached node-to-node shortest paths."""

    def __init__(self, net):
        self.u, self.v, self.L = net["arc_u"], net["arc_v"], net["arc_length_m"]
        keep = net["arc_is_canonical"]
        n = len(net["node_lon"])
        self.G = csr_matrix((np.maximum(self.L[keep], 0.01), (self.u[keep], self.v[keep])), shape=(n, n))
        self.pair_to_arc = {(a, b): i for i, a, b in zip(np.flatnonzero(keep), self.u[keep], self.v[keep])}
        self.cache = {}
        self.searched = set()

    def node_dist(self, pairs):
        """Shortest distance for (src, dst) node pairs; inf if farther than MAX_ROUTE_M."""
        need = sorted({s for s, _ in pairs} - self.searched)
        for k in range(0, len(need), 64):
            src = need[k:k + 64]
            D = dijkstra(self.G, indices=src, limit=MAX_ROUTE_M)
            for row, s in enumerate(src):
                reach = np.flatnonzero(np.isfinite(D[row]))
                self.cache.update({(s, int(d)): float(D[row, d]) for d in reach})
            self.searched.update(src)
        return np.array([0.0 if s == d else self.cache.get((s, d), np.inf) for s, d in pairs])

    def path_arcs(self, a, b):
        """Arcs strictly between arc a and arc b on the shortest path (for route reconstruction)."""
        s, d = self.v[a], self.u[b]
        if s == d:
            return []
        _, pred = dijkstra(self.G, indices=s, limit=MAX_ROUTE_M, return_predecessors=True)
        if pred[d] < 0:
            return None
        nodes, x = [d], d
        while x != s:
            x = pred[x]
            nodes.append(x)
        nodes = nodes[::-1]
        return [self.pair_to_arc[(p, q)] for p, q in zip(nodes[:-1], nodes[1:])]


def build_steps(cands, obs, graph):
    """Pre-compute everything about transitions that does not depend on tunable parameters.

    cands: rows of one trip with idx, arc_id, proj_t, p (classifier probability)
    obs:   observations of the trip in time order with idx, lon, lat, t, segment_id
    Returns per decodable observation: candidate arcs + log emissions, and per consecutive pair:
    the route-minus-straight-line distance matrix, a U-turn mask and the gap flag.
    """
    g = cands.sort_values(["idx", "p"], ascending=[True, False]).groupby("idx", sort=True)
    by_idx = {i: d for i, d in g}
    seq = [i for i in obs.idx if i in by_idx]          # observations with no candidate are skipped
    pos = obs.set_index("idx").loc[seq]
    u, v, L = graph.u, graph.v, graph.L

    states = [(by_idx[i].arc_id.to_numpy(), by_idx[i].proj_t.to_numpy(), by_idx[i].p.to_numpy()) for i in seq]
    new_segment = np.r_[True, pos.segment_id.to_numpy()[1:] != pos.segment_id.to_numpy()[:-1]]
    gc = np.r_[np.nan, haversine_m(pos.lon.to_numpy()[:-1], pos.lat.to_numpy()[:-1],
                                   pos.lon.to_numpy()[1:], pos.lat.to_numpy()[1:])]

    # all node pairs needed, solved in one batch per trip
    pairs = set()
    for k in range(1, len(seq)):
        if new_segment[k]:
            continue
        a, b = states[k - 1][0], states[k][0]
        pairs.update(zip(np.repeat(v[a], len(b)).tolist(), np.tile(u[b], len(a)).tolist()))
    pairs = list(pairs)
    dist = dict(zip(pairs, graph.node_dist(pairs))) if pairs else {}

    trans = [None]
    for k in range(1, len(seq)):
        if new_segment[k]:
            trans.append(None)
            continue
        (a, ta, _), (b, tb, _) = states[k - 1], states[k]
        A, B = a[:, None], b[None, :]
        nd = np.array([[dist[(v[x], u[y])] for y in b] for x in a])
        route = (1 - ta[:, None]) * L[A] + nd + tb[None, :] * L[B]
        same = A == B
        route = np.where(same, np.abs(tb[None, :] - ta[:, None]) * L[A], route)
        uturn = (u[A] == v[B]) & (v[A] == u[B])
        trans.append((np.abs(route - gc[k]), uturn))
    return {"idx": np.array(seq), "states": states, "trans": trans, "new_segment": new_segment}


def viterbi(steps, beta, w_emit, uturn_pen, eps=1e-6):
    """Returns the chosen arc per decodable observation and the number of breaks."""
    states, trans = steps["states"], steps["trans"]
    n = len(states)
    choice = np.empty(n, dtype=np.int64)
    breaks = 0
    k0 = 0
    score = w_emit * np.log(states[0][2] + eps)
    back = [None]
    for k in range(1, n + 1):
        boundary = k == n or trans[k] is None
        if not boundary:
            diff, uturn = trans[k]
            logT = -diff / beta - uturn_pen * uturn
            tot = score[:, None] + logT
            best = tot.argmax(0)
            new = tot[best, np.arange(tot.shape[1])]
            if not np.isfinite(new).any():        # no feasible transition: break the chain here
                boundary = True
                breaks += 1
            else:
                score = new + w_emit * np.log(states[k][2] + eps)
                back.append(best)
                continue
        # backtrack the finished chain k0 .. k-1
        j = int(np.argmax(score))
        for t in range(k - 1, k0 - 1, -1):
            choice[t] = states[t][0][j]
            if t > k0:
                j = back[t - k0][j]
        if k < n:
            k0, back = k, [None]
            score = w_emit * np.log(states[k][2] + eps)
    return choice, breaks


def classifier_only(steps):
    return np.array([s[0][0] for s in steps["states"]])   # states are sorted by p, highest first


# ---------------------------------------------------------------- assignment and evaluation helpers
def assign_all(steps, obs, chosen):
    """Every observation gets an arc: ones skipped by the decoder (no candidate) take the nearest
    decoded observation's arc in the same time-gap segment."""
    s = pd.Series(chosen, index=steps["idx"], dtype=float).reindex(obs.idx)
    seg = obs.set_index("idx").segment_id
    return s.groupby(seg).transform(lambda x: x.ffill().bfill())


def top1(pred, ref):
    p = pred.reindex(ref.index)
    return int((p == ref).sum()), len(ref)


def continuity(pred, obs, graph):
    """Consecutive observation pairs (same segment) whose arcs are the same, or joined by a forward
    network path at most 100 m longer than the straight-line distance between the fixes."""
    o = obs.set_index("idx")
    a, b = pred.to_numpy()[:-1], pred.to_numpy()[1:]
    keep = (o.segment_id.to_numpy()[:-1] == o.segment_id.to_numpy()[1:]) & ~np.isnan(a) & ~np.isnan(b)
    a, b = a[keep].astype(int), b[keep].astype(int)
    if not len(a):
        return 0, 0
    gc = haversine_m(o.lon.to_numpy()[:-1][keep], o.lat.to_numpy()[:-1][keep],
                     o.lon.to_numpy()[1:][keep], o.lat.to_numpy()[1:][keep])
    nd = graph.node_dist(list(zip(graph.v[a].tolist(), graph.u[b].tolist())))
    ok = (a == b) | (nd <= gc + 100)
    return int(ok.sum()), len(ok)


def rebuild_route(pred, obs, graph):
    """Ordered arcs the vehicle passes: distinct consecutive arcs, joined by shortest paths."""
    seg = obs.set_index("idx").segment_id
    arcs, prev, prev_seg = [], None, None
    for i, a in pred.items():
        if np.isnan(a):
            continue
        a = int(a)
        if prev is not None and seg[i] == prev_seg and a != prev:
            arcs.extend(graph.path_arcs(prev, a) or [])
        if a != prev:
            arcs.append(a)
        prev, prev_seg = a, seg[i]
    return arcs


def route_prf(arcs, ref_arcs, arc_len):
    """Length-weighted precision, recall and F1 of a predicted arc set against reference arcs."""
    pred = set(arcs)
    hit = sum(arc_len[a] for a in pred & ref_arcs)
    prec = hit / max(sum(arc_len[a] for a in pred), 1e-9)
    rec = hit / max(sum(arc_len[a] for a in ref_arcs), 1e-9)
    return prec, rec, (2 * prec * rec / (prec + rec) if prec + rec else 0.0)
