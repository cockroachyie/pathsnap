"""PathSnap end-to-end pipeline for one raw trip: cleaning -> features -> classifier -> HMM/Viterbi.

Prediction and evaluation are kept apart on purpose:

* `predict(trip_dir)` copies ONLY the .track, .nodes and .arcs files into a private sandbox and runs the
  whole pipeline there. The .route file (ground truth) is never copied, so no prediction step can read it.
* `evaluate(trip_dir, prediction)` runs afterwards, server-side only. It is the only code that opens the
  .route file, and it returns aggregate accuracy figures, never the reference route itself.

Cleaning and feature extraction reuse the training scripts unchanged (`clean_dataset.py`,
`extract_features.py`), so preprocessing is identical to what the models were trained on.
"""
import importlib.util
import json
import shutil
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import hmm

REPO = Path(__file__).resolve().parents[2]


def _load_script(filename):
    # folder names differ in case between machines ("Data Cleaning" / "data cleaning"): find the file itself
    path = next(p for p in REPO.glob(f"*/{filename}") if ".venv" not in p.parts)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


clean = _load_script("clean_dataset.py")
features = _load_script("extract_features.py")


class Pipeline:
    def __init__(self, artifacts_dir):
        artifacts_dir = Path(artifacts_dir)
        self.model = joblib.load(artifacts_dir / "hgb_model.joblib")
        self.config = json.loads((artifacts_dir / "hmm_config.json").read_text())
        self.beta = float(self.config["beta_m"])
        self.uturn = float(self.config["uturn_penalty"])
        self.w_emit = float(self.config.get("emission_weight", 1.0))
        hmm.MAX_ROUTE_M = float(self.config.get("max_route_m", hmm.MAX_ROUTE_M))
        self.feature_names = features.FEATURES

    # ------------------------------------------------------------------ prediction (no labels)
    def predict(self, trip_dir):
        trip_dir = Path(trip_dir)
        tid = trip_dir.name
        timings = {}
        with tempfile.TemporaryDirectory(prefix="pathsnap_pred_") as tmp:
            tmp = Path(tmp)
            sandbox = tmp / "raw" / "eval" / tid
            sandbox.mkdir(parents=True)
            for ext in ("track", "nodes", "arcs"):          # .route deliberately NOT copied
                shutil.copy(trip_dir / f"{tid}.{ext}", sandbox / f"{tid}.{ext}")

            t0 = time.perf_counter()
            clean.process_trip(sandbox, "eval", tmp / "clean")
            timings["cleaning_s"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            summary = features.process_trip(tmp / "clean", "eval", tid, tmp / "features")
            rows = pd.read_csv(tmp / "features" / "eval" / f"{tid}.csv.gz", dtype={"trip_id": str})
            timings["features_s"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            rows["p"] = self.model.predict_proba(rows[self.feature_names])[:, 1] if len(rows) else []
            timings["classifier_s"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            net = dict(np.load(tmp / "clean" / "eval" / tid / "network_clean.npz"))  # read now: sandbox is deleted
            graph = hmm.TripGraph(net)
            node_lon, node_lat = net["node_lon"], net["node_lat"]
            trk = pd.read_csv(tmp / "clean" / "eval" / tid / "track_clean.csv")
            obs = trk[~trk.is_spike & (trk.segment_id >= 0)].reset_index(drop=True)
            steps = hmm.build_steps(rows, obs, graph)
            chosen, breaks = hmm.viterbi(steps, self.beta, self.w_emit, self.uturn)
            pred_hmm = hmm.assign_all(steps, obs, chosen)
            pred_clf = hmm.assign_all(steps, obs, hmm.classifier_only(steps))
            route = hmm.rebuild_route(pred_hmm, obs, graph)
            timings["hmm_s"] = time.perf_counter() - t0

        def coords(arc):
            return [[round(float(node_lat[graph.u[arc]]), 6), round(float(node_lon[graph.u[arc]]), 6)],
                    [round(float(node_lat[graph.v[arc]]), 6), round(float(node_lon[graph.v[arc]]), 6)]]

        # ordered predicted route as polylines: a new line starts wherever consecutive arcs do not join
        lines, cur, prev = [], [], None
        for a in route:
            if prev is not None and graph.v[prev] == graph.u[a]:
                cur.append(coords(a)[1])
            else:
                if cur:
                    lines.append(cur)
                cur = coords(a)
            prev = a
        if cur:
            lines.append(cur)

        clf_arcs = pred_clf.dropna().astype(int).unique()
        cont_h, pairs = hmm.continuity(pred_hmm, obs, graph)
        cont_c, _ = hmm.continuity(pred_clf, obs, graph)
        return {
            "trip_id": tid,
            "obs": obs, "graph": graph, "pred_hmm": pred_hmm, "pred_clf": pred_clf, "route_arcs": route,
            "response": {
                "trip_id": tid,
                "gps": obs[["lat", "lon"]].round(6).to_numpy().tolist(),
                "route_hmm": lines,
                "route_classifier": [coords(a) for a in clf_arcs],
                "predictions": {"idx": obs.idx.tolist(),
                                "arc_hmm": [None if np.isnan(x) else int(x) for x in pred_hmm.to_numpy()]},
                "stats": {"gps_points": int(len(trk)), "observations": int(len(obs)),
                          "segments": int(obs.segment_id.nunique()), "candidate_rows": int(len(rows)),
                          "mean_candidates": float(summary["mean_cands"]),
                          "observations_without_candidates": int(summary["obs_without_candidates"]),
                          "hmm_breaks": int(breaks), "route_arcs": len(route)},
                "continuity": {"hmm": cont_h / max(pairs, 1), "classifier": cont_c / max(pairs, 1)},
                "timings_s": {k: round(v, 3) for k, v in timings.items()},
            },
        }

    # ------------------------------------------------------------------ evaluation (server-side labels)
    def evaluate(self, trip_dir, pred):
        """Accuracy of a finished prediction against the trip's reference route. Only aggregate numbers leave here."""
        trip_dir = Path(trip_dir)
        tid = trip_dir.name
        route_file = trip_dir / f"{tid}.route"
        if not route_file.exists():
            return {"available": False, "reason": "no reference route on the server for this trip"}
        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="pathsnap_eval_") as tmp:
            tmp = Path(tmp)
            raw = tmp / "raw" / "eval" / tid
            raw.mkdir(parents=True)
            for ext in ("track", "nodes", "arcs", "route"):
                shutil.copy(trip_dir / f"{tid}.{ext}", raw / f"{tid}.{ext}")
            clean.process_trip(raw, "eval", tmp / "clean")
            features.process_trip(tmp / "clean", "eval", tid, tmp / "features")
            pts = pd.read_csv(tmp / "features" / "labels" / "eval" / f"{tid}_points.csv.gz")
            ref_route = pd.read_csv(tmp / "clean" / "labels" / "eval" / tid / "route_clean.csv")
        ref = pts[pts.labelled].set_index("idx").ref_arc
        ref_arcs = set(ref_route.canonical_arc_id[~ref_route.is_filled])
        graph = pred["graph"]
        out = {"available": True, "labelled_observations": int(len(ref))}
        for name, series in (("hmm", pred["pred_hmm"]), ("classifier", pred["pred_clf"])):
            k, n = hmm.top1(series, ref)
            arcs = pred["route_arcs"] if name == "hmm" else hmm.rebuild_route(series, pred["obs"], graph)
            prec, rec, f1 = hmm.route_prf(arcs, ref_arcs, graph.L)
            out[name] = {"top1": k / max(n, 1), "route_precision": prec, "route_recall": rec, "route_f1": f1}
        out["candidate_miss_rate"] = float(1 - pts[pts.labelled].ref_in_candidates.mean()) if len(ref) else None
        out["evaluation_s"] = round(time.perf_counter() - t0, 3)
        return out
