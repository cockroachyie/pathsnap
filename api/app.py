"""PathSnap FastAPI backend (Laptop 1).

Endpoints
    GET  /health          model loaded, trips available
    GET  /trips           the evaluation trip IDs (no labels, no coordinates)
    POST /predict         {"trip_id": "..."} -> runs the full pipeline on that raw trip, then scores it
                          against the server-side reference route; returns the predicted route + accuracy

Every request and every pipeline stage is logged to the console and to logs/api.log.

Run (from the api folder):
    TRIPS_DIR="/path/to/evaluation trips" uvicorn app:app --host 0.0.0.0 --port 8000
"""
import logging
import os
import re
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathsnap.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
TRIPS_DIR = Path(os.environ.get("TRIPS_DIR", HERE.parents[1] / "Student folder" / "student_test"))
ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", HERE / "artifacts"))

# ---------------------------------------------------------------- logging: console + file
(HERE / "logs").mkdir(exist_ok=True)
fmt = logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", "%Y-%m-%d %H:%M:%S")
log = logging.getLogger("pathsnap")
log.setLevel(logging.INFO)
for h in (logging.StreamHandler(), RotatingFileHandler(HERE / "logs" / "api.log", maxBytes=5_000_000, backupCount=3)):
    h.setFormatter(fmt)
    log.addHandler(h)
log.propagate = False

pipeline = Pipeline(ARTIFACTS)
TRIP_ID = re.compile(r"^[0-9A-Za-z_-]{1,64}$")


def list_trips():
    if not TRIPS_DIR.is_dir():
        return []
    return sorted(d.name for d in TRIPS_DIR.iterdir()
                  if d.is_dir() and all((d / f"{d.name}.{e}").exists() for e in ("track", "nodes", "arcs")))


log.info(f"startup | model {pipeline.config.get('classifier')} | HMM beta={pipeline.beta} m, "
         f"U-turn penalty={pipeline.uturn} | trips dir {TRIPS_DIR} ({len(list_trips())} trips)")

app = FastAPI(title="PathSnap API", version="1.0")
# The UI runs on another laptop, so the browser calls this API cross-origin.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    if request.method != "OPTIONS":
        log.info(f"{request.client.host if request.client else '-'} {request.method} {request.url.path} "
                 f"-> {response.status_code} in {time.perf_counter() - t0:.2f}s")
    return response


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "classifier": pipeline.config.get("classifier"),
            "hmm": {"beta_m": pipeline.beta, "uturn_penalty": pipeline.uturn}, "trips": len(list_trips())}


@app.get("/trips")
def trips():
    return {"trips": list_trips()}


@app.post("/predict")
def predict(req: PredictRequest):
    tid = req.trip_id.strip()
    if not TRIP_ID.match(tid) or tid not in list_trips():
        log.warning(f"predict | unknown trip id {tid!r}")
        raise HTTPException(404, f"unknown trip {tid}")
    trip_dir = TRIPS_DIR / tid

    log.info(f"predict | trip {tid} | start: copying .track .nodes .arcs into a sandbox (no .route)")
    try:
        pred = pipeline.predict(trip_dir)
    except Exception:
        log.exception(f"predict | trip {tid} | pipeline failed")
        raise HTTPException(500, "prediction pipeline failed; see API log")
    r = pred["response"]
    t, s = r["timings_s"], r["stats"]
    log.info(f"predict | trip {tid} | cleaning {t['cleaning_s']:.2f}s | features {t['features_s']:.2f}s "
             f"({s['candidate_rows']:,} candidate rows, {s['mean_candidates']:.1f}/obs) | classifier "
             f"{t['classifier_s']:.2f}s | HMM {t['hmm_s']:.2f}s ({s['segments']} segments, {s['hmm_breaks']} breaks)")

    log.info(f"evaluate | trip {tid} | reading server-side reference route")
    ev = pipeline.evaluate(trip_dir, pred)
    if ev.get("available"):
        log.info(f"evaluate | trip {tid} | top-1 classifier {ev['classifier']['top1']:.1%} -> HMM {ev['hmm']['top1']:.1%} "
                 f"| route F1 {ev['hmm']['route_f1']:.3f} | {ev['labelled_observations']:,} labelled obs "
                 f"| {ev['evaluation_s']:.2f}s")
    else:
        log.info(f"evaluate | trip {tid} | {ev['reason']}")
    r["evaluation"] = ev
    return r
