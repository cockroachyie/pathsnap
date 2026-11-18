# PathSnap — API (Laptop 1) and evaluator UI (Laptop 2)

```
Laptop 2: ui/index.html  ──HTTP──▶  Laptop 1: FastAPI (api/app.py)
  trip list, map, results            raw trip ─▶ cleaning ─▶ features ─▶ HistGradientBoosting ─▶ HMM/Viterbi
                                     then, server-side only: score against the trip's .route file
```

* The API runs the **complete pipeline live** on the raw `.track`, `.nodes`, `.arcs` files of the selected trip,
  reusing `Data Cleaning/clean_dataset.py` and `Feature Extraction/extract_features.py` unchanged.
* **Ground truth never reaches the model or the UI.** Prediction runs in a sandbox that receives only
  `.track/.nodes/.arcs`; the `.route` file is opened afterwards by the evaluation step alone, which returns
  aggregate accuracy (top-1, route precision/recall/F1). The reference route itself is never sent.
* Every request and pipeline stage is logged to the terminal and to `api/logs/api.log`.

## Laptop 1 — API

**Once:** set up the environment and build the model artifacts (needs `data_features/` and `data_clean/`
next to the repo, produced by the cleaning and feature-extraction scripts).

```bash
cd api
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python build_artifacts.py            # trains the final model here -> artifacts/
```

**Each run:** point `TRIPS_DIR` at the folder holding the evaluation trips, in the raw dataset layout
(`<trip>/<trip>.track`, `.nodes`, `.arcs`, `.route`), and start the server on all network interfaces:

```bash
cd api && source .venv/bin/activate
TRIPS_DIR="/path/to/evaluation_trips" uvicorn app:app --host 0.0.0.0 --port 8000
```

Windows (PowerShell): `$env:TRIPS_DIR="C:\path\to\evaluation_trips"; uvicorn app:app --host 0.0.0.0 --port 8000`

Without `TRIPS_DIR` it serves `Student folder/student_test` (10 trips) — useful for rehearsal.

Find Laptop 1's address for the UI: macOS `ipconfig getifaddr en0`, Windows `ipconfig` (IPv4 Address).
Allow incoming connections for Python if the firewall asks. Both laptops must be on the same network.

Check it: `http://<laptop1-ip>:8000/health` and `/trips` in a browser; interactive docs at `/docs`.

## Laptop 2 — UI

Copy the `ui` folder (only `index.html`) and serve it:

```bash
cd ui
python3 -m http.server 5500
```

Open `http://localhost:5500`, enter `http://<laptop1-ip>:8000` and press **Connect**
(or open `http://localhost:5500/?api=http://<laptop1-ip>:8000`). Pick any of the trips: the API runs the
pipeline and the page shows the predicted route, classifier-only vs HMM accuracy, and the stage timings.
The map needs internet for the map library and tiles; the results panel works without it.

## Endpoints

| Method | Path | Returns |
| --- | --- | --- |
| GET | `/health` | model and decoder settings, number of trips |
| GET | `/trips` | evaluation trip IDs |
| POST | `/predict` `{"trip_id": "..."}` | GPS points, predicted route, per-observation predicted arc, accuracy, timings |

## Rehearsal results (student_test as stand-in, 10 trips)

All 10 trips run in 0.2–2.9 s each; HMM top-1 87.4–100% per trip, 95.97% overall, identical to
`Model Training/hmm_viterbi.ipynb`.
