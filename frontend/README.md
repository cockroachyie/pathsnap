# PathSnap Evaluator UI

Static, build-free frontend for the evaluator. It runs on Laptop 2 and talks only to the API on Laptop 1. It lists the evaluation trips, sends the selected trip ID to `POST /predict`, and shows the returned route, GPS points, and metrics on a Leaflet map.

The UI never contains ground-truth labels or trip answers. All scoring happens server-side.

## Files

| File | Purpose |
|---|---|
| `index.html` | Page layout |
| `styles.css` | Styling |
| `app.js` | API calls, map rendering, results display |

## Running

The page must be served over HTTP (not opened as `file://`). From this folder:

```bash
python -m http.server 5500
```

Then open `http://localhost:5500` on Laptop 2.

To point at the API on Laptop 1, either:

- type the base URL into the **API base URL** box and click **Connect** (remembered in the browser), or
- open `http://localhost:5500/?api=http://<laptop1-ip>:8000`.

The default is `http://localhost:8000`.

## API contract

The API must expose these endpoints and allow cross-origin requests from the UI origin (CORS).

### `GET /trips`

Returns the ten evaluation trip IDs.

```json
{ "trips": ["00000000", "00000007", "..."] }
```

A bare JSON array of strings is also accepted.

### `POST /predict`

Request:

```json
{ "trip_id": "00000000" }
```

Response (fields other than `trip_id` are optional; the UI shows what is present):

```json
{
  "trip_id": "00000000",
  "gps_points": [[lon, lat], "..."],
  "predicted_route": [[lon, lat], "..."],
  "predicted_arcs": [123, 456],
  "road_arcs": [[[lon, lat], [lon, lat]]],
  "metrics": {
    "top1_accuracy": 0.91,
    "macro_f1": 0.88,
    "route_continuity": 0.97,
    "candidate_miss_rate": 0.02,
    "log_loss": 0.31,
    "inference_time_ms": 14.2
  }
}
```

- Coordinates are `[longitude, latitude]`, as in the raw dataset.
- Accuracy-type metrics are fractions in `[0, 1]` and are shown as percentages.
- Errors: a non-2xx response with a JSON `detail`, `error`, or `message` field is shown in the UI.

## Notes

- Map tiles and Leaflet are loaded from public CDNs (OpenStreetMap, unpkg), so Laptop 2 needs internet access for the base map. The route and GPS layers render without tiles.
- The backend in this repository is not yet implemented against this contract. Verify the field names against the API before the evaluation.
