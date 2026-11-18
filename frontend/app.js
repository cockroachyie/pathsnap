// PathSnap evaluator UI.
// The UI never holds ground truth. It asks the API for the trip list and for a
// prediction on the selected trip, then renders what the API returns.
//
// API contract (see README.md):
//   GET  {base}/trips    -> { "trips": ["00000000", ...] }
//   POST {base}/predict  body { "trip_id": "..." }
//        -> {
//             "trip_id": "...",
//             "gps_points":      [[lon, lat], ...],
//             "predicted_route": [[lon, lat], ...],
//             "predicted_arcs":  [arc_id, ...],                    // optional
//             "road_arcs":       [[[lon, lat], [lon, lat]], ...],  // optional
//             "metrics": { top1_accuracy, macro_f1, route_continuity,
//                          candidate_miss_rate, log_loss, inference_time_ms }  // all optional
//           }

const DEFAULT_API = "http://localhost:8000";
const STORAGE_KEY = "pathsnap_api_base";
const REQUEST_TIMEOUT_MS = 30000;

// Order and presentation of metric cards. The first entry is the primary metric.
const METRIC_DEFS = [
  { key: "top1_accuracy", label: "Top-1 accuracy", fmt: "pct", primary: true },
  { key: "macro_f1", label: "Macro F1", fmt: "num" },
  { key: "route_continuity", label: "Route continuity", fmt: "pct" },
  { key: "candidate_miss_rate", label: "Candidate miss rate", fmt: "pct" },
  { key: "log_loss", label: "Log loss", fmt: "num" },
  { key: "inference_time_ms", label: "Inference time", fmt: "ms" },
];

const $ = (id) => document.getElementById(id);

const el = {
  form: $("connect-form"),
  apiUrl: $("api-url"),
  connectBtn: $("connect-btn"),
  status: $("status"),
  statusText: $("status").querySelector(".status-text"),
  tripSelect: $("trip-select"),
  runBtn: $("run-btn"),
  error: $("error"),
  metrics: $("metrics"),
  metricsHint: $("metrics-hint"),
  routeInfo: $("route-info"),
  log: $("log"),
  mapEmpty: $("map-empty"),
  mapLoading: $("map-loading"),
};

let apiBase = "";
let map = null;
let overlayLayers = [];
let layerControl = null;
let predicting = false;

// ---------- helpers ----------

function normalizeBase(url) {
  return url.trim().replace(/\/+$/, "");
}

function logLine(message, kind = "info") {
  const li = document.createElement("li");
  li.className = kind;
  li.textContent = `${new Date().toLocaleTimeString()}  ${message}`;
  el.log.prepend(li);
}

function setStatus(text, state) {
  el.statusText.textContent = text;
  el.status.className = `status${state ? ` is-${state}` : ""}`;
}

function showError(message) {
  el.error.textContent = message;
  el.error.hidden = false;
}

function clearError() {
  el.error.textContent = "";
  el.error.hidden = true;
}

function formatMetric(value, fmt) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "N/A";
  if (fmt === "pct") return `${(value * 100).toFixed(2)}%`;
  if (fmt === "ms") return `${value.toFixed(1)} ms`;
  return value.toFixed(4).replace(/\.?0+$/, "");
}

function setBusy(busy) {
  predicting = busy;
  el.runBtn.classList.toggle("is-busy", busy);
  el.runBtn.disabled = busy || !apiBase || el.tripSelect.disabled;
  el.mapLoading.hidden = !busy;
}

async function apiFetch(path, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  let res;
  try {
    res = await fetch(apiBase + path, {
      ...options,
      signal: controller.signal,
      headers: { Accept: "application/json", "Content-Type": "application/json" },
    });
  } catch (err) {
    const reason = err.name === "AbortError" ? "the request timed out" : "the request failed";
    throw new Error(
      `Cannot reach the API at ${apiBase} (${reason}). Check that it is running, the address is correct, and CORS is enabled.`
    );
  } finally {
    clearTimeout(timer);
  }

  let body = null;
  try {
    body = await res.json();
  } catch (_) {
    body = null;
  }

  if (!res.ok) {
    const detail = (body && (body.detail || body.error || body.message)) || res.statusText || "unknown error";
    throw new Error(`API returned ${res.status}: ${typeof detail === "string" ? detail : JSON.stringify(detail)}`);
  }
  return body;
}

// ---------- map ----------

function ensureMap() {
  if (map) return;
  map = L.map("map", { zoomControl: true, attributionControl: true });
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(map);
  map.setView([20, 0], 2);
}

function clearMapLayers() {
  overlayLayers.forEach((layer) => layer.remove());
  overlayLayers = [];
  if (layerControl) {
    layerControl.remove();
    layerControl = null;
  }
}

// API returns [lon, lat]; Leaflet expects [lat, lon].
const toLatLng = ([lon, lat]) => [lat, lon];

function renderMap(result) {
  ensureMap();
  clearMapLayers();

  const layers = {};
  const bounds = [];

  if (Array.isArray(result.road_arcs) && result.road_arcs.length) {
    layers["Road network"] = L.layerGroup(
      result.road_arcs.map((seg) =>
        L.polyline(seg.map(toLatLng), { color: "#94a3b8", weight: 2, opacity: 0.7 })
      )
    );
  }

  if (Array.isArray(result.gps_points) && result.gps_points.length) {
    layers["GPS observations"] = L.layerGroup(
      result.gps_points.map((p) => {
        bounds.push(toLatLng(p));
        return L.circleMarker(toLatLng(p), {
          radius: 4,
          color: "#ffffff",
          weight: 1,
          fillColor: "#2563eb",
          fillOpacity: 0.95,
        });
      })
    );
  }

  if (Array.isArray(result.predicted_route) && result.predicted_route.length) {
    const latlngs = result.predicted_route.map(toLatLng);
    latlngs.forEach((ll) => bounds.push(ll));
    layers["Predicted route"] = L.polyline(latlngs, {
      color: "#e8590c",
      weight: 5,
      opacity: 0.9,
      lineJoin: "round",
    });
  }

  overlayLayers = Object.values(layers);
  overlayLayers.forEach((layer) => layer.addTo(map));
  layerControl = L.control.layers(null, layers, { collapsed: false, position: "topright" }).addTo(map);

  if (bounds.length) {
    map.fitBounds(bounds, { padding: [32, 32], maxZoom: 17 });
  }
  el.mapEmpty.hidden = true;
}

// ---------- results ----------

function renderMetrics(metrics) {
  el.metrics.innerHTML = "";

  for (const def of METRIC_DEFS) {
    if (!(def.key in metrics)) continue;

    const card = document.createElement("div");
    card.className = `metric${def.primary ? " is-primary" : ""}`;

    const label = document.createElement("div");
    label.className = "metric-label";
    label.textContent = def.label;

    const value = document.createElement("div");
    value.className = "metric-value";
    value.textContent = formatMetric(metrics[def.key], def.fmt);

    card.append(label, value);
    el.metrics.append(card);
  }

  const hasMetrics = el.metrics.children.length > 0;
  el.metricsHint.hidden = hasMetrics;
  el.metricsHint.textContent = hasMetrics
    ? ""
    : "The API returned no metrics for this trip.";
}

function renderResult(result) {
  renderMetrics(result.metrics || {});

  const parts = [];
  if (Array.isArray(result.gps_points)) parts.push(`${result.gps_points.length} GPS points`);
  if (Array.isArray(result.predicted_arcs)) parts.push(`${result.predicted_arcs.length} predicted arcs`);
  if (Array.isArray(result.predicted_route)) parts.push(`${result.predicted_route.length} route vertices`);
  el.routeInfo.textContent = parts.join("  ·  ");

  renderMap(result);
}

// ---------- actions ----------

async function loadTrips() {
  const data = await apiFetch("/trips");
  const trips = Array.isArray(data) ? data : data && data.trips;
  if (!Array.isArray(trips) || trips.length === 0) {
    throw new Error("The API returned no evaluation trips.");
  }

  el.tripSelect.innerHTML = "";
  for (const t of trips) {
    const id = typeof t === "string" ? t : String(t.trip_id);
    el.tripSelect.add(new Option(id, id));
  }
  el.tripSelect.disabled = false;
  return trips.length;
}

function resetTrips(message) {
  el.tripSelect.innerHTML = "";
  el.tripSelect.add(new Option(message, ""));
  el.tripSelect.disabled = true;
  el.runBtn.disabled = true;
}

async function connect(event) {
  if (event) event.preventDefault();

  const base = normalizeBase(el.apiUrl.value);
  if (!base) {
    showError("Enter the API base URL first.");
    return;
  }

  apiBase = base;
  localStorage.setItem(STORAGE_KEY, apiBase);
  clearError();
  setStatus("Connecting…", "connecting");
  el.connectBtn.disabled = true;
  resetTrips("Loading trips…");

  try {
    const count = await loadTrips();
    setStatus("Connected", "ok");
    el.runBtn.disabled = predicting;
    logLine(`Connected to ${apiBase}; ${count} trips loaded`, "ok");
  } catch (err) {
    resetTrips("Connect to load trips");
    setStatus("Connection failed", "error");
    showError(err.message);
    logLine(err.message, "error");
  } finally {
    el.connectBtn.disabled = false;
  }
}

async function runPrediction() {
  const tripId = el.tripSelect.value;
  if (!tripId || predicting) return;

  clearError();
  setBusy(true);
  logLine(`POST /predict  trip_id=${tripId}`);

  try {
    const result = await apiFetch("/predict", {
      method: "POST",
      body: JSON.stringify({ trip_id: tripId }),
    });
    renderResult(result);
    logLine(`Prediction received for trip ${tripId}`, "ok");
  } catch (err) {
    showError(err.message);
    logLine(err.message, "error");
  } finally {
    setBusy(false);
  }
}

// ---------- init ----------

function init() {
  const fromQuery = new URLSearchParams(location.search).get("api");
  const initial = fromQuery || localStorage.getItem(STORAGE_KEY) || DEFAULT_API;
  el.apiUrl.value = normalizeBase(initial);

  el.form.addEventListener("submit", connect);
  el.runBtn.addEventListener("click", runPrediction);
  el.tripSelect.addEventListener("change", () => {
    el.runBtn.disabled = predicting || el.tripSelect.disabled || !el.tripSelect.value;
  });

  ensureMap();
  connect();
}

init();
