"""Build the inference artifacts the API loads: the final classifier and the HMM configuration.

Trains HistGradientBoosting (Model 2's configuration) on all usable train trips and writes
    artifacts/hgb_model.joblib
    artifacts/hmm_config.json   (decoder parameters chosen in the HMM notebook on train data only)

Training here, on the machine that serves the API, guarantees the scikit-learn version that saves the model
is the one that loads it. A model trained on Colab can be used instead: copy its hgb_model.joblib and
hmm_config.json into artifacts/, but only if Colab and this machine run the same scikit-learn version.

Usage (from the api folder):
    python build_artifacts.py --features ../../data_features --flags ../../data_clean/trip_flags.csv
"""
import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = Path(__file__).resolve().parent

CLF_CONFIG = dict(learning_rate=0.1, max_iter=300, max_leaf_nodes=31, min_samples_leaf=20,
                  l2_regularization=0.0, max_bins=255, early_stopping=False, random_state=42)

# Chosen in "Model Training/hmm_viterbi.ipynb" by grid search on train out-of-fold predictions only.
HMM_CONFIG = {"beta_m": 1.0, "uturn_penalty": 40.0, "emission_weight": 1.0, "max_route_m": 2000.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=HERE.parents[1] / "data_features")
    ap.add_argument("--flags", type=Path, default=HERE.parents[1] / "data_clean" / "trip_flags.csv")
    ap.add_argument("--out", type=Path, default=HERE / "artifacts")
    args = ap.parse_args()

    feature_names = json.loads((args.features / "feature_params.json").read_text())["FEATURES"]
    flags = pd.read_csv(args.flags, dtype={"trip_id": str})
    trips = sorted(flags[(flags.partition == "train") & flags.use_for_tuning.astype(bool)].trip_id)

    parts = []
    for t in trips:
        f = pd.read_csv(args.features / "train" / f"{t}.csv.gz", dtype={"trip_id": str})
        f["label"] = pd.read_csv(args.features / "labels" / "train" / f"{t}.csv.gz").label.values
        parts.append(f[f.label >= 0])
    train = pd.concat(parts, ignore_index=True)

    t0 = time.time()
    model = HistGradientBoostingClassifier(**CLF_CONFIG).fit(train[feature_names], train.label)
    print(f"trained on {len(trips)} trips, {len(train):,} rows in {time.time() - t0:.1f}s")

    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out / "hgb_model.joblib")
    config = {"classifier": "HistGradientBoosting (Model 2)", "classifier_config": CLF_CONFIG, **HMM_CONFIG,
              "features": feature_names, "train_trips": trips,
              "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__}}
    (args.out / "hmm_config.json").write_text(json.dumps(config, indent=2))
    print("wrote", args.out / "hgb_model.joblib", "and", args.out / "hmm_config.json")


if __name__ == "__main__":
    main()
