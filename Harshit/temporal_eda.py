from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent


def load_track(track_file):
    return pd.read_csv(
        track_file,
        sep=r"\s+",
        names=["longitude", "latitude", "time"],
        dtype=float,
    )


def find_track_files():
    return sorted(PROJECT_ROOT.glob("train/*/*.track")) + sorted(
        PROJECT_ROOT.glob("student_test/*/*.track")
    )


def haversine_distance(longitude_1, latitude_1, longitude_2, latitude_2):
    earth_radius = 6371000.0
    latitude_1, longitude_1, latitude_2, longitude_2 = map(
        np.radians,
        [latitude_1, longitude_1, latitude_2, longitude_2],
    )
    latitude_difference = latitude_2 - latitude_1
    longitude_difference = longitude_2 - longitude_1
    value = (
        np.sin(latitude_difference / 2.0) ** 2
        + np.cos(latitude_1)
        * np.cos(latitude_2)
        * np.sin(longitude_difference / 2.0) ** 2
    )
    return 2 * earth_radius * np.arcsin(np.sqrt(np.clip(value, 0, 1)))


def analyze_track(track_file):
    track = load_track(track_file).sort_values("time").reset_index(drop=True)

    track["time_diff"] = track["time"].diff()
    track["distance_m"] = haversine_distance(
        track["longitude"].shift(),
        track["latitude"].shift(),
        track["longitude"],
        track["latitude"],
    )
    track["speed_mps"] = track["distance_m"] / track["time_diff"]
    track["speed_kph"] = track["speed_mps"] * 3.6
    track["acceleration_mps2"] = (
        track["speed_mps"].diff() / track["time_diff"]
    )

    gaps = track["time_diff"].dropna()
    valid_speeds = track["speed_kph"].replace([np.inf, -np.inf], np.nan).dropna()

    timestamp_summary = {
        "trip": track_file.parent.name,
        "dataset": track_file.parts[-3],
        "observations": len(track),
        "duration_seconds": float(track["time"].max() - track["time"].min()),
        "average_gap_seconds": float(gaps.mean()) if len(gaps) else 0.0,
        "minimum_gap_seconds": float(gaps.min()) if len(gaps) else 0.0,
        "maximum_gap_seconds": float(gaps.max()) if len(gaps) else 0.0,
        "one_second_gaps": int(gaps.eq(1.0).sum()),
        "one_second_percentage": (
            100 * gaps.eq(1.0).sum() / len(gaps) if len(gaps) else 0.0
        ),
        "gaps_over_5_seconds": int(gaps.gt(5.0).sum()),
        "duplicate_times": int(track["time"].duplicated().sum()),
        "negative_gaps": int(gaps.lt(0.0).sum()),
    }

    acceleration = (
        track["acceleration_mps2"].replace([np.inf, -np.inf], np.nan).abs().dropna()
    )
    reliable = track[
        track["time_diff"].ge(0.999)
        & track["time_diff"].le(5.0)
        & track["speed_kph"].ge(0)
        & track["speed_kph"].le(250)
    ]

    movement_summary = {
        "trip": track_file.parent.name,
        "dataset": track_file.parts[-3],
        "observations": len(track),
        "total_distance_m": float(track["distance_m"].sum()),
        "average_speed_kph": float(valid_speeds.mean()) if len(valid_speeds) else 0,
        "median_speed_kph": float(valid_speeds.median()) if len(valid_speeds) else 0,
        "maximum_speed_kph": float(valid_speeds.max()) if len(valid_speeds) else 0,
        "reliable_average_speed_kph": (
            float(reliable["speed_kph"].mean()) if len(reliable) else 0
        ),
        "maximum_acceleration": float(acceleration.max()) if len(acceleration) else 0,
        "reliable_speeds": reliable["speed_kph"].tolist(),
        "speeds_over_180_kph": int(valid_speeds.gt(180).sum()),
        "jumps_over_500m": int(track["distance_m"].gt(500).sum()),
        "jumps_over_1000m": int(track["distance_m"].gt(1000).sum()),
        "acceleration_over_5": int(
            track["acceleration_mps2"].abs().gt(5).sum()
        ),
        "stationary_positions": int(track["distance_m"].le(2).sum()),
    }

    return track, timestamp_summary, movement_summary


def save_histogram(values, title, xlabel, filename, bins=50, value_range=None):
    clean_values = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    if value_range:
        clean_values = clean_values[
            clean_values.ge(value_range[0]) & clean_values.le(value_range[1])
        ]
    plt.figure(figsize=(10, 6))
    plt.hist(clean_values, bins=bins, edgecolor="black")
    plt.xlabel(xlabel)
    plt.ylabel("Number of observations")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close()


def save_speed_vs_time(track):
    plt.figure(figsize=(12, 6))
    plt.plot(track["time"], track["speed_kph"])
    plt.xlabel("Time (seconds)")
    plt.ylabel("Speed (km/h)")
    plt.title("Speed versus Time")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "speed_vs_time.png", dpi=150)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp_results = []
    movement_results = []
    all_gaps = []
    all_speeds = []
    all_accelerations = []
    all_reliable_speeds = []
    all_durations = []
    speed_vs_time_track = None
    smallest_duration = float("inf")

    for track_file in find_track_files():
        track, timestamp_result, movement_result = analyze_track(track_file)
        timestamp_results.append(timestamp_result)
        movement_results.append(movement_result)

        all_gaps.extend(track["time_diff"].dropna().tolist())
        all_speeds.extend(track["speed_kph"].tolist())
        all_accelerations.extend(track["acceleration_mps2"].tolist())
        all_reliable_speeds.extend(movement_result["reliable_speeds"])
        all_durations.append(timestamp_result["duration_seconds"])

        duration = timestamp_result["duration_seconds"]
        if duration < smallest_duration:
            smallest_duration = duration
            speed_vs_time_track = track

    timestamps = pd.DataFrame(timestamp_results)
    movements = pd.DataFrame(movement_results)
    timestamps.to_csv(OUTPUT_DIR / "timestamp_summary.csv", index=False)
    movements.drop(columns=["reliable_speeds"]).to_csv(
        OUTPUT_DIR / "movement_summary.csv", index=False
    )

    save_histogram(
        all_gaps,
        "Sampling Interval Distribution (0-60 seconds)",
        "Sampling interval (seconds)",
        "sampling_interval_analysis.png",
        bins=60,
        value_range=(0, 60),
    )
    save_histogram(
        all_gaps,
        "Time-Gap Distribution (0-60 seconds)",
        "Time gap (seconds)",
        "time_gap_analysis.png",
        bins=60,
        value_range=(0, 60),
    )
    save_histogram(
        all_speeds,
        "Speed Distribution (0-250 km/h)",
        "Speed (km/h)",
        "speed_distribution.png",
        bins=100,
        value_range=(0, 250),
    )
    save_histogram(
        all_accelerations,
        "Acceleration Distribution (-20 to 20 m/s²)",
        "Acceleration (m/s²)",
        "acceleration_distribution.png",
        bins=100,
        value_range=(-20, 20),
    )
    save_histogram(
        all_durations,
        "Trip-Duration Distribution",
        "Trip duration (seconds)",
        "trip_duration_analysis.png",
        bins=50,
    )
    save_speed_vs_time(speed_vs_time_track)

    gap_series = pd.Series(all_gaps).replace([np.inf, -np.inf], np.nan).dropna()
    speed_series = pd.Series(all_speeds).replace([np.inf, -np.inf], np.nan).dropna()
    reliable_speeds = pd.Series(all_reliable_speeds).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    top_gap_trips = timestamps.nlargest(10, "maximum_gap_seconds")[
        ["trip", "dataset", "maximum_gap_seconds", "gaps_over_5_seconds"]
    ]
    top_speed_trips = movements.nlargest(10, "maximum_speed_kph")[
        ["trip", "dataset", "maximum_speed_kph", "speeds_over_180_kph"]
    ]

    report = [
        "# Temporal EDA and Movement Findings",
        "",
        "## Dataset Coverage",
        f"- Trips analysed: {len(timestamps)}",
        f"- Training trips: {int((timestamps['dataset'] == 'train').sum())}",
        f"- Test trips: {int((timestamps['dataset'] == 'student_test').sum())}",
        f"- Total observations: {int(timestamps['observations'].sum())}",
        "",
        "## Timestamp Findings",
        f"- Consecutive sampling gaps: {len(gap_series)}",
        f"- 1-second gaps: {int(gap_series.eq(1.0).sum())}",
        f"- 1-second gap percentage: {100 * gap_series.eq(1.0).sum() / len(gap_series):.2f}%",
        f"- Average sampling interval: {gap_series.mean():.3f} seconds",
        f"- Maximum sampling interval: {gap_series.max():.3f} seconds",
        f"- Gaps over 5 seconds: {int(gap_series.gt(5.0).sum())}",
        f"- Duplicate timestamps: {int(timestamps['duplicate_times'].sum())}",
        f"- Negative timestamp gaps: {int(timestamps['negative_gaps'].sum())}",
        f"- Median trip duration: {timestamps['duration_seconds'].median():.3f} seconds",
        f"- Minimum trip duration: {timestamps['duration_seconds'].min():.3f} seconds",
        f"- Maximum trip duration: {timestamps['duration_seconds'].max():.3f} seconds",
        "",
        "## Movement Findings",
        f"- Average GPS speed: {speed_series.mean():.3f} km/h",
        f"- Median GPS speed: {speed_series.median():.3f} km/h",
        f"- Maximum GPS speed: {speed_series.max():.3f} km/h",
        f"- Average reliable GPS speed: {reliable_speeds.mean():.3f} km/h",
        f"- Observations over 180 km/h: {int(movements['speeds_over_180_kph'].sum())}",
        f"- Jumps over 500 metres: {int(movements['jumps_over_500m'].sum())}",
        f"- Jumps over 1,000 metres: {int(movements['jumps_over_1000m'].sum())}",
        f"- Acceleration readings over 5 m/s²: {int(movements['acceleration_over_5'].sum())}",
        f"- Stationary consecutive positions: {int(movements['stationary_positions'].sum())}",
        "",
        "## Interpretation",
        "- Most observations are sampled every second.",
        "- Large gaps are likely missing observations or timestamp irregularities.",
        "- Large timestamp gaps make segment-speed estimates unreliable; GPS noise or position jumps can create high-speed outliers.",
        "- Reliable speeds were filtered to a 1-5 second gap and 0-250 km/h range.",
        "- The reliable-speed average now uses the same gap filter, so it is not inflated by large time gaps.",
        "- Stationary observations indicate that several consecutive GPS points are very close.",
        "",
        "## Trips with the Largest Timestamp Gaps",
        top_gap_trips.to_markdown(index=False),
        "",
        "## Trips with the Highest Raw GPS Speeds",
        top_speed_trips.to_markdown(index=False),
        "",
        "## Conclusion",
        "Timestamp irregularities and extreme gaps should be flagged before route",
        "prediction. Raw speed values should not be treated as reliable until gaps",
        "have been cleaned or excluded.",
        "",
    ]
    (OUTPUT_DIR / "temporal_findings.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print(f"Analysed {len(timestamps)} trips.")
    for filename in [
        "timestamp_summary.csv",
        "movement_summary.csv",
        "sampling_interval_analysis.png",
        "speed_distribution.png",
        "speed_vs_time.png",
        "acceleration_distribution.png",
        "time_gap_analysis.png",
        "trip_duration_analysis.png",
        "temporal_findings.md",
    ]:
        print(f"Created: {OUTPUT_DIR / filename}")


if __name__ == "__main__":
    main()
