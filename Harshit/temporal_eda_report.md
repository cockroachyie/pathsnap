# Temporal EDA and Movement Analysis Report

**Prepared by:** Harshit

**Project area:** GPS trajectory data analysis

**Date:** 18 November 2026

## 1. Purpose and scope

This work examines how GPS observations change over time and how the recorded
movement behaves between consecutive observations. It covers timestamp
regularity, trip duration, distance, speed, approximate acceleration, and
potentially suspicious movement.

This assignment is exploratory data analysis (EDA). It does not train or
implement a machine-learning model.

## 2. Data analysed

The analysis processed 90 trips:

- 80 training trips
- 10 test trips
- 227,577 GPS observations in total

Each `.track` observation provides longitude, latitude, and a time value. The
time values are treated as elapsed seconds between observations.

## 3. Analysis method

For each trip, observations are sorted by time. Between each consecutive pair,
the script calculates:

1. **Time interval:** difference between consecutive time values.
2. **Distance:** approximate geographic distance in metres using the Haversine
   formula.
3. **Speed:** distance divided by time interval, converted to km/h.
4. **Approximate acceleration:** change in speed in metres per second divided
   by the time interval.

The analysis uses initial, rule-based thresholds to identify values for review:

- Time gap over 5 seconds
- Speed over 180 km/h
- Consecutive-point distance over 500 metres or 1,000 metres
- Absolute acceleration over 5 m/s²
- Consecutive GPS points within 2 metres, counted as near-stationary steps

For a separate filtered speed summary, intervals must be at least 0.999 seconds
and no more than 5 seconds, and the calculated speed must be from 0 to 250 km/h.
These are heuristic filters, not validated ground truth.

## 4. Main findings

### Timestamp behaviour

- 227,487 consecutive-observation intervals were analysed.
- 219,934 intervals were exactly 1 second: **96.68%**.
- 1,260 intervals were longer than 5 seconds.
- The average interval was 2.087 seconds; this is affected by extreme gaps.
- The largest interval was 153,064 seconds.
- No duplicate timestamps were counted.
- The median trip duration was 1,710.001 seconds (about 28.5 minutes).
- The shortest trip was 265 seconds; the longest was 154,030 seconds.

Trips 00000090 and 00000078 had the largest timestamp gaps. These gaps may
indicate missing observations or other timestamp irregularities.

### Movement behaviour

- Average raw calculated speed: 49.808 km/h.
- Median raw calculated speed: 40.662 km/h.
- Maximum raw calculated speed: 648.497 km/h.
- Average speed after the stated interval and speed filters: 49.815 km/h.
- 1,532 observations exceeded 180 km/h.
- Four consecutive-point distances exceeded 500 metres; none exceeded 1,000
  metres.
- 107,915 acceleration estimates exceeded 5 m/s² in absolute value.
- 6,010 consecutive point pairs were within 2 metres.

Trips 00000079, 00000033, and 00000013 had the highest raw calculated speeds.
These values should be treated as suspicious observations, not proof of actual
vehicle speeds.

## 5. Interpretation and limitations

Most observations are sampled at one-second intervals, but a small number of
extreme gaps strongly affect averages and make some trip durations unusual.
A large time gap makes the movement estimate for that segment less
representative because movement during the gap was not observed. For a fixed
distance, a larger time interval lowers the calculated average speed; GPS
position noise or jumps can instead create high-speed outliers.

Acceleration is especially sensitive to GPS position noise and irregular
sampling, so the high acceleration-flag count requires further review.
Thresholds in this EDA are screening rules and have not been calibrated against
vehicle type, road type, or ground-truth labels. Timestamps are sorted before
interval checks, so the negative-gap count does not test whether the original
file rows were out of order.

The filtered speed average is a heuristic summary only. The analysis does not
perform map matching, remove or repair GPS errors, validate suspicious values
against ground truth, or train a route-prediction model.

## 6. Visualizations and deliverables

The `Harshit` folder contains:

- `temporal_eda.py` — reusable analysis script
- `temporal_eda.ipynb` — notebook for presenting and exploring the work
- `sampling_interval_analysis.png` — sampling intervals, displayed from 0 to
  60 seconds
- `speed_distribution.png` — raw speed distribution, displayed from 0 to
  250 km/h
- `speed_vs_time.png` — speed over time for the shortest-duration trip
- `time_gap_analysis.png` — time-gap distribution, displayed from 0 to
  60 seconds
- `trip_duration_analysis.png` — trip-duration distribution
- `acceleration_distribution.png` — acceleration distribution, displayed from
  -20 to 20 m/s²
- `timestamp_summary.csv` — timestamp metrics per trip
- `movement_summary.csv` — movement metrics per trip
- `temporal_findings.md` — concise findings summary
- `temporal_eda_report.md` — this full report

## 7. Completion and handoff

The analysis script was run locally across all 90 trips and generated the
summaries and visualizations above. The packaged Harshit files were uploaded to
Colab for display and copied to Google Drive. Displaying the packaged results
in Colab does not rerun the analysis; rerunning requires the raw `train` and
`student_test` data folders.

The Harshit work has also been pushed to the `harshit-temporal-eda` branch of
the GitHub repository. It has not been merged into the default branch.

## 8. Conclusion

The GPS observations are mostly sampled every second, but extreme time gaps,
high calculated speeds, and noisy acceleration estimates require attention.
These findings can guide data-quality checks before the trajectories are used
for route prediction or other downstream analysis.
