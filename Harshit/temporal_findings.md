# Temporal EDA and Movement Findings

## Dataset Coverage
- Trips analysed: 90
- Training trips: 80
- Test trips: 10
- Total observations: 227577

## Timestamp Findings
- Consecutive sampling gaps: 227487
- 1-second gaps: 219934
- 1-second gap percentage: 96.68%
- Average sampling interval: 2.087 seconds
- Maximum sampling interval: 153064.000 seconds
- Gaps over 5 seconds: 1260
- Duplicate timestamps: 0
- Negative timestamp gaps: 0
- Median trip duration: 1710.001 seconds
- Minimum trip duration: 265.000 seconds
- Maximum trip duration: 154030.000 seconds

## Movement Findings
- Average GPS speed: 49.808 km/h
- Median GPS speed: 40.662 km/h
- Maximum GPS speed: 648.497 km/h
- Average reliable GPS speed: 49.815 km/h
- Observations over 180 km/h: 1532
- Jumps over 500 metres: 4
- Jumps over 1,000 metres: 0
- Acceleration readings over 5 m/s²: 107915
- Stationary consecutive positions: 6010

## Interpretation
- Most observations are sampled every second.
- Large gaps are likely missing observations or timestamp irregularities.
- Large timestamp gaps make segment-speed estimates unreliable; GPS noise or position jumps can create high-speed outliers.
- Reliable speeds were filtered to a 1-5 second gap and 0-250 km/h range.
- The reliable-speed average now uses the same gap filter, so it is not inflated by large time gaps.
- Stationary observations indicate that several consecutive GPS points are very close.

## Trips with the Largest Timestamp Gaps
|     trip | dataset   |   maximum_gap_seconds |   gaps_over_5_seconds |
|---------:|:----------|----------------------:|----------------------:|
| 00000090 | train     |                153064 |                    20 |
| 00000078 | train     |                 52810 |                   124 |
| 00000069 | train     |                  2658 |                     2 |
| 00000021 | train     |                  2161 |                     2 |
| 00000011 | train     |                   909 |                     1 |
| 00000042 | train     |                   670 |                    45 |
| 00000081 | train     |                   536 |                   129 |
| 00000024 | train     |                   458 |                     4 |
| 00000044 | train     |                   341 |                     2 |
| 00000084 | train     |                   312 |                     4 |

## Trips with the Highest Raw GPS Speeds
|     trip | dataset      |   maximum_speed_kph |   speeds_over_180_kph |
|---------:|:-------------|--------------------:|----------------------:|
| 00000079 | train        |             648.497 |                     3 |
| 00000033 | train        |             511.023 |                     3 |
| 00000013 | train        |             509.554 |                   112 |
| 00000043 | train        |             473.988 |                    34 |
| 00000092 | train        |             305.974 |                     4 |
| 00000057 | train        |             301.705 |                     1 |
| 00000032 | train        |             299.799 |                   889 |
| 00000040 | student_test |             262.36  |                     3 |
| 00000001 | train        |             260.716 |                   340 |
| 00000053 | train        |             256.271 |                     7 |

## Conclusion
Timestamp irregularities and extreme gaps should be flagged before route
prediction. Raw speed values should not be treated as reliable until gaps
have been cleaned or excluded.
