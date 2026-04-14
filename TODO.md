## TODO

### Data normalization

1. ~~Read through the source code to run the nbastatsv3 download process, and save it to data_kretsinger~~ **(Not needed)** — nbastatsv3 data is already on disk at `data_kretsinger/NBA/` as tar.xz archives, loaded via `NBADataLoader`. There is no network download step.

2. ~~Use nbastatsv3 to load & inject exogenous timeout data into cdnnba, instead of the heuristic method we've been using~~ **(Infeasible for cdnnba era)** — The NBA removed `Official` timeout labels from their data feed in the 2017-18 season. nbastatsv3 for 2020-2025 (the cdnnba era) only contains `Regular` and `Coach Challenge` subtypes, identical to cdnnba. There is no ground-truth label to merge in.

   **Alternative implemented:** rulebook-based injection, validated against pre-2017 nbastatsv3 ground truth. See `CDNNBADatasetPL.infer_tv_timeouts_rulebook` and `_infer_tv_timeouts_rulebook`. Performance on 2013-2016 validation:

   | Season | Precision | Recall | F1 | Timing (within 10s) |
   |---|---|---|---|---|
   | 2013 | 0.880 | 0.972 | 0.924 | 70.8% |
   | 2014 | 0.861 | 0.973 | 0.913 | 66.0% |
   | 2015 | 0.813 | 0.966 | 0.883 | 70.0% |
   | 2016 | 0.811 | 0.956 | 0.877 | 70.3% |

   The rulebook method is now the default injection strategy for cdnnba
   via `_USE_RULEBOOK_INJECTION = True`. It replaces the earlier
   real-time-excess heuristic, which over-predicted by ~2.2x and
   included noise from replay reviews and other long stoppages.

### Data analysis

Run a comprehensive study of the timeout impact (similar to what's done in `plot_stoppage_run_impact`) but conditioning on as many variables as possible. Use your judgement to determine which variables are most important to condition on, and report on the results. This will be a long process, and will require you to write a lot of code, but it will be worth it in the end. The goal is to understand how timeouts impact the game, and to identify any patterns or trends that might be present. Log the results in a clear and concise manner, and be sure to include any relevant visualizations or charts to help illustrate your findings.

Variables:
- Timeout type (TV, official, coach challenge)
- Game situation (score margin, time remaining, quarter, player fatigue)
- Current team momentum (recent scoring runs, player performance)
- Team characteristics (offensive/defensive ratings, season record, average player age)
- Head-to-head stats (score margin, points per possession, points per minute) - figure out what the "mean" we're comparing against is for these stats, and condition on that
- home vs away
- Regular season vs playoffs
- timeout substitutions
