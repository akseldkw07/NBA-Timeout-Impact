# NBA Data Analysis and Visualization

## Research questions

### Timeouts

Do timeouts have an impact on the game?

- Home vs Away | Time Remaining | Score Margin | Regular vs Post Season | Team Streak / Run
- Compare the impact of timeouts called by the coach vs timeouts called by the players vs the impact of media timeouts
- Compare the point differential in the 5 possessions after a timeout vs the 5 possessions before a timeout, and see if there is a significant difference.

## Data pipeline

### Spine: cdn.nba.com play-by-play (2020-2025)

The core dataset is the cdnnba enriched play-by-play data, accessed via
`CDNNBADatasetPL.load_from_parquet()` and wrapped in `CDNNBAMemoPL`.

Supplemental datasets (boxscores, player stats, rotations, stints) are
loaded alongside the spine via `CDNNBAMemoPL.load_all()` and aligned to
the spine with pointer memo methods (`ptr_boxscores`, `ptr_stints`, etc.).

### TV / Official timeout inference

cdnnba only records coach-called timeouts (`subType` in `{full,
challenge}`); the NBA data feed does not include mandatory TV / official
timeouts. To enable exogenous-timeout analyses (per Weimer et al. 2023),
these are inferred at load time.

Two methods are implemented; the rulebook method is the default.

1. **Rulebook method (default, `_USE_RULEBOOK_INJECTION = True`):**
   deterministic walk through each period's events. For each threshold
   in the post-2017 mandatory schedule (7:00 and 3:00 remaining in every
   quarter), inject a `timeout / official_inferred` row at the first
   dead ball past the threshold, unless a coach timeout already fired in
   that window (absorption).

   Validated against labeled `Official` timeouts in nbastatsv3 2013-2016
   (pre-2017 rulebook thresholds 9:00/6:00/3:00 in Q2 and Q4):
   F1 = 0.88-0.92, timing within 10s for ~70% of matches.

   Injection at cdnnba era: ~14,561 events across ~7,000 games (1.95/game).

2. **Heuristic method (fallback, real-time excess):** detects large
   `timeActual` gaps (>= 90s of real time with no game-clock
   advancement). Over-predicts by ~2.2x and picks up replay reviews /
   injury stoppages in addition to mandatory timeouts.

Injected rows always have `actionType = "timeout"`, `subType =
"official_inferred"`. Downstream memo series use this classification:
- `f_timeout_endogenous` → `subType in {full, challenge}` (coach-called)
- `f_timeout_exogenous` → `subType == official_inferred` (TV/mandatory)
- `f_timeout` → all

## References

Key papers in the `Papers/` directory:

- **Weimer et al. (2023)** *A causal approach for detecting team-level momentum in NBA games* — uses TV timeouts as an exogenous instrument; finds -11.2% scoring for the momentum team after TV timeouts.
- **Assis et al. (2020)** *Stop the Clock: Are Timeout Effects Real?* — causal inference with matching on coach timeouts; finds no effect (regression to the mean).
- **Gibbs, Elmore, Fosdick (2021)** *The Causal Effect of a Timeout at Stopping an Opposing Run in the NBA* — Rubin causal model with genetic matching; finds coach timeouts slightly *hurt* the calling team (ATT = -0.35).
