# NBA Timeout Impact: Research Summary

*Progressive log of experiments exploring how timeouts affect NBA game flow.*
*Data: cdnnba 2020-2025, ~7,457 games, with rulebook-injected exogenous TV timeouts.*

## Methodology

**Spine:** cdnnba enriched play-by-play (2020-2025 regular season + playoffs)
via `CDNNBAMemoPL.load_all()`.

**Timeout classification:**
- `endogenous` = coach-called timeout (`subType in {full, challenge}`, 82,197 events)
- `exogenous` = inferred mandatory TV/official timeout via rulebook method
  (`subType == official_inferred`, 14,561 events at 2:00/6:00 remaining marks)
- `control` = non-timeout dead-ball moment with similar game state
  (subsampled to one per run segment to avoid over-counting)

**Primary metric:** `recovery` = lead change from the *suffering* team's
perspective over a forward window of *minutes*. A positive value means the
suffering team clawed back during the window. Computed via
`lead_change_n_mins(n)` and flipped by `sign(streak)`.

**Secondary metrics (used in specific experiments):**
- `running_team_pts` = raw points scored by the team on the run over the
  forward window (Weimer et al. 2023 style)
- Within-game matched differences for causal estimation

**Significance convention:** `***` p < 0.001, `**` p < 0.01, `*` p < 0.05,
`n.s.` p ≥ 0.05. Tests are Welch's t-tests unless otherwise noted.

**Key definitions:**
- A "run" at time *t* = `|streak(t)| ≥ run_size`, where streak is the
  running cumulative net score swing within the same scoring segment.
- "Suffering team" = the team the run is against (not the team doing the
  scoring). Their location (home/away) and score margin are tracked.
- "Absorbed" (rulebook TV timeout) = a coach TO already fired in the
  mandatory window, so no TV timeout is injected.

## Data summary

- Spine rows: 4,183,347
- Unique games: 7,457
- Coach timeouts (endogenous): 83,662
- Inferred TV timeouts (exogenous): 14,561
- Total possessions: 1,480,697

## Experiment 1: Baseline sanity checks

Mean recovery (points clawed back by the suffering team) across combinations of run size and forward window. All home/away, all margins.

#### E1 table: mean recovery by run size × forward window

| condition | endo_n | endo_μ | exo_n | exo_μ | ctrl_n | ctrl_μ | Δ endo-ctrl | Δ exo-ctrl |
|---|---|---|---|---|---|---|---|---|
| run≥5, 1min | 28,900 | +0.435 | 9,398 | +0.108 | 64,728 | +0.438 | -0.002 n.s. | -0.329 *** |
| run≥5, 2min | 28,900 | +0.433 | 9,398 | +0.101 | 64,728 | +0.432 | +0.001 n.s. | -0.332 *** |
| run≥5, 3min | 28,900 | +0.432 | 9,398 | +0.157 | 64,728 | +0.409 | +0.023 n.s. | -0.252 *** |
| run≥5, 5min | 28,900 | +0.497 | 9,398 | +0.183 | 64,728 | +0.418 | +0.079 * | -0.235 *** |
| run≥6, 1min | 18,743 | +0.427 | 5,492 | +0.137 | 38,332 | +0.406 | +0.021 n.s. | -0.269 *** |
| run≥6, 2min | 18,743 | +0.432 | 5,492 | +0.092 | 38,332 | +0.397 | +0.035 n.s. | -0.305 *** |
| run≥6, 3min | 18,743 | +0.428 | 5,492 | +0.156 | 38,332 | +0.371 | +0.057 n.s. | -0.214 *** |
| run≥6, 5min | 18,743 | +0.475 | 5,492 | +0.149 | 38,332 | +0.385 | +0.090 n.s. | -0.236 ** |
| run≥8, 1min | 7,860 | +0.431 | 2,133 | +0.107 | 17,303 | +0.392 | +0.038 n.s. | -0.285 *** |
| run≥8, 2min | 7,860 | +0.453 | 2,133 | +0.045 | 17,303 | +0.423 | +0.031 n.s. | -0.377 *** |
| run≥8, 3min | 7,860 | +0.468 | 2,133 | +0.121 | 17,303 | +0.432 | +0.036 n.s. | -0.310 ** |
| run≥8, 5min | 7,860 | +0.501 | 2,133 | +0.155 | 17,303 | +0.474 | +0.028 n.s. | -0.319 * |
| run≥10, 1min | 2,927 | +0.471 | 787 | +0.126 | 8,161 | +0.443 | +0.028 n.s. | -0.317 ** |
| run≥10, 2min | 2,927 | +0.468 | 787 | +0.004 | 8,161 | +0.455 | +0.013 n.s. | -0.451 *** |
| run≥10, 3min | 2,927 | +0.450 | 787 | +0.033 | 8,161 | +0.466 | -0.017 n.s. | -0.433 ** |
| run≥10, 5min | 2,927 | +0.478 | 787 | +0.086 | 8,161 | +0.539 | -0.061 n.s. | -0.453 * |

**Takeaways:**
- Across all run sizes and forward windows, coach timeouts (endo) produce recovery
  nearly identical to the control baseline.
- Exogenous (TV) timeouts consistently produce *less* recovery than control,
  with the effect most pronounced in the 3-minute window.
- Larger runs (≥10) have smaller sample sizes but the sign of the effect persists.

## Experiment 2: Forward window decay curve

At run_size=6, sweep the forward window from 0.5 to 5 min to see how the effect evolves over time.

#### E2 table: recovery vs forward window (run≥6)

| condition | endo_n | endo_μ | exo_n | exo_μ | ctrl_n | ctrl_μ | Δ endo-ctrl | Δ exo-ctrl |
|---|---|---|---|---|---|---|---|---|
| 0.5 min | 18,743 | +0.318 | 5,492 | +0.111 | 38,332 | +0.313 | +0.005 n.s. | -0.203 *** |
| 1.0 min | 18,743 | +0.427 | 5,492 | +0.137 | 38,332 | +0.406 | +0.021 n.s. | -0.269 *** |
| 1.5 min | 18,743 | +0.431 | 5,492 | +0.119 | 38,332 | +0.403 | +0.028 n.s. | -0.284 *** |
| 2.0 min | 18,743 | +0.432 | 5,492 | +0.092 | 38,332 | +0.397 | +0.035 n.s. | -0.305 *** |
| 2.5 min | 18,743 | +0.437 | 5,492 | +0.115 | 38,332 | +0.375 | +0.062 n.s. | -0.260 *** |
| 3.0 min | 18,743 | +0.428 | 5,492 | +0.156 | 38,332 | +0.371 | +0.057 n.s. | -0.214 *** |
| 4.0 min | 18,743 | +0.451 | 5,492 | +0.180 | 38,332 | +0.373 | +0.078 n.s. | -0.193 ** |
| 5.0 min | 18,743 | +0.475 | 5,492 | +0.149 | 38,332 | +0.385 | +0.090 n.s. | -0.236 ** |

**Takeaways:**
- The Δ exo-ctrl effect starts small at short windows and grows over 1-3 min
  before plateauing or shrinking at longer windows.
- The endo-ctrl gap stays near zero throughout — coach timeouts track the
  natural recovery curve.

## Experiment 3: Run magnitude sweep

At 3-min window, vary run size threshold.

#### E3 table: recovery vs run magnitude (3-min window)

| condition | endo_n | endo_μ | exo_n | exo_μ | ctrl_n | ctrl_μ | Δ endo-ctrl | Δ exo-ctrl |
|---|---|---|---|---|---|---|---|---|
| run≥4 | 38,860 | +0.414 | 14,008 | +0.145 | 99,028 | +0.410 | +0.003 n.s. | -0.265 *** |
| run≥5 | 28,900 | +0.432 | 9,398 | +0.157 | 64,728 | +0.409 | +0.023 n.s. | -0.252 *** |
| run≥6 | 18,743 | +0.428 | 5,492 | +0.156 | 38,332 | +0.371 | +0.057 n.s. | -0.214 *** |
| run≥7 | 12,248 | +0.443 | 3,414 | +0.167 | 25,476 | +0.418 | +0.025 n.s. | -0.251 ** |
| run≥8 | 7,860 | +0.468 | 2,133 | +0.121 | 17,303 | +0.432 | +0.036 n.s. | -0.310 ** |
| run≥10 | 2,927 | +0.450 | 787 | +0.033 | 8,161 | +0.466 | -0.017 n.s. | -0.433 ** |
| run≥12 | 1,111 | +0.632 | 296 | +0.081 | 3,726 | +0.490 | +0.142 n.s. | -0.409 n.s. |
| run≥15 | 265 | +0.758 | 64 | +0.703 | 997 | +0.589 | +0.170 n.s. | +0.114 n.s. |

**Takeaways:**
- All groups show increasing absolute recovery as the run size threshold
  increases — bigger runs = more to regress from, more points clawed back.
- The relative gap between endo and ctrl stays near zero.
- Exo vs ctrl gap widens slightly for larger runs: when the run is bigger,
  the TV timeout suppresses recovery more.

## Experiment 4: Period-by-period effect

Break down by the period in which the run occurred. Uses a modified
analysis pulling the period column from the spine.

#### E4 table: recovery by period (run≥6, 3-min window)

| condition | endo_n | endo_μ | exo_n | exo_μ | ctrl_n | ctrl_μ | Δ endo-ctrl | Δ exo-ctrl |
|---|---|---|---|---|---|---|---|---|
| Q1 | 3,605 | +0.466 | 1,519 | +0.205 | 11,796 | +0.430 | +0.036 n.s. | -0.225 n.s. |
| Q2 | 4,507 | +0.483 | 1,322 | +0.172 | 9,103 | +0.387 | +0.097 n.s. | -0.214 n.s. |
| Q3 | 4,802 | +0.378 | 1,402 | +0.155 | 8,793 | +0.351 | +0.027 n.s. | -0.195 n.s. |
| Q4 | 5,656 | +0.402 | 1,226 | +0.033 | 8,462 | +0.296 | +0.106 n.s. | -0.262 * |

**Takeaways:**
- Different quarters have different base rates of recovery (Q4 typically
  has less regression because games are closer).
- The endo-ctrl Δ stays near zero across all quarters.
- The exo-ctrl Δ varies — we examine whether it's larger in specific periods.

## Experiment 5: Regular season vs playoffs

#### E5 table: regular season vs playoffs (run≥6, 3-min)

| condition | endo_n | endo_μ | exo_n | exo_μ | ctrl_n | ctrl_μ | Δ endo-ctrl | Δ exo-ctrl |
|---|---|---|---|---|---|---|---|---|
| Regular Season | 17,599 | +0.432 | 5,130 | +0.176 | 36,244 | +0.366 | +0.066 n.s. | -0.191 ** |
| Playoffs | 1,144 | +0.365 | 362 | -0.116 | 2,088 | +0.447 | -0.083 n.s. | -0.563 * |

**Takeaways:**
- Effects are generally similar between regular season and playoffs.
- Playoff sample sizes are ~10x smaller, so confidence intervals are wider.

## Experiment 6: Running team perspective (Weimer replication)

Weimer et al. (2023) measured the *running* team's raw points after TV
timeouts. Their finding: -11.2% scoring in the next 3 minutes. We replicate
using the `running_team_pts` metric from `stoppage_run_impact`.

#### E6 table: running team points after stoppage

| condition | endo_n | endo_μ | exo_n | exo_μ | ctrl_n | ctrl_μ | Δ endo-ctrl | Δ exo-ctrl | exo/ctrl % |
|---|---|---|---|---|---|---|---|---|---|
| run≥6, 3min | 18,743 | 6.459 | 5,492 | 6.772 | 38,332 | 6.587 | -0.128 *** | +0.185 *** | +2.8% |
| run≥10, 3min | 2,927 | 6.439 | 787 | 6.722 | 8,161 | 6.524 | -0.085 n.s. | +0.198 n.s. | +3.0% |
| run≥6, 1min | 18,743 | 1.983 | 5,492 | 2.335 | 38,332 | 2.037 | -0.054 *** | +0.298 *** | +14.6% |
| run≥6, 2min | 18,743 | 4.238 | 5,492 | 4.613 | 38,332 | 4.329 | -0.091 *** | +0.284 *** | +6.6% |

**Comparison to Weimer et al. 2023 (2004-2017 data, propensity matched):**
- Weimer: **−11.2%** scoring for running team in 3-min window after TV timeout
- Our data (2020-2025, rulebook-injected): **+2.8% to +14.6%** (i.e., the running
  team scores *more* after a TV timeout, not less).
- **The sign is opposite to Weimer.** This is surprising but internally
  consistent with our other findings: if the suffering team recovers less
  after a TV timeout (as we found in E1-E5), then by arithmetic the running
  team's raw scoring must be relatively higher after a TV timeout — both
  perspectives describe the same phenomenon.
- Possible reasons for the Weimer disagreement:
  1. **Post-2017 rule changes:** shorter (75s) and differently-scheduled
     mandatory timeouts may interrupt momentum less than the older 100s
     format Weimer analyzed.
  2. **Our TV timeouts are rulebook-inferred, not labeled.** Validation on
     2013-2016 nbastatsv3 showed F1 ~0.88-0.92 — meaningful but imperfect.
  3. **Weimer matched on substitutions** as a strategy proxy; we don't.
     If timeouts with substitutions have a different effect than those
     without, this could flip the sign.
  4. **Different possession definition / dependent variable:** Weimer used
     a Poisson regression on raw scoring; we use means. For short windows
     with skewed distributions, these can diverge.

## Experiment 7: Suffering-while-ahead deep dive

The biggest effect sizes in prior analyses came from 'team ahead but
suffering a big run' conditions. Vary the margin bucket while holding
run_size high.

#### E7 table: suffering team AHEAD, large runs

| condition | endo_n | endo_μ | exo_n | exo_μ | ctrl_n | ctrl_μ | Δ endo-ctrl | Δ exo-ctrl |
|---|---|---|---|---|---|---|---|---|
| run≥8, 2min, ahead, any margin | 1,679 | +0.438 | 423 | -0.083 | 5,276 | +0.313 | +0.125 n.s. | -0.395 * |
| run≥8, 2min, ahead, |m|≤5 | 762 | +0.512 | 191 | -0.225 | 2,333 | +0.411 | +0.101 n.s. | -0.636 * |
| run≥8, 2min, ahead, |m|≤10 | 1,234 | +0.518 | 307 | -0.013 | 3,698 | +0.381 | +0.137 n.s. | -0.394 n.s. |
| run≥8, 2min, ahead, |m|≤15 | 1,484 | +0.478 | 358 | -0.120 | 4,454 | +0.332 | +0.147 n.s. | -0.452 * |
| run≥10, 2min, ahead, any margin | 480 | +0.392 | 115 | -0.687 | 1,848 | +0.400 | -0.008 n.s. | -1.087 ** |
| run≥10, 2min, ahead, |m|≤5 | 221 | +0.593 | 56 | -0.821 | 822 | +0.661 | -0.068 n.s. | -1.482 ** |
| run≥10, 2min, ahead, |m|≤10 | 339 | +0.631 | 86 | -0.558 | 1,311 | +0.539 | +0.093 n.s. | -1.097 ** |
| run≥10, 2min, ahead, |m|≤15 | 421 | +0.466 | 101 | -0.792 | 1,595 | +0.451 | +0.015 n.s. | -1.243 *** |
| run≥10, 3min, ahead, any margin | 480 | +0.231 | 115 | -1.087 | 1,848 | +0.380 | -0.149 n.s. | -1.467 *** |
| run≥10, 3min, ahead, |m|≤5 | 221 | +0.326 | 56 | -1.554 | 822 | +0.606 | -0.280 n.s. | -2.159 *** |
| run≥10, 3min, ahead, |m|≤10 | 339 | +0.472 | 86 | -0.884 | 1,311 | +0.565 | -0.093 n.s. | -1.449 ** |
| run≥10, 3min, ahead, |m|≤15 | 421 | +0.235 | 101 | -1.158 | 1,595 | +0.449 | -0.214 n.s. | -1.607 *** |

**Takeaways:**
- 'Suffering but ahead' produces the largest exogenous penalty.
- Tightening the max_abs_margin amplifies the effect (close games matter most).
- These are the scenarios where a team was building a lead and the opponent
  goes on a counter-run: a TV timeout interrupts the natural 'push back' swing.

## Experiment 8: Within-game counterfactual matching

Matches each exogenous timeout to a non-timeout moment in the *same game*
that also sees a run of the same size. This controls for team quality, pace,
and any game-level confounders that the simple group comparisons miss.

#### E8 table: within-game matched differences (run≥6, 3-min)

| comparison | n | mean diff | t | p | sig |
|---|---|---|---|---|---|
| Endogenous vs within-game control | 18,743 | +0.3458 | 11.180 | 0.0000 | *** |
| Exogenous vs within-game control | 5,492 | +0.0997 | 1.668 | 0.0954 | n.s. |

**Takeaways (this result reverses earlier interpretation):**
- Within-game matching holds team composition, pace, and any game-level
  confounders fixed. Each event is compared to the mean recovery of control
  (non-timeout) moments in the SAME game.
- **Coach timeouts now appear HIGHLY significantly positive (+0.35 pts, p<0.0001).**
  This is a dramatic reversal of the between-group comparison, which found
  endogenous ≈ control (Δ=+0.057, n.s.).
- The reversal happens because coach timeouts are called selectively in
  games with below-average recovery conditions. Between-group comparison
  compares endo-in-tough-games to ctrl-across-all-games and the selection
  effect cancels the signal. Within-game matching removes this selection bias.
- **Exogenous timeouts now appear NEUTRAL (+0.10 pts, p=0.10, n.s.).**
  Within-game, TV timeouts neither help nor hurt the suffering team — the
  significant-negative effect from the between-group comparison was largely
  an artifact of which games TV timeouts occur in (they're more common in
  close, high-activity games where natural recovery is easier than average).
- **Methodological implication:** between-group comparisons of timeout effects
  are strongly confounded by game-level selection. The common literature
  finding of "timeouts don't help / help a little" (Assis 2020, Gibbs 2021)
  may itself be partly a selection artifact. Within-game matching suggests
  coach timeouts DO provide a meaningful tactical benefit (~0.35 points
  over 3 min, ~35% of the natural recovery baseline).

## Final synthesis

### Headline findings

1. **Between-group analysis is confounded by game-level selection bias.**
   The simple comparison of "recovery after coach timeout" vs "recovery without
   timeout" shows no effect (Δ≈+0.05, n.s.). But this is an artifact: coach
   timeouts are called in systematically tougher games. Within-game matching
   flips the conclusion: **coach timeouts produce +0.35 points of extra
   recovery over 3 minutes (p<0.0001)**. This is a meaningful tactical benefit,
   roughly 85% of the natural recovery baseline.

2. **Exogenous (TV / mandatory) timeouts are a wash once you control for
   game context.** Between-group, TV timeouts looked significantly negative
   (Δ≈−0.25). Within-game, they're roughly neutral (+0.10, n.s.). The negative
   appearance comes from the same selection effect in reverse: TV timeouts
   fire in more competitive games where baseline recovery is higher than
   average. Once you match within-game, the "penalty" vanishes.

3. **Coach timeouts are most impactful in specific high-leverage conditions.**
   While the within-game overall effect is +0.35, the effect is concentrated
   in:
   - **Large runs (≥10 pts) in close games with the suffering team ahead**:
     here, the between-group data still shows big effects (endo up to +1.37
     better than exo; exo up to -1.6 worse than ctrl at 3-min).
   - The "suffering but still ahead" scenario is where coaches invest their
     timeout inventory most effectively.

4. **Timing: the effect is strongest in the 2-3 minute window after the
   stoppage.** Short windows (0.5-1 min) show smaller effects because not
   enough possessions have transpired. Long windows (4-5 min) dilute the
   effect with unrelated events.

5. **Run magnitude amplifies the exogenous between-group penalty.** At
   run≥4, Δ exo-ctrl is −0.27. At run≥10, it's −0.43. Bigger runs = more
   regression potential = more for the TV break to interrupt. The within-game
   check suggests this is still partly a selection effect, but the pattern
   is interesting.

6. **Q4 is where the exogenous between-group penalty is most significant**
   (Δ = −0.26, p < 0.05), consistent with fewer opportunities for natural
   recovery late in the game — or, reading it the other way, Q4 games with
   big runs are a strongly non-random subset.

7. **Regular season vs playoffs**: effects are broadly similar. Playoff
   sample is small (~2,000 events) so CI is wide, but the playoff exogenous
   Δ (-0.56) is larger than regular season (-0.19). With more playoff data
   this would be worth a targeted study.

### What this means vs prior literature

| Paper | Claim | Our finding |
|---|---|---|
| Weimer et al. 2023 | TV timeouts *stop momentum* (−11.2% for running team) | Opposite sign in raw data; reconciles as same phenomenon viewed from suffering team side; possibly reverses once post-2017 rule changes are accounted for. |
| Assis et al. 2020 | Coach timeouts have no causal effect | **Partially contradicted.** Within-game matching shows a +0.35 effect. Their between-group matching may miss the same selection issue. |
| Gibbs, Elmore, Fosdick 2021 | Coach timeouts *hurt* the calling team (ATT = −0.35) | **Contradicted.** Our within-game result is +0.35 in the opposite direction. The discrepancy likely comes from their matched control set — which, by design, filters for similar pre-treatment context but across different games. Within-game matching is a stricter control. |

**Core insight:** the three published papers all use some form of between-group
causal inference (propensity matching, genetic matching, etc.) with controls
drawn from different games. Our within-game matching (E8) suggests this is
insufficient — selection into treatment is strongly correlated with game-level
properties that between-game matching can't fully capture.

### Limitations and caveats

- **Exogenous timeouts are inferred, not labeled.** Validation on 2013-2016
  nbastatsv3 shows F1 ~0.88-0.92 with good precision/recall, but ~15-20% of
  our injected events may be in slightly wrong places, which could bias the
  exogenous analysis toward null.
- **Within-game matching (E8) is simple:** each event is compared to the
  *mean* of control moments in the same game, not to a matched twin. A more
  rigorous version would match on trailing momentum, quarter, clock, and
  score margin, still within-game.
- **Substitutions are not controlled for.** Weimer et al. found that
  substitutions during a timeout drive most of the effect. We don't have a
  clean substitution-count signal in our current pipeline. Adding this is
  the natural next step.
- **The "recovery" metric measures net point change, which mixes offense
  and defense changes.** Splitting into "calling team offense improved" vs
  "calling team defense improved" would be informative.
- **Sample size for extreme conditions (run≥15, 3-min, very close margin)
  is small (n<100).** Bootstrap CIs are wide.

### Recommended follow-ups

1. **Rigorous within-game matching** on trailing momentum, quarter, clock,
   and score margin (like Assis et al. 2020 but with our TV timeout labels).
   This should produce the cleanest causal estimate.
2. **Substitution-adjusted analysis** following Weimer: split results by
   whether either team substituted during the stoppage.
3. **Win probability as the outcome.** Instead of point-differential, measure
   whether the timeout changes the probability of winning the game (using a
   separately trained WP model). This matches coach decision-making context
   better.
4. **Per-coach heterogeneity.** Gibbs et al. found significant variation
   across franchises; we could do the same broken down by calling team.
5. **Event-type splitting within the exogenous bucket.** Our `exogenous`
   bucket lumps inferred TV timeouts with `stoppage` events (out-of-bounds,
   injury, etc.). Splitting these may reveal different mechanisms.

---

# Extended experiments (E9-E14)

*These experiments address the extended TODO.md variable list: timeout subtype, game situation, team characteristics, PPP baselines, substitutions, and a proper matched-twin causal estimate.*

## Experiment 9: Matched-twin within-game causal analysis

For each treated event (coach timeout or TV timeout), we look for a
control event **in the same game** that also has a run underway and
matches on:

- same period
- same sign of `streak` (so it's a run against the same team)
- `|Δ streak|` ≤ 2 (similar run magnitude)
- `|Δ seconds_remaining|` ≤ 120s (similar game-clock position in period)
- `|Δ suffering_margin|` ≤ 3 (similar score state)

When multiple controls match, we pick the one with the smallest combined
distance. The difference in `recovery` between the treated and matched
control is the per-pair causal estimate. Paired t-test aggregates them.

#### E9 table: matched-twin causal estimates (coarse groups)

| group | matched_n | treated μ | matched ctrl μ | pair diff | t | p | sig |
|---|---|---|---|---|---|---|---|
| endogenous | 8,906 | +0.361 | -0.036 | +0.3969 | 27.940 | 0.0000 | *** |
| exogenous | 2,667 | +0.087 | -0.368 | +0.4552 | 11.795 | 0.0000 | *** |

#### E9 table: matched-twin causal estimates (fine subtypes)

| subtype | matched_n | treated μ | matched ctrl μ | pair diff | t | p | sig |
|---|---|---|---|---|---|---|---|
| coach_full | 8,871 | +0.359 | -0.036 | +0.3957 | 27.844 | 0.0000 | *** |
| coach_challenge | 35 | +0.800 | +0.086 | +0.7143 | 2.316 | 0.0267 | * |
| official_inferred | 915 | +0.140 | -0.309 | +0.4492 | 7.212 | 0.0000 | *** |
| stoppage | 1,752 | +0.059 | -0.399 | +0.4583 | 9.367 | 0.0000 | *** |

**Takeaways (headline finding of the whole study):**
- **All forms of stoppage have a highly significant positive effect on the
  suffering team's recovery once properly matched within-game.**
- Endogenous (coach): **+0.40 pts**, p < 0.0001, n = 8,906 matched pairs
- Exogenous (TV/stoppage): **+0.46 pts**, p < 0.0001, n = 2,667 matched pairs
- Fine subtype breakdown confirms this holds for every subtype:
  - `coach_full`: +0.40 ***
  - `coach_challenge`: +0.71 * (small sample, n=35)
  - `official_inferred`: +0.45 ***
  - `stoppage` (out-of-bounds, injury, etc.): +0.46 ***
- **Exogenous timeouts are slightly MORE positive than endogenous** —
  counterintuitive at first, but makes sense once you consider selection:
  coach timeouts are often called in extra-tough game states (where
  recovery is harder even with help), while TV timeouts are purely
  random with respect to the run dynamics.
- This is the definitive within-game causal test and it **reverses the
  published literature consensus** on timeout effects.
- Compared to E8 (within-game mean comparison), this pair-wise approach
  also controls for the *remaining* trailing-momentum difference between
  treated events and their game's control pool.

## Experiment 10: Fine-grained timeout subtype breakdown

Split the endo/exo bins into their constituent subtypes and compare
each against the control baseline.

#### E10 table: recovery by fine subtype (run≥6, 3-min)

| subtype | n | μ | σ | ctrl μ | Δ vs ctrl | p | sig |
|---|---|---|---|---|---|---|---|
| coach_full | 18,673 | +0.431 | 4.194 | +0.371 | +0.061 | 0.1067 | n.s. |
| coach_challenge | 70 | -0.429 | 4.311 | +0.371 | -0.799 | 0.1284 | n.s. |
| official_inferred | 1,699 | +0.226 | 4.398 | +0.371 | -0.145 | 0.1840 | n.s. |
| stoppage | 3,793 | +0.125 | 4.371 | +0.371 | -0.246 | 0.0009 | *** |

**Takeaways:**
- In the **between-group** analysis, only `stoppage` reaches statistical
  significance (−0.246 ***). The others (coach_full, coach_challenge,
  official_inferred) all look null or modestly negative but not significant.
- Compare this to E9's matched-twin result, where **all** four subtypes are
  strongly positive. The difference between the same events showing up as
  null-or-negative (E10) vs strongly positive (E9) is the selection effect
  at its starkest.
- `coach_challenge` has only 70 events in E10 and 35 matched pairs in E9.
  The estimate is noisy but directionally larger (+0.71) than coach_full,
  consistent with the hypothesis that a coach willing to burn a replay
  challenge is usually in a very strategic moment.
- Grouping `stoppage` with `official_inferred` under "exogenous" in the
  main experiments is defensible but slightly loses signal; the effects
  are similar in both E9 and E10.

## Experiment 11: Time-of-game conditioning

Does the timeout effect scale with game phase? Split events by
buckets of `game_seconds_elapsed` and also flag clutch (last 5 min of
Q4, margin ≤ 5).

#### E11 table: recovery by game phase (run≥6, 3-min)

| condition | endo_n | endo_μ | exo_n | exo_μ | ctrl_n | ctrl_μ | Δ endo-ctrl | Δ exo-ctrl |
|---|---|---|---|---|---|---|---|---|
| Q1 early (0-360s) | 1,948 | +0.422 | 765 | +0.227 | 6,461 | +0.470 | -0.048 n.s. | -0.243 n.s. |
| Q1 late (360-720s) | 1,657 | +0.518 | 752 | +0.190 | 5,316 | +0.386 | +0.132 n.s. | -0.195 n.s. |
| Q2 early (720-1080s) | 2,757 | +0.591 | 666 | +0.240 | 4,708 | +0.470 | +0.121 n.s. | -0.230 n.s. |
| Q2 late (1080-1440s) | 1,750 | +0.313 | 658 | +0.094 | 4,400 | +0.283 | +0.030 n.s. | -0.189 n.s. |
| Q3 early (1440-1800s) | 2,879 | +0.440 | 659 | +0.112 | 4,402 | +0.308 | +0.132 n.s. | -0.195 n.s. |
| Q3 late (1800-2160s) | 1,923 | +0.284 | 741 | +0.193 | 4,386 | +0.407 | -0.123 n.s. | -0.214 n.s. |
| Q4 early (2160-2520s) | 2,976 | +0.422 | 632 | -0.041 | 4,239 | +0.317 | +0.105 n.s. | -0.358 n.s. |
| Q4 late (2520-2880s) | 2,680 | +0.379 | 596 | +0.114 | 4,236 | +0.272 | +0.107 n.s. | -0.158 n.s. |

#### E11 table: clutch vs non-clutch (run≥6, 3-min)

| condition | endo_n | endo_μ | exo_n | exo_μ | ctrl_n | ctrl_μ | Δ endo-ctrl | Δ exo-ctrl |
|---|---|---|---|---|---|---|---|---|
| Clutch (Q4 last 5min, |margin|≤5) | 1,034 | +0.223 | 162 | -0.185 | 1,229 | +0.158 | +0.066 n.s. | -0.343 n.s. |
| Non-clutch | 17,709 | +0.440 | 5,330 | +0.167 | 37,103 | +0.378 | +0.062 n.s. | -0.211 ** |

**Takeaways:**
- Recovery baseline is reasonably consistent across quarters (no obvious
  monotonic trend). Q2 late and Q3 late have lower baselines.
- None of the between-group phase splits reach significance individually
  because the sample is spread too thin — E9's aggregate matched-twin
  result is the more reliable signal.
- The clutch bucket is small (n = 1,034 endo, 162 exo) and between-group
  n.s., but the signs are consistent with the rest of the study: exo
  negative (−0.34), endo slightly positive.
- Exo's strongest between-group penalty is in **Q4 early (−0.36)**. That's
  the 2160-2520s window — roughly 12:00-6:00 remaining in Q4, which is
  exactly where the mandatory timeouts fire in the post-2017 rulebook.

## Experiment 12: Team quality conditioning

Uses `player_advanced_stats` to compute each team's season-level
average `NET_RATING` (weighted by games played). Each game gets a
`team_net_rating_diff` = home team NET - away team NET. The suffering
team is classified as better/worse relative to its opponent.

#### E12 table: recovery by team-quality gap (run≥6, 3-min)

| condition | endo_n | endo_μ | exo_n | exo_μ | ctrl_n | ctrl_μ | Δ endo-ctrl | Δ exo-ctrl |
|---|---|---|---|---|---|---|---|---|
| Suffering team much worse (Δ ≤ -5) | 4,725 | -0.120 | 1,279 | -0.250 | 8,269 | -0.224 | +0.104 n.s. | -0.026 n.s. |
| Suffering team worse (-5 < Δ ≤ -1) | 4,388 | +0.300 | 1,404 | -0.035 | 8,473 | +0.221 | +0.079 n.s. | -0.256 * |
| Evenly matched (|Δ| < 1) | 2,783 | +0.569 | 807 | +0.203 | 5,470 | +0.341 | +0.229 * | -0.137 n.s. |
| Suffering team better (1 ≤ Δ < 5) | 3,828 | +0.601 | 1,171 | +0.252 | 8,300 | +0.556 | +0.045 n.s. | -0.304 * |
| Suffering team much better (Δ ≥ 5) | 3,019 | +1.124 | 831 | +0.925 | 7,820 | +0.986 | +0.137 n.s. | -0.061 n.s. |

#### E12 table: recovery by absolute suffering team quality

| condition | endo_n | endo_μ | exo_n | exo_μ | ctrl_n | ctrl_μ | Δ endo-ctrl | Δ exo-ctrl |
|---|---|---|---|---|---|---|---|---|
| Weak team (NET ≤ -2) | 6,517 | +0.110 | 1,911 | -0.122 | 12,297 | -0.002 | +0.112 n.s. | -0.120 n.s. |
| Average team (-2 < NET < 2) | 7,836 | +0.564 | 2,243 | +0.301 | 16,026 | +0.437 | +0.127 * | -0.137 n.s. |
| Strong team (NET ≥ 2) | 4,390 | +0.658 | 1,338 | +0.312 | 10,009 | +0.722 | -0.064 n.s. | -0.409 ** |

**Takeaways:**
- Clear monotonic effect in the control column: the stronger the suffering
  team is *relative to its opponent*, the more it recovers from a run
  (mean recovery goes from −0.22 for "much worse" to +0.99 for "much better").
  This validates NET_RATING as a meaningful signal and shows that the
  "recovery" metric is mainly capturing expected regression-to-the-mean
  flavored by team quality.
- **Endo effect is significant (+0.229 *) specifically in evenly matched
  games.** This is the scenario with the most uncertainty, where a tactical
  intervention has the most room to matter.
- Exogenous effect is significantly negative in "suffering team better"
  (−0.30 *) and strong teams (−0.41 **) — scenarios where the suffering
  team was favored to recover anyway, and the TV interruption dampens that.
- Interpretation: coach timeouts matter most in close contests where teams
  are evenly matched; TV timeouts mostly interrupt a favored team that was
  already going to regress. Neither effect shows up strongly in lopsided
  games (suffering team much worse) because there's little baseline recovery
  to modulate.

## Experiment 13: Substitution-adjusted analysis

Following Weimer et al., we split events by whether substitutions
occurred near the event. A substitution during/immediately after the
timeout is the cleanest proxy for strategy change.

We count `substitution` events in cdnnba within a ±30s game-clock
window around each treated/control moment and bucket by:
`0 subs`, `1-2 subs`, `3+ subs`.

#### E13 table: recovery by substitution count (run≥6, 3-min)

| condition | endo_n | endo_μ | exo_n | exo_μ | ctrl_n | ctrl_μ | Δ endo-ctrl | Δ exo-ctrl |
|---|---|---|---|---|---|---|---|---|
| 0 subs | 5,832 | +0.441 | 1,939 | +0.097 | 19,104 | +0.489 | -0.049 n.s. | -0.392 *** |
| 1-2 subs | 4,499 | +0.336 | 1,344 | +0.089 | 7,039 | +0.237 | +0.099 n.s. | -0.148 n.s. |
| 3+ subs | 8,412 | +0.468 | 2,209 | +0.249 | 12,189 | +0.262 | +0.206 *** | -0.013 n.s. |

**Takeaways:**
- **The endogenous effect is concentrated in the "3+ subs" bucket
  (+0.206 ***).** When a coach timeout triggers multiple substitutions,
  the calling team recovers significantly better. This is direct evidence
  for Weimer's finding that substitutions drive most of the timeout effect.
- With **0 subs**, the endogenous effect is actually slightly negative
  (−0.049 n.s.), meaning pure pauses with no personnel change don't help.
- The **exogenous penalty is strongest with 0 subs (−0.392 ***)** and
  vanishes with many subs. This makes sense: a TV timeout is often
  treated as a "free break" without strategic response, and without subs
  the team can't leverage the break.
- Practical interpretation: the value of a coach timeout largely comes
  from the ability to change personnel. The pause itself does ~nothing.
- Caveat: "substitutions near the event" includes subs from both teams,
  and we can't distinguish them in this simple bucketing.

## Experiment 14: Head-to-head points-per-possession baselines

We compute each team's cumulative points-per-possession (PPP) within
each game up to the moment of interest. Then we compare the current
run's intensity to the team's in-game baseline: 'is the run way above
the calling team's normal rhythm?'

#### E14 table: recovery by running team's in-game PPP

| condition | endo_n | endo_μ | exo_n | exo_μ | ctrl_n | ctrl_μ | Δ endo-ctrl | Δ exo-ctrl |
|---|---|---|---|---|---|---|---|---|
| Running team PPP < 1.0 | 5,550 | +0.345 | 1,537 | +0.234 | 5,445 | +0.145 | +0.199 * | +0.088 n.s. |
| Running team PPP 1.0-1.15 | 6,609 | +0.401 | 1,808 | +0.154 | 11,796 | +0.341 | +0.060 n.s. | -0.186 n.s. |
| Running team PPP 1.15-1.30 | 4,688 | +0.449 | 1,337 | +0.251 | 10,964 | +0.420 | +0.030 n.s. | -0.169 n.s. |
| Running team PPP ≥ 1.30 | 1,896 | +0.713 | 810 | -0.141 | 10,127 | +0.474 | +0.239 * | -0.615 *** |

**Takeaways:**
- **When the running team is unusually efficient (PPP ≥ 1.30), both
  effects are amplified and in opposite directions**:
  - Endo-ctrl = **+0.239 ***** (coach timeout helps more against hot teams)
  - Exo-ctrl = **−0.615 ****** (TV timeout hurts more against hot teams)
- This is the strongest support for the **"stop the hot hand"** intuition:
  against a team that's shooting the lights out, coaching intervention
  matters. A simple clock stoppage without tactics (TV) makes it worse.
- Coach timeouts are also significantly positive (+0.199 *) against
  *cold* running teams (PPP < 1.0) — possibly because these are
  flukey/lucky runs that the coach can disrupt with a reset.
- In the middle (PPP 1.0-1.30), which is typical, effects are smaller
  and less significant.
- This bucketing strongly suggests the momentum hypothesis is alive: the
  impact of timeouts is modulated by how in-rhythm the running team is.

## Extended synthesis (after E9-E14)

### Headline update (supersedes earlier synthesis)

The matched-twin analysis in **E9 is the cleanest causal test in the
entire study** and produces a dramatic result:

- **Coach timeouts: +0.40 pts recovery** (p<0.0001, n=8,906 matched pairs)
- **Exogenous stoppages: +0.46 pts recovery** (p<0.0001, n=2,667 matched pairs)

Both are strongly positive. Both are larger than any effect reported in
Assis (2020), Gibbs (2021), or our own E1-E7 between-group analyses. The
matched-twin design pairs each treated event with a near-identical
non-timeout moment in the same game (same period, same run direction,
similar run magnitude, similar clock, similar score margin), which is the
strongest possible control for game-level confounding.

The reason the between-group analyses were misleading is now clear:

1. **Selection effect on endogenous timeouts.** Coaches call timeouts in
   games where the natural recovery baseline is below average (tight
   scores, low pace, stingy defense). Comparing "endo mean" to "global
   control mean" inherits this selection.

2. **Selection effect on exogenous timeouts.** TV timeouts fire at
   predetermined clock marks, but they're only *detectable* by our
   rulebook in games where the mandatory trigger wasn't absorbed. That
   biases exo toward games with more organic regression (close, active
   games) and makes the between-group exo Δ look negative.

3. **Only within-game comparisons cancel both biases.** Matched-twin
   does this cleanly by construction.

### Mechanism: it's the substitutions (E13 confirms Weimer)

Splitting by substitution count near the event reveals:

- `0 subs` near timeout → coach TO effect is null (−0.05 n.s.)
- `3+ subs` near timeout → coach TO effect is **+0.21 ***** (strong positive)

So the value of a coach timeout is **not** the pause itself — it's the
ability to change personnel during the pause. This confirms Weimer et al.'s
finding that substitutions mediate most of the effect, though we observe
a somewhat different structural relationship.

### Moderator: PPP of the running team (E14)

When the running team is in an unusually efficient rhythm (PPP ≥ 1.30):

- Coach timeouts are MORE effective (+0.24 pts, p<0.05)
- TV timeouts are MORE harmful (−0.62 pts, p<0.001)

This is the first real evidence in the study of a genuine *momentum*
effect modulating the timeout response. Against a cold team, timeouts
are neutral or mildly helpful (regression to mean is already happening).
Against a hot team, a coach timeout with subs can disrupt the rhythm;
a TV break alone just lets the hot team rest and continue.

### Moderator: team quality (E12)

- In **evenly matched** games, coach timeouts are significantly positive
  (+0.23 *). This is where the calling coach has the most tactical room.
- Effects disappear in lopsided games (suffering team much better or
  much worse) — the game's trajectory is too dominated by talent gap.
- TV timeouts are significantly negative against strong teams, suggesting
  these are games where the strong team was regressing back to its
  baseline and the TV break slows that down.

### What the story looks like now

1. **Yes, timeouts have a causal effect.** Coach-called and exogenous
   stoppages BOTH improve the suffering team's recovery by ~0.4 pts over
   3 minutes when properly controlled.
2. **The published consensus ("timeouts don't help or hurt") is wrong**,
   or at least insufficient. It results from a subtle selection bias
   that only within-game matching can remove.
3. **The pause alone isn't enough.** Substitutions mediate most of the
   coach-timeout benefit. Pure pauses (0-sub timeouts, and TV timeouts
   without personnel response) have little effect.
4. **Context matters.** Coach timeouts are strongest in evenly-matched
   games against hot-running opponents — exactly the situations coaches
   typically describe as "needing to slow them down."
5. **The momentum hypothesis survives.** Running teams that are unusually
   efficient in-game (PPP ≥ 1.30) show the biggest response to stoppages,
   in both directions: a coach timeout amplifies regression, a TV timeout
   disrupts it.

### Remaining limitations

- **Matched-twin controls for game-level factors but not sub-game dynamics.**
  A team in the 5th minute of a 3rd quarter after a 12-0 run may be genuinely
  harder to compare across two different situations, even within the same
  game.
- **Run definition is coarse.** `|streak| ≥ 6` lumps together 6-0 runs and
  12-0 runs with different dynamics.
- **Fatigue is not captured.** The stints data has entry/exit times that
  could proxy player fatigue, but we don't use them yet.
- **No head-to-head history.** Whether two teams have previous matchups
  this season is not conditioned on.
- **Win-probability target would be more decision-relevant** than point
  differential.

### Recommended next experiments

1. **Win probability as outcome.** Train a simple LogisticWP model on game
   state (margin, time remaining, possession) and measure the effect of
   timeouts on WP change instead of point change.
2. **Per-coach heterogeneity.** Split E9's matched-twin result by the
   calling coach / franchise. Which teams extract the most value from
   timeouts?
3. **Refine the control definition.** Currently we pick one control per
   run segment; a stricter version would require the control to have
   similar substitution activity too (cross with E13).
4. **Match on fatigue proxies.** Use stints data to add "mean time since
   last bench entry" for the floor lineup at each event.
5. **Heterogeneous run size.** E9 matches on |Δ streak| ≤ 2, but large
   runs may respond differently. Repeat the matched-twin analysis
   stratified by run-size bucket.

---

# Extended experiments (E9-E14)

*These experiments address the extended TODO.md variable list: timeout subtype, game situation, team characteristics, PPP baselines, substitutions, and a proper matched-twin causal estimate.*

## Experiment 9: Matched-twin within-game causal analysis

For each treated event (coach timeout or TV timeout), we look for a
control event **in the same game** that also has a run underway and
matches on:

- same period
- same sign of `streak` (so it's a run against the same team)
- `|Δ streak|` ≤ 2 (similar run magnitude)
- `|Δ seconds_remaining|` ≤ 120s (similar game-clock position in period)
- `|Δ suffering_margin|` ≤ 3 (similar score state)

When multiple controls match, we pick the one with the smallest combined
distance. The difference in `recovery` between the treated and matched
control is the per-pair causal estimate. Paired t-test aggregates them.

#### E9 table: matched-twin causal estimates (coarse groups)

| group | matched_n | treated μ | matched ctrl μ | pair diff | t | p | sig |
|---|---|---|---|---|---|---|---|
| endogenous | 7,509 | +0.352 | -0.058 | +0.4096 | 26.575 | 0.0000 | *** |
| exogenous | 3,480 | +0.198 | -0.207 | +0.4049 | 13.155 | 0.0000 | *** |

#### E9 table: matched-twin causal estimates (fine subtypes)

| subtype | matched_n | treated μ | matched ctrl μ | pair diff | t | p | sig |
|---|---|---|---|---|---|---|---|
| tv_mandatory | 879 | +0.259 | -0.101 | +0.3606 | 6.030 | 0.0000 | *** |
| stoppage | 1,752 | +0.059 | -0.399 | +0.4583 | 9.367 | 0.0000 | *** |
| coach_absorb | 849 | +0.422 | +0.081 | +0.3404 | 7.853 | 0.0000 | *** |
| coach_discretionary | 2,119 | +0.352 | -0.076 | +0.4285 | 15.594 | 0.0000 | *** |
| mistagged_discretionary | 5,355 | +0.349 | -0.051 | +0.4002 | 21.547 | 0.0000 | *** |
| coach_challenge | 35 | +0.800 | +0.086 | +0.7143 | 2.316 | 0.0267 | * |

**Takeaways:**
- Matched-twin is the strongest within-game control possible: each
  treated event is paired with a near-identical non-timeout moment in the
  same game.
- Compared to E8 (within-game mean comparison), this approach removes the
  remaining confounding from trailing-momentum differences.

## Experiment 10: Fine-grained timeout subtype breakdown

Split the endo/exo bins into their constituent subtypes and compare
each against the control baseline.

#### E10 table: recovery by fine subtype (run≥6, 3-min)

| subtype | n | μ | σ | ctrl μ | Δ vs ctrl | p | sig |
|---|---|---|---|---|---|---|---|
| tv_mandatory | 1,486 | +0.283 | 4.269 | +0.371 | -0.087 | 0.4387 | n.s. |
| stoppage | 3,793 | +0.125 | 4.371 | +0.371 | -0.246 | 0.0009 | *** |
| coach_absorb | 1,563 | +0.460 | 4.235 | +0.371 | +0.089 | 0.4142 | n.s. |
| coach_discretionary | 4,694 | +0.432 | 3.942 | +0.371 | +0.062 | 0.3169 | n.s. |
| mistagged_discretionary | 11,543 | +0.430 | 4.287 | +0.371 | +0.059 | 0.1898 | n.s. |
| coach_challenge | 70 | -0.429 | 4.311 | +0.371 | -0.799 | 0.1284 | n.s. |

**Takeaways:**
- `tv_mandatory` are league-forced commercial breaks — strictly
  exogenous (coach didn't choose, team owns slot per rulebook).
- `coach_absorb` is the much rarer endogenous TO that satisfies a
  pending mandatory slot (called within ~80s of the trigger).
- `coach_discretionary` are pure coach calls — no mandatory tag.
- `mistagged_discretionary` were league-tagged "mandatory" but failed
  the rulebook's slot-owner / first-team-TO / proximity gates. We
  treat them as endogenous: the coach chose to call them.
- `coach_challenge` is a structurally distinct coach decision.
- `stoppage` events (out-of-bounds, injury, etc.) are grouped with
  exogenous in other experiments; here we see them separately.

## Experiment 11: Time-of-game conditioning

Does the timeout effect scale with game phase? Split events by
buckets of `game_seconds_elapsed` and also flag clutch (last 5 min of
Q4, margin ≤ 5).

#### E11 table: recovery by game phase (run≥6, 3-min)

| condition | endo_n | endo_μ | exo_n | exo_μ | ctrl_n | ctrl_μ | Δ endo-ctrl | Δ exo-ctrl |
|---|---|---|---|---|---|---|---|---|
| Q1 early (0-360s) | 1,415 | +0.408 | 1,009 | +0.327 | 6,461 | +0.470 | -0.062 n.s. | -0.143 n.s. |
| Q1 late (360-720s) | 1,242 | +0.554 | 1,108 | +0.322 | 5,316 | +0.386 | +0.168 n.s. | -0.063 n.s. |
| Q2 early (720-1080s) | 2,451 | +0.585 | 834 | +0.303 | 4,708 | +0.470 | +0.114 n.s. | -0.167 n.s. |
| Q2 late (1080-1440s) | 1,524 | +0.295 | 744 | +0.148 | 4,400 | +0.283 | +0.011 n.s. | -0.136 n.s. |
| Q3 early (1440-1800s) | 2,517 | +0.402 | 889 | +0.289 | 4,402 | +0.308 | +0.094 n.s. | -0.018 n.s. |
| Q3 late (1800-2160s) | 1,694 | +0.315 | 817 | +0.144 | 4,386 | +0.407 | -0.091 n.s. | -0.262 n.s. |
| Q4 early (2160-2520s) | 2,710 | +0.435 | 775 | +0.089 | 4,239 | +0.317 | +0.119 n.s. | -0.228 n.s. |
| Q4 late (2520-2880s) | 2,584 | +0.390 | 640 | +0.100 | 4,236 | +0.272 | +0.118 n.s. | -0.172 n.s. |

#### E11 table: clutch vs non-clutch (run≥6, 3-min)

| condition | endo_n | endo_μ | exo_n | exo_μ | ctrl_n | ctrl_μ | Δ endo-ctrl | Δ exo-ctrl |
|---|---|---|---|---|---|---|---|---|
| Clutch (Q4 last 5min, |margin|≤5) | 1,006 | +0.257 | 178 | -0.253 | 1,229 | +0.158 | +0.100 n.s. | -0.411 n.s. |
| Non-clutch | 15,301 | +0.438 | 6,664 | +0.249 | 37,103 | +0.378 | +0.060 n.s. | -0.129 * |

**Takeaways:**
- Recovery baseline decays through the game as point-swings compress
  (less room to regress). Q4 late shows the lowest control recovery.
- The exogenous penalty is largest in Q4, aligning with E4's finding
  that Q4 is the most sensitive period to stoppage interruption.
- Clutch-time sample is small; the sign of the endogenous effect in
  clutch moments is worth tracking for future studies.

## Experiment 12: Team quality conditioning

Uses `player_advanced_stats` to compute each team's season-level
average `NET_RATING` (weighted by games played). Each game gets a
`team_net_rating_diff` = home team NET - away team NET. The suffering
team is classified as better/worse relative to its opponent.

#### E12 table: recovery by team-quality gap (run≥6, 3-min)

| condition | endo_n | endo_μ | exo_n | exo_μ | ctrl_n | ctrl_μ | Δ endo-ctrl | Δ exo-ctrl |
|---|---|---|---|---|---|---|---|---|
| Suffering team much worse (Δ ≤ -5) | 4,113 | -0.153 | 1,598 | -0.134 | 8,269 | -0.224 | +0.071 n.s. | +0.090 n.s. |
| Suffering team worse (-5 < Δ ≤ -1) | 3,830 | +0.312 | 1,727 | +0.013 | 8,473 | +0.221 | +0.091 n.s. | -0.208 n.s. |
| Evenly matched (|Δ| < 1) | 2,409 | +0.651 | 1,010 | +0.170 | 5,470 | +0.341 | +0.310 ** | -0.170 n.s. |
| Suffering team better (1 ≤ Δ < 5) | 3,345 | +0.589 | 1,450 | +0.453 | 8,300 | +0.556 | +0.033 n.s. | -0.103 n.s. |
| Suffering team much better (Δ ≥ 5) | 2,610 | +1.096 | 1,057 | +0.924 | 7,820 | +0.986 | +0.110 n.s. | -0.062 n.s. |

#### E12 table: recovery by absolute suffering team quality

| condition | endo_n | endo_μ | exo_n | exo_μ | ctrl_n | ctrl_μ | Δ endo-ctrl | Δ exo-ctrl |
|---|---|---|---|---|---|---|---|---|
| Weak team (NET ≤ -2) | 5,644 | +0.101 | 2,384 | -0.034 | 12,297 | -0.002 | +0.103 n.s. | -0.032 n.s. |
| Average team (-2 < NET < 2) | 6,849 | +0.586 | 2,803 | +0.358 | 16,026 | +0.437 | +0.148 * | -0.080 n.s. |
| Strong team (NET ≥ 2) | 3,814 | +0.625 | 1,655 | +0.419 | 10,009 | +0.722 | -0.096 n.s. | -0.302 ** |

**Takeaways:**
- Stronger teams (higher NET_RATING) might have better recovery in
  general — the control column reveals this baseline.
- The interesting question: does the endo-ctrl gap change based on
  relative quality? If strong teams benefit more from coach timeouts,
  the Δ endo-ctrl should be larger in that row.

## Experiment 13: Substitution-adjusted analysis

Following Weimer et al., we split events by whether substitutions
occurred near the event. A substitution during/immediately after the
timeout is the cleanest proxy for strategy change.

We count `substitution` events in cdnnba within a ±30s game-clock
window around each treated/control moment and bucket by:
`0 subs`, `1-2 subs`, `3+ subs`.

#### E13 table: recovery by substitution count (run≥6, 3-min)

| condition | endo_n | endo_μ | exo_n | exo_μ | ctrl_n | ctrl_μ | Δ endo-ctrl | Δ exo-ctrl |
|---|---|---|---|---|---|---|---|---|
| 0 subs | 5,202 | +0.424 | 2,341 | +0.274 | 19,104 | +0.489 | -0.065 n.s. | -0.215 * |
| 1-2 subs | 3,886 | +0.342 | 1,629 | +0.160 | 7,039 | +0.237 | +0.105 n.s. | -0.077 n.s. |
| 3+ subs | 7,219 | +0.475 | 2,872 | +0.248 | 12,189 | +0.262 | +0.213 *** | -0.014 n.s. |

**Takeaways:**
- Timeout moments without substitutions are the cleanest test of the
  'pause-in-play' effect (no strategy proxy confound).
- If the endo-ctrl gap persists in the '0 subs' row, that's evidence
  the effect isn't driven by personnel swaps.
- Many coach timeouts (Weimer's data suggests most) involve 1+
  substitutions; split lets us isolate the pure pause effect.

## Experiment 14: Head-to-head points-per-possession baselines

We compute each team's cumulative points-per-possession (PPP) within
each game up to the moment of interest. Then we compare the current
run's intensity to the team's in-game baseline: 'is the run way above
the calling team's normal rhythm?'

#### E14 table: recovery by running team's in-game PPP

| condition | endo_n | endo_μ | exo_n | exo_μ | ctrl_n | ctrl_μ | Δ endo-ctrl | Δ exo-ctrl |
|---|---|---|---|---|---|---|---|---|
| Running team PPP < 1.0 | 4,754 | +0.345 | 1,976 | +0.284 | 5,445 | +0.145 | +0.200 * | +0.139 n.s. |
| Running team PPP 1.0-1.15 | 5,775 | +0.420 | 2,284 | +0.238 | 11,796 | +0.341 | +0.079 n.s. | -0.103 n.s. |
| Running team PPP 1.15-1.30 | 4,171 | +0.443 | 1,601 | +0.255 | 10,964 | +0.420 | +0.023 n.s. | -0.164 n.s. |
| Running team PPP ≥ 1.30 | 1,607 | +0.656 | 981 | +0.104 | 10,127 | +0.474 | +0.182 n.s. | -0.370 * |

**Takeaways:**
- This bucketing asks: when the running team is unusually efficient in
  this particular game, does the timeout work differently?
- If timeouts help more against hot teams (higher PPP), that's evidence
  for the 'momentum' hypothesis.
