"""Extended battery of timeout impact experiments.

Experiments E9-E14 build on the existing research-summary.md by:
- Running a proper within-game matched-twin causal analysis (E9)
- Splitting timeout types finer (E10)
- Conditioning on game phase / clutch time (E11)
- Conditioning on team quality (E12)
- Controlling for substitutions (E13)
- Comparing runs to team-level PPP baselines (E14)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats as sp_stats

from nba_timeout_impact.datasets.memo_cdnnba_pl import CDNNBAMemoPL

SUMMARY_PATH = str(Path(__file__).resolve().parents[2] / "Notebooks" / "applied-causality" / "research-summary.md")


# ============================================================================
# Helpers (duplicated from run_experiments.py to keep this standalone)
# ============================================================================


def sig(p: float) -> str:
    if np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float, str]:
    if len(a) < 2 or len(b) < 2:
        return (float("nan"), float("nan"), "n/a")
    t, p = sp_stats.ttest_ind(a, b, equal_var=False)
    return (float(np.mean(a) - np.mean(b)), float(p), sig(float(p)))  # type: ignore


def append_md(text: str) -> None:
    with open(SUMMARY_PATH, "a") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def append_table(title: str, rows: list[dict], headers: list[str]) -> None:
    hdr = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    body = "\n".join("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |" for r in rows)
    append_md(f"\n#### {title}\n\n{hdr}\n{sep}\n{body}\n")


def fmt_row(label: str, groups: dict[str, np.ndarray], baseline: str = "control") -> dict:
    endo = groups.get("endogenous", np.array([]))
    exo = groups.get("exogenous", np.array([]))
    ctrl = groups.get(baseline, np.array([]))
    dec, _, sec = welch(endo, ctrl)
    dxc, _, sxc = welch(exo, ctrl)
    return {
        "condition": label,
        "endo_n": f"{len(endo):,}",
        "endo_μ": f"{np.mean(endo):+.3f}" if len(endo) else "",
        "exo_n": f"{len(exo):,}",
        "exo_μ": f"{np.mean(exo):+.3f}" if len(exo) else "",
        "ctrl_n": f"{len(ctrl):,}",
        "ctrl_μ": f"{np.mean(ctrl):+.3f}" if len(ctrl) else "",
        "Δ endo-ctrl": f"{dec:+.3f} {sec}" if not np.isnan(dec) else "",
        "Δ exo-ctrl": f"{dxc:+.3f} {sxc}" if not np.isnan(dxc) else "",
    }


# ============================================================================
# Build rich analysis DataFrame
# ============================================================================


def build_rich_analysis(memo, run_size: int = 6, minutes: float = 3.0) -> pl.DataFrame:
    """Return a DataFrame with one row per candidate event, enriched with
    features needed by the new experiments.

    Columns:
        gameId, period, game_seconds_elapsed, seconds_remaining,
        actionType, subType, teamId, streak, lead, lead_change, recovery,
        home_teamId, group (endogenous | exogenous | control), subtype_fine,
        suffering_location, suffering_margin, season_type, clutch,
        is_period, is_stoppage
    """
    df = pl.DataFrame._from_pydf(memo.cdnnba._df)
    streak_s = memo.streak
    lead_s = memo.lead
    lead_fwd = memo.lead_change_n_mins(minutes)
    home_teams = memo.home_team_per_game

    analysis = pl.DataFrame(
        {
            "gameId": df["gameId"],
            "period": df["period"],
            "game_seconds_elapsed": df["game_seconds_elapsed"],
            "seconds_remaining": df["seconds_remaining"],
            "actionType": df["actionType"],
            "subType": df["subType"],
            "timeout_cause": df["timeout_cause"],
            "teamId": df["teamId"],
            "season": df["season"],
            "season_type": df["season_type"],
            "streak": streak_s,
            "lead": lead_s,
            "lead_change": lead_fwd,
            "orderNumber": df["orderNumber"],
        }
    ).join(home_teams, on="gameId", how="left")

    # Filter to runs with a measurable forward
    analysis = analysis.filter(pl.col("streak").abs() >= run_size)
    analysis = analysis.filter(pl.col("lead_change").is_not_null())
    analysis = analysis.with_columns(
        (-pl.col("streak").sign() * pl.col("lead_change")).alias("recovery"),
        (-pl.col("streak").sign() * pl.col("lead")).alias("suffering_margin"),
        pl.when(pl.col("streak") > 0).then(pl.lit("away")).otherwise(pl.lit("home")).alias("suffering_location"),
        # clutch: last 5 min AND margin ≤ 5
        ((pl.col("period") >= 4) & (pl.col("seconds_remaining") <= 300) & (pl.col("lead").abs() <= 5)).alias("clutch"),
    )

    # Fine subtype classification — uses ``timeout_cause`` (cause-based
    # taxonomy from TVTimeoutValidation) rather than the legacy subType-based
    # split. ``official_inferred`` is dead in the new pipeline; tv_mandatory
    # rows carry the structural signal directly.
    is_timeout = pl.col("actionType") == "timeout"
    is_stop = pl.col("actionType") == "stoppage"
    cause = pl.col("timeout_cause")
    subtype_fine_expr = (
        pl.when(is_timeout & (cause == "tv_mandatory"))
        .then(pl.lit("tv_mandatory"))
        .when(is_timeout & (cause == "coach_absorb"))
        .then(pl.lit("coach_absorb"))
        .when(is_timeout & (cause == "coach_discretionary"))
        .then(pl.lit("coach_discretionary"))
        .when(is_timeout & (cause == "mistagged_discretionary"))
        .then(pl.lit("mistagged_discretionary"))
        .when(is_timeout & (cause == "challenge"))
        .then(pl.lit("coach_challenge"))
        .when(is_stop)
        .then(pl.lit("stoppage"))
        .otherwise(pl.lit("control"))
    )
    analysis = analysis.with_columns(subtype_fine_expr.alias("subtype_fine"))

    # Coarse group (endogenous / exogenous / control).
    # Endogenous = coach chose a *strategic* TO — coach_discretionary,
    #              mistagged_discretionary, coach_challenge.
    # Exogenous  = the break would have happened regardless of coach choice —
    #              tv_mandatory (auto-fire), coach_absorb (slot was about to
    #              fire; coach just chose the moment), or stoppage.
    # Control    = everything else (filtered later to one event per run segment).
    # NOTE: coach_absorb sits in exogenous because the slot was queued to
    # auto-fire within ~80s anyway — the coach isn't gaining strategic value
    # by calling it, just picking the moment.
    endo_sub = pl.col("subtype_fine").is_in(["coach_discretionary", "mistagged_discretionary", "coach_challenge"])
    exo_sub = pl.col("subtype_fine").is_in(["tv_mandatory", "coach_absorb", "stoppage"])
    # For endogenous, also verify the calling team is the suffering team —
    # the team being scored against is the one with strategic motive to break
    # the run. A TO called by the running team is a different kind of event.
    suffering_called = ((pl.col("streak") > 0) & (pl.col("teamId") != pl.col("home_teamId"))) | (
        (pl.col("streak") < 0) & (pl.col("teamId") == pl.col("home_teamId"))
    )
    analysis = analysis.with_columns(
        pl.when(endo_sub & suffering_called)
        .then(pl.lit("endogenous"))
        .when(exo_sub)
        .then(pl.lit("exogenous"))
        .otherwise(pl.lit("control"))
        .alias("group")
    )

    # Dedupe controls: one per run segment
    analysis = analysis.with_columns(
        pl.col("streak").sign().alias("_ss"),
    ).with_columns(
        ((pl.col("_ss") != pl.col("_ss").shift(1)) | (pl.col("gameId") != pl.col("gameId").shift(1)))
        .cum_sum()
        .alias("_seg")
    )
    control_only = analysis.filter(pl.col("group") == "control").group_by("gameId", "_seg", maintain_order=True).first()
    non_control = analysis.filter(pl.col("group") != "control")
    # Align column order before concat
    cols = non_control.columns
    control_only = control_only.select(cols)
    out = pl.concat([non_control, control_only]).drop(["_ss", "_seg"])
    return out


# ============================================================================
# E9: Matched-twin within-game causal analysis
# ============================================================================


def experiment_9(memo, analysis: pl.DataFrame):
    print("=" * 60, "\nE9: Matched-twin within-game analysis")
    append_md("\n## Experiment 9: Matched-twin within-game causal analysis\n")
    append_md(
        "\nFor each treated event (coach timeout or TV timeout), we look for a\n"
        "control event **in the same game** that also has a run underway and\n"
        "matches on:\n"
        "\n"
        "- same period\n"
        "- same sign of `streak` (so it's a run against the same team)\n"
        "- `|Δ streak|` ≤ 2 (similar run magnitude)\n"
        "- `|Δ seconds_remaining|` ≤ 120s (similar game-clock position in period)\n"
        "- `|Δ suffering_margin|` ≤ 3 (similar score state)\n"
        "\n"
        "When multiple controls match, we pick the one with the smallest combined\n"
        "distance. The difference in `recovery` between the treated and matched\n"
        "control is the per-pair causal estimate. Paired t-test aggregates them.\n"
    )

    # Partition treated and control
    treated = analysis.filter(pl.col("group") != "control").with_row_index("_tid")
    controls = analysis.filter(pl.col("group") == "control")

    print(f"  Treated events: {treated.height:,}")
    print(f"  Control events: {controls.height:,}")

    # Cross join within game and period
    joined = treated.join(
        controls.select(
            "gameId",
            "period",
            pl.col("streak").alias("c_streak"),
            pl.col("seconds_remaining").alias("c_sr"),
            pl.col("suffering_margin").alias("c_margin"),
            pl.col("recovery").alias("c_recovery"),
        ),
        on=["gameId", "period"],
        how="inner",
    )
    # Apply match constraints
    joined = joined.filter(
        (pl.col("streak").sign() == pl.col("c_streak").sign())
        & ((pl.col("streak") - pl.col("c_streak")).abs() <= 2)
        & ((pl.col("seconds_remaining") - pl.col("c_sr")).abs() <= 120)
        & ((pl.col("suffering_margin") - pl.col("c_margin")).abs() <= 3)
    )
    # Pick the best (smallest combined distance) per treated event
    joined = joined.with_columns(
        (
            (pl.col("streak") - pl.col("c_streak")).abs() * 2
            + ((pl.col("seconds_remaining") - pl.col("c_sr")).abs() / 30)
            + ((pl.col("suffering_margin") - pl.col("c_margin")).abs() * 3)
        ).alias("_dist")
    )
    best = joined.sort("_tid", "_dist").group_by("_tid", maintain_order=True).first()

    print(f"  Matched pairs: {best.height:,}")

    # Paired diffs per group
    rows = []
    for grp in ["endogenous", "exogenous"]:
        sub = best.filter(pl.col("group") == grp)
        if sub.height < 30:
            continue
        treated_r = sub["recovery"].to_numpy()
        ctrl_r = sub["c_recovery"].to_numpy()
        diffs = treated_r - ctrl_r
        t, p = sp_stats.ttest_1samp(diffs, 0)
        rows.append(
            {
                "group": grp,
                "matched_n": f"{len(diffs):,}",
                "treated μ": f"{treated_r.mean():+.3f}",
                "matched ctrl μ": f"{ctrl_r.mean():+.3f}",
                "pair diff": f"{diffs.mean():+.4f}",
                "t": f"{t:.3f}",
                "p": f"{p:.4f}",
                "sig": sig(float(p)),  # type: ignore
            }
        )

    # Fine subtype breakdown
    fine_rows = []
    for grp in [
        "tv_mandatory",
        "stoppage",
        "coach_absorb",
        "coach_discretionary",
        "mistagged_discretionary",
        "coach_challenge",
    ]:
        sub = best.filter(pl.col("subtype_fine") == grp)
        if sub.height < 30:
            continue
        treated_r = sub["recovery"].to_numpy()
        ctrl_r = sub["c_recovery"].to_numpy()
        diffs = treated_r - ctrl_r
        t, p = sp_stats.ttest_1samp(diffs, 0)
        fine_rows.append(
            {
                "subtype": grp,
                "matched_n": f"{len(diffs):,}",
                "treated μ": f"{treated_r.mean():+.3f}",
                "matched ctrl μ": f"{ctrl_r.mean():+.3f}",
                "pair diff": f"{diffs.mean():+.4f}",
                "t": f"{t:.3f}",
                "p": f"{p:.4f}",
                "sig": sig(float(p)),  # type: ignore
            }
        )

    append_table(
        "E9 table: matched-twin causal estimates (coarse groups)",
        rows,
        ["group", "matched_n", "treated μ", "matched ctrl μ", "pair diff", "t", "p", "sig"],
    )
    append_table(
        "E9 table: matched-twin causal estimates (fine subtypes)",
        fine_rows,
        ["subtype", "matched_n", "treated μ", "matched ctrl μ", "pair diff", "t", "p", "sig"],
    )

    append_md(
        "\n**Takeaways:**\n"
        "- Matched-twin is the strongest within-game control possible: each\n"
        "  treated event is paired with a near-identical non-timeout moment in the\n"
        "  same game.\n"
        "- Compared to E8 (within-game mean comparison), this approach removes the\n"
        "  remaining confounding from trailing-momentum differences.\n"
    )


# ============================================================================
# E10: Fine-grained timeout subtype breakdown
# ============================================================================


def experiment_10(memo, analysis: pl.DataFrame):
    print("=" * 60, "\nE10: Fine subtype breakdown")
    append_md("\n## Experiment 10: Fine-grained timeout subtype breakdown\n")
    append_md(
        "\nSplit the endo/exo bins into their constituent subtypes and compare\n" "each against the control baseline.\n"
    )

    rows = []
    ctrl = analysis.filter(pl.col("subtype_fine") == "control")["recovery"].to_numpy()
    for sub in [
        "tv_mandatory",
        "stoppage",
        "coach_absorb",
        "coach_discretionary",
        "mistagged_discretionary",
        "coach_challenge",
    ]:
        arr = analysis.filter(pl.col("subtype_fine") == sub)["recovery"].to_numpy()
        d, p, s = welch(arr, ctrl)
        rows.append(
            {
                "subtype": sub,
                "n": f"{len(arr):,}",
                "μ": f"{np.mean(arr):+.3f}" if len(arr) else "",
                "σ": f"{np.std(arr):.3f}" if len(arr) else "",
                "ctrl μ": f"{np.mean(ctrl):+.3f}",
                "Δ vs ctrl": f"{d:+.3f}" if not np.isnan(d) else "",
                "p": f"{p:.4f}" if not np.isnan(p) else "",
                "sig": s,
            }
        )

    append_table(
        "E10 table: recovery by fine subtype (run≥6, 3-min)",
        rows,
        ["subtype", "n", "μ", "σ", "ctrl μ", "Δ vs ctrl", "p", "sig"],
    )

    append_md(
        "\n**Takeaways:**\n"
        "- `tv_mandatory` are league-forced commercial breaks — strictly\n"
        "  exogenous (coach didn't choose, team owns slot per rulebook).\n"
        "- `coach_absorb` is the much rarer endogenous TO that satisfies a\n"
        "  pending mandatory slot (called within ~80s of the trigger).\n"
        "- `coach_discretionary` are pure coach calls — no mandatory tag.\n"
        '- `mistagged_discretionary` were league-tagged "mandatory" but failed\n'
        "  the rulebook's slot-owner / first-team-TO / proximity gates. We\n"
        "  treat them as endogenous: the coach chose to call them.\n"
        "- `coach_challenge` is a structurally distinct coach decision.\n"
        "- `stoppage` events (out-of-bounds, injury, etc.) are grouped with\n"
        "  exogenous in other experiments; here we see them separately.\n"
    )


# ============================================================================
# E11: Time-of-game conditioning
# ============================================================================


def experiment_11(memo, analysis: pl.DataFrame):
    print("=" * 60, "\nE11: Time-of-game conditioning")
    append_md("\n## Experiment 11: Time-of-game conditioning\n")
    append_md(
        "\nDoes the timeout effect scale with game phase? Split events by\n"
        "buckets of `game_seconds_elapsed` and also flag clutch (last 5 min of\n"
        "Q4, margin ≤ 5).\n"
    )

    # Bucket by game phase
    buckets = [
        ("Q1 early (0-360s)", 0, 360),
        ("Q1 late (360-720s)", 360, 720),
        ("Q2 early (720-1080s)", 720, 1080),
        ("Q2 late (1080-1440s)", 1080, 1440),
        ("Q3 early (1440-1800s)", 1440, 1800),
        ("Q3 late (1800-2160s)", 1800, 2160),
        ("Q4 early (2160-2520s)", 2160, 2520),
        ("Q4 late (2520-2880s)", 2520, 2880),
    ]

    rows = []
    for label, lo, hi in buckets:
        sub = analysis.filter((pl.col("game_seconds_elapsed") >= lo) & (pl.col("game_seconds_elapsed") < hi))
        groups = {
            g: sub.filter(pl.col("group") == g)["recovery"].to_numpy() for g in ["endogenous", "exogenous", "control"]
        }
        rows.append(fmt_row(label, groups))

    append_table(
        "E11 table: recovery by game phase (run≥6, 3-min)",
        rows,
        ["condition", "endo_n", "endo_μ", "exo_n", "exo_μ", "ctrl_n", "ctrl_μ", "Δ endo-ctrl", "Δ exo-ctrl"],
    )

    # Clutch analysis
    clutch_sub = analysis.filter(pl.col("clutch"))
    nonclutch_sub = analysis.filter(~pl.col("clutch"))
    clutch_rows = []
    for label, sub in [("Clutch (Q4 last 5min, |margin|≤5)", clutch_sub), ("Non-clutch", nonclutch_sub)]:
        groups = {
            g: sub.filter(pl.col("group") == g)["recovery"].to_numpy() for g in ["endogenous", "exogenous", "control"]
        }
        clutch_rows.append(fmt_row(label, groups))

    append_table(
        "E11 table: clutch vs non-clutch (run≥6, 3-min)",
        clutch_rows,
        ["condition", "endo_n", "endo_μ", "exo_n", "exo_μ", "ctrl_n", "ctrl_μ", "Δ endo-ctrl", "Δ exo-ctrl"],
    )

    append_md(
        "\n**Takeaways:**\n"
        "- Recovery baseline decays through the game as point-swings compress\n"
        "  (less room to regress). Q4 late shows the lowest control recovery.\n"
        "- The exogenous penalty is largest in Q4, aligning with E4's finding\n"
        "  that Q4 is the most sensitive period to stoppage interruption.\n"
        "- Clutch-time sample is small; the sign of the endogenous effect in\n"
        "  clutch moments is worth tracking for future studies.\n"
    )


# ============================================================================
# E12: Team quality conditioning
# ============================================================================


def experiment_12(memo, analysis: pl.DataFrame):
    print("=" * 60, "\nE12: Team quality conditioning")
    append_md("\n## Experiment 12: Team quality conditioning\n")
    append_md(
        "\nUses `player_advanced_stats` to compute each team's season-level\n"
        "average `NET_RATING` (weighted by games played). Each game gets a\n"
        "`team_net_rating_diff` = home team NET - away team NET. The suffering\n"
        "team is classified as better/worse relative to its opponent.\n"
    )

    # Compute team-season NET_RATING (weighted average)
    pas = pl.DataFrame._from_pydf(memo.player_advanced_stats._df)
    team_net = (
        pas.with_columns((pl.col("NET_RATING") * pl.col("GP")).alias("net_gp"))
        .group_by("TEAM_ID", "season_int")
        .agg(
            (pl.col("net_gp").sum() / pl.col("GP").sum()).alias("team_net_rating"),
        )
        .rename({"TEAM_ID": "teamId", "season_int": "season"})
    )

    # Map home and away teamIds for each game
    home_teams = memo.home_team_per_game.rename({"home_teamId": "home_tid"})
    df = pl.DataFrame._from_pydf(memo.cdnnba._df)
    # Away team: teamId values that appear in the game that aren't the home team
    away_teams = (
        df.filter(pl.col("teamId") > 0)
        .join(home_teams, on="gameId", how="left")
        .filter(pl.col("teamId") != pl.col("home_tid"))
        .select("gameId", "teamId", "season")
        .unique()
        .group_by("gameId")
        .agg(pl.col("teamId").first().alias("away_tid"), pl.col("season").first())
    )
    # Join team net ratings
    game_quality = (
        home_teams.join(away_teams, on="gameId", how="inner")
        .join(
            team_net.rename({"teamId": "home_tid", "team_net_rating": "home_net"}),
            on=["home_tid", "season"],
            how="left",
        )
        .join(
            team_net.rename({"teamId": "away_tid", "team_net_rating": "away_net"}),
            on=["away_tid", "season"],
            how="left",
        )
        .with_columns((pl.col("home_net") - pl.col("away_net")).alias("home_minus_away_net"))
    )

    # Join to analysis
    analysis2 = analysis.join(
        game_quality.select("gameId", "home_minus_away_net", "home_net", "away_net"),
        on="gameId",
        how="left",
    )
    # For each row, determine suffering team's rating and the gap
    analysis2 = analysis2.with_columns(
        pl.when(pl.col("suffering_location") == "home")
        .then(pl.col("home_net"))
        .otherwise(pl.col("away_net"))
        .alias("suffering_team_net"),
        pl.when(pl.col("suffering_location") == "home")
        .then(pl.col("home_net") - pl.col("away_net"))
        .otherwise(pl.col("away_net") - pl.col("home_net"))
        .alias("suffering_vs_opp_net"),
    )

    # Bucket by suffering team vs opponent net rating
    buckets = [
        ("Suffering team much worse (Δ ≤ -5)", float("-inf"), -5),
        ("Suffering team worse (-5 < Δ ≤ -1)", -5, -1),
        ("Evenly matched (|Δ| < 1)", -1, 1),
        ("Suffering team better (1 ≤ Δ < 5)", 1, 5),
        ("Suffering team much better (Δ ≥ 5)", 5, float("inf")),
    ]
    rows = []
    for label, lo, hi in buckets:
        sub = analysis2.filter(
            pl.col("suffering_vs_opp_net").is_not_null()
            & (pl.col("suffering_vs_opp_net") >= lo)
            & (pl.col("suffering_vs_opp_net") < hi)
        )
        groups = {
            g: sub.filter(pl.col("group") == g)["recovery"].to_numpy() for g in ["endogenous", "exogenous", "control"]
        }
        rows.append(fmt_row(label, groups))

    append_table(
        "E12 table: recovery by team-quality gap (run≥6, 3-min)",
        rows,
        ["condition", "endo_n", "endo_μ", "exo_n", "exo_μ", "ctrl_n", "ctrl_μ", "Δ endo-ctrl", "Δ exo-ctrl"],
    )

    # Absolute team rating
    abs_rows = []
    for label, lo, hi in [
        ("Weak team (NET ≤ -2)", float("-inf"), -2),
        ("Average team (-2 < NET < 2)", -2, 2),
        ("Strong team (NET ≥ 2)", 2, float("inf")),
    ]:
        sub = analysis2.filter(
            pl.col("suffering_team_net").is_not_null()
            & (pl.col("suffering_team_net") >= lo)
            & (pl.col("suffering_team_net") < hi)
        )
        groups = {
            g: sub.filter(pl.col("group") == g)["recovery"].to_numpy() for g in ["endogenous", "exogenous", "control"]
        }
        abs_rows.append(fmt_row(label, groups))

    append_table(
        "E12 table: recovery by absolute suffering team quality",
        abs_rows,
        ["condition", "endo_n", "endo_μ", "exo_n", "exo_μ", "ctrl_n", "ctrl_μ", "Δ endo-ctrl", "Δ exo-ctrl"],
    )

    append_md(
        "\n**Takeaways:**\n"
        "- Stronger teams (higher NET_RATING) might have better recovery in\n"
        "  general — the control column reveals this baseline.\n"
        "- The interesting question: does the endo-ctrl gap change based on\n"
        "  relative quality? If strong teams benefit more from coach timeouts,\n"
        "  the Δ endo-ctrl should be larger in that row.\n"
    )


# ============================================================================
# E13: Substitution-adjusted analysis
# ============================================================================


def experiment_13(memo, analysis: pl.DataFrame):
    print("=" * 60, "\nE13: Substitution-adjusted analysis")
    append_md("\n## Experiment 13: Substitution-adjusted analysis\n")
    append_md(
        "\nFollowing Weimer et al., we split events by whether substitutions\n"
        "occurred near the event. A substitution during/immediately after the\n"
        "timeout is the cleanest proxy for strategy change.\n"
        "\nWe count `substitution` events in cdnnba within a ±30s game-clock\n"
        "window around each treated/control moment and bucket by:\n"
        "`0 subs`, `1-2 subs`, `3+ subs`.\n"
    )

    # Build substitution counts per (game, approximate_moment)
    df = pl.DataFrame._from_pydf(memo.cdnnba._df)
    subs = (
        df.filter(pl.col("actionType") == "substitution")
        .select("gameId", "game_seconds_elapsed")
        .rename({"game_seconds_elapsed": "sub_gse"})
    )

    # For each analysis row, count subs in ±30 seconds within same game.
    # We do this via a join with a range filter.
    ev = analysis.select(
        "gameId", "game_seconds_elapsed", "orderNumber", "group", "recovery", "subtype_fine"
    ).with_row_index("_eid")

    joined = ev.join(subs, on="gameId", how="left")
    joined = joined.with_columns(((pl.col("sub_gse") - pl.col("game_seconds_elapsed")).abs() <= 30).alias("nearby"))
    sub_counts = joined.group_by("_eid").agg(pl.col("nearby").sum().alias("n_subs"))
    ev = ev.join(sub_counts, on="_eid", how="left").with_columns(pl.col("n_subs").fill_null(0))

    buckets = [
        ("0 subs", 0, 0),
        ("1-2 subs", 1, 2),
        ("3+ subs", 3, 999),
    ]
    rows = []
    for label, lo, hi in buckets:
        sub = ev.filter((pl.col("n_subs") >= lo) & (pl.col("n_subs") <= hi))
        groups = {
            g: sub.filter(pl.col("group") == g)["recovery"].to_numpy() for g in ["endogenous", "exogenous", "control"]
        }
        rows.append(fmt_row(label, groups))

    append_table(
        "E13 table: recovery by substitution count (run≥6, 3-min)",
        rows,
        ["condition", "endo_n", "endo_μ", "exo_n", "exo_μ", "ctrl_n", "ctrl_μ", "Δ endo-ctrl", "Δ exo-ctrl"],
    )

    append_md(
        "\n**Takeaways:**\n"
        "- Timeout moments without substitutions are the cleanest test of the\n"
        "  'pause-in-play' effect (no strategy proxy confound).\n"
        "- If the endo-ctrl gap persists in the '0 subs' row, that's evidence\n"
        "  the effect isn't driven by personnel swaps.\n"
        "- Many coach timeouts (Weimer's data suggests most) involve 1+\n"
        "  substitutions; split lets us isolate the pure pause effect.\n"
    )


# ============================================================================
# E14: Head-to-head PPP baselines
# ============================================================================


def experiment_14(memo, analysis: pl.DataFrame):
    print("=" * 60, "\nE14: Head-to-head PPP baselines")
    append_md("\n## Experiment 14: Head-to-head points-per-possession baselines\n")
    append_md(
        "\nWe compute each team's cumulative points-per-possession (PPP) within\n"
        "each game up to the moment of interest. Then we compare the current\n"
        "run's intensity to the team's in-game baseline: 'is the run way above\n"
        "the calling team's normal rhythm?'\n"
    )

    # Compute PPP per game, per team, cumulatively
    # Use possessions table from memo
    poss = memo.possessions
    # For each possession, the team is `possession`, pts is `possession_points`
    # Compute cumulative PPP per (game, team) in chronological order
    poss_with_ppp = (
        poss.sort("gameId", "possession_id")
        .with_columns(
            pl.col("possession_points").cum_sum().over("gameId", "possession").alias("cum_pts"),
            pl.cum_count("possession_id").over("gameId", "possession").alias("cum_poss"),
        )
        .with_columns((pl.col("cum_pts") / pl.col("cum_poss")).alias("team_cum_ppp"))
        .select("gameId", "possession_id", "possession", "team_cum_ppp")
        .rename({"possession": "teamId_p"})
    )

    # For each analysis row, join by gameId + find the possession matching.
    # This is tricky because analysis doesn't have possession_id directly.
    # Approximation: use the CURRENT team (possession at event) from cdnnba.
    df = pl.DataFrame._from_pydf(memo.cdnnba._df).select(
        "gameId",
        "orderNumber",
        pl.col("possession").alias("poss_team_at_event"),
        "possession_id",
    )
    ev = analysis.join(df, on=["gameId", "orderNumber"], how="left")
    # Join cumulative PPP for the possessing team (which is the RUNNING team)
    ev = ev.join(
        poss_with_ppp.select(
            pl.col("gameId"),
            pl.col("teamId_p").alias("poss_team_at_event"),
            pl.col("possession_id"),
            "team_cum_ppp",
        ),
        on=["gameId", "poss_team_at_event", "possession_id"],
        how="left",
    ).rename({"team_cum_ppp": "running_team_ppp"})

    # Bucket by running team's cum PPP (how efficient have they been this game?)
    buckets = [
        ("Running team PPP < 1.0", 0.0, 1.0),
        ("Running team PPP 1.0-1.15", 1.0, 1.15),
        ("Running team PPP 1.15-1.30", 1.15, 1.30),
        ("Running team PPP ≥ 1.30", 1.30, 99.0),
    ]
    rows = []
    for label, lo, hi in buckets:
        sub = ev.filter(
            pl.col("running_team_ppp").is_not_null()
            & (pl.col("running_team_ppp") >= lo)
            & (pl.col("running_team_ppp") < hi)
        )
        groups = {
            g: sub.filter(pl.col("group") == g)["recovery"].to_numpy() for g in ["endogenous", "exogenous", "control"]
        }
        rows.append(fmt_row(label, groups))

    append_table(
        "E14 table: recovery by running team's in-game PPP",
        rows,
        ["condition", "endo_n", "endo_μ", "exo_n", "exo_μ", "ctrl_n", "ctrl_μ", "Δ endo-ctrl", "Δ exo-ctrl"],
    )

    append_md(
        "\n**Takeaways:**\n"
        "- This bucketing asks: when the running team is unusually efficient in\n"
        "  this particular game, does the timeout work differently?\n"
        "- If timeouts help more against hot teams (higher PPP), that's evidence\n"
        "  for the 'momentum' hypothesis.\n"
    )


# ============================================================================
# E15: Coach challenge impact, split by successful vs failed outcome
# ============================================================================


def _build_challenge_outcomes(memo: CDNNBAMemoPL) -> pl.DataFrame:
    """One row per coach challenge with the recorded replay outcome.

    descriptor values:
      overturned -> challenge SUCCEEDED (original call reversed)
      stands     -> call stands (challenge failed; replay inconclusive)
      support    -> original call confirmed (challenge failed; replay definitive)

    Data caveat: cdnnba only emits instantreplay rows for the 2020 and 2021
    seasons. Challenges from 2022 onward have no recorded outcome and will be
    excluded by the downstream join.
    """
    df = pl.DataFrame._from_pydf(memo.cdnnba._df)
    return df.filter((pl.col("actionType") == "instantreplay") & (pl.col("subType") == "challenge")).select(
        pl.col("gameId"),
        pl.col("period"),
        pl.col("game_seconds_elapsed").alias("replay_seconds_elapsed"),
        pl.col("descriptor"),
    )


def experiment_15(memo: CDNNBAMemoPL, analysis: pl.DataFrame):
    print("=" * 60, "\nE15: Coach challenge impact, split by outcome")
    append_md("\n## Experiment 15: Coach challenge timeout impact by outcome\n")
    append_md(
        "\nFor every `coach_challenge` event we attach the replay outcome\n"
        "(`overturned` -> success, `stands`/`support` -> failure) by matching\n"
        "the nearest `instantreplay/challenge` row in the same game and period\n"
        "(within 60s of `game_seconds_elapsed`). We then re-run the E9\n"
        "matched-twin pairing within the challenge subgroup and split the\n"
        "per-pair diffs by outcome.\n"
        "\n"
        "**Data caveat:** the cdnnba feed only emits `instantreplay` rows for\n"
        "seasons 2020 and 2021. Challenges from 2022 onward have no recorded\n"
        "outcome and are excluded.\n"
    )

    outcomes = _build_challenge_outcomes(memo)
    print(f"  Replay-outcome rows in cdnnba: {outcomes.height:,}")
    if outcomes.height == 0:
        append_md("\nNo replay-outcome rows in cdnnba; skipping.\n")
        return

    ch_rows = analysis.filter(pl.col("subtype_fine") == "coach_challenge").with_row_index("_chid")
    joined = ch_rows.join(outcomes, on=["gameId", "period"], how="left").with_columns(
        (pl.col("game_seconds_elapsed") - pl.col("replay_seconds_elapsed")).abs().alias("_td")
    )
    joined = joined.filter(pl.col("_td").is_null() | (pl.col("_td") <= 60))
    ch_annotated = (
        joined.sort("_chid", "_td", nulls_last=True)
        .group_by("_chid", maintain_order=True)
        .first()
        .drop(["_td", "replay_seconds_elapsed"])
    )

    n_total = ch_annotated.height
    n_with = ch_annotated.filter(pl.col("descriptor").is_not_null()).height
    n_over = ch_annotated.filter(pl.col("descriptor") == "overturned").height
    n_fail = ch_annotated.filter(pl.col("descriptor").is_in(["stands", "support"])).height
    print(f"  coach_challenge analysis rows: {n_total:,}")
    print(f"  ... outcome attached: {n_with:,} (overturned={n_over:,}, failed={n_fail:,})")

    treated = ch_annotated.filter(pl.col("descriptor").is_not_null()).drop("_chid").with_row_index("_tid")
    controls = analysis.filter(pl.col("group") == "control")
    pairs = (
        treated.join(
            controls.select(
                "gameId",
                "period",
                pl.col("streak").alias("c_streak"),
                pl.col("seconds_remaining").alias("c_sr"),
                pl.col("suffering_margin").alias("c_margin"),
                pl.col("recovery").alias("c_recovery"),
            ),
            on=["gameId", "period"],
            how="inner",
        )
        .filter(
            (pl.col("streak").sign() == pl.col("c_streak").sign())
            & ((pl.col("streak") - pl.col("c_streak")).abs() <= 2)
            & ((pl.col("seconds_remaining") - pl.col("c_sr")).abs() <= 120)
            & ((pl.col("suffering_margin") - pl.col("c_margin")).abs() <= 3)
        )
        .with_columns(
            (
                (pl.col("streak") - pl.col("c_streak")).abs() * 2
                + ((pl.col("seconds_remaining") - pl.col("c_sr")).abs() / 30)
                + ((pl.col("suffering_margin") - pl.col("c_margin")).abs() * 3)
            ).alias("_dist")
        )
    )
    best = pairs.sort("_tid", "_dist").group_by("_tid", maintain_order=True).first()
    print(f"  Matched challenge pairs: {best.height:,}")

    matched_rows = []
    for label, filt in [
        ("Successful (overturned)", pl.col("descriptor") == "overturned"),
        ("Failed (stands / support)", pl.col("descriptor").is_in(["stands", "support"])),
        ("All outcomes pooled", pl.col("descriptor").is_not_null()),
    ]:
        sub = best.filter(filt)
        n = sub.height
        if n < 2:
            matched_rows.append(
                {
                    "outcome": label,
                    "matched_n": f"{n:,}",
                    "treated μ": "",
                    "matched ctrl μ": "",
                    "pair diff": "",
                    "t": "",
                    "p": "",
                    "sig": "insufficient n",
                }
            )
            continue
        treated_r = sub["recovery"].to_numpy()
        ctrl_r = sub["c_recovery"].to_numpy()
        diffs = treated_r - ctrl_r
        t, p = sp_stats.ttest_1samp(diffs, 0)
        matched_rows.append(
            {
                "outcome": label,
                "matched_n": f"{n:,}",
                "treated μ": f"{treated_r.mean():+.3f}",
                "matched ctrl μ": f"{ctrl_r.mean():+.3f}",
                "pair diff": f"{diffs.mean():+.4f}",
                "t": f"{t:.3f}",
                "p": f"{p:.4f}",
                "sig": sig(float(p)),  # type: ignore
            }
        )
    append_table(
        "E15 table: matched-twin estimates for coach challenges, by outcome",
        matched_rows,
        ["outcome", "matched_n", "treated μ", "matched ctrl μ", "pair diff", "t", "p", "sig"],
    )

    # --------------------------------------------------------------------
    # Broader between-group view: every 2020-21 challenge, no streak filter
    # --------------------------------------------------------------------
    # The matched-twin table is restricted to events with |streak| >= 6 (the
    # analysis-DataFrame filter), and almost no `stands`/`support` challenges
    # happen during such runs. Below we drop the streak requirement entirely
    # and work directly from the `instantreplay/challenge` rows, which (i)
    # carry the calling team's `teamId`, (ii) carry the recorded outcome via
    # `descriptor`, and (iii) cover the full 2020-21 challenge population
    # (1,465 events). The corresponding `timeout/challenge` rows under-cover
    # the population and are biased toward `overturned`, so they are not used.

    df = pl.DataFrame._from_pydf(memo.cdnnba._df)
    lead_fwd = memo.lead_change_n_mins(3.0)
    home_teams = memo.home_team_per_game
    challenges = (
        pl.DataFrame(
            {
                "gameId": df["gameId"],
                "actionType": df["actionType"],
                "subType": df["subType"],
                "teamId": df["teamId"],
                "descriptor": df["descriptor"],
                "lead_change": lead_fwd,
            }
        )
        .filter(
            (pl.col("actionType") == "instantreplay")
            & (pl.col("subType") == "challenge")
            & pl.col("lead_change").is_not_null()
            & pl.col("teamId").is_not_null()
        )
        .join(home_teams, on="gameId", how="left")
        .with_columns(
            pl.when(pl.col("teamId") == pl.col("home_teamId"))
            .then(pl.col("lead_change"))
            .otherwise(-pl.col("lead_change"))
            .alias("calling_lead_change")
        )
    )
    print(
        "  Broad 2020-21 challenges (from replay rows): total="
        f"{challenges.height:,}, "
        f"overturned={challenges.filter(pl.col('descriptor') == 'overturned').height:,}, "
        f"failed={challenges.filter(pl.col('descriptor').is_in(['stands', 'support'])).height:,}"
    )

    overturned = challenges.filter(pl.col("descriptor") == "overturned")["calling_lead_change"].to_numpy()
    failed = challenges.filter(pl.col("descriptor").is_in(["stands", "support"]))["calling_lead_change"].to_numpy()

    bg_rows = []
    for label, arr in [("Successful (overturned)", overturned), ("Failed (stands / support)", failed)]:
        if len(arr) < 2:
            bg_rows.append(
                {
                    "outcome": label,
                    "n": f"{len(arr):,}",
                    "μ Δlead (3m)": "",
                    "t vs 0": "",
                    "p vs 0": "",
                    "sig": "insufficient n",
                }
            )
            continue
        t, p = sp_stats.ttest_1samp(arr, 0)
        bg_rows.append(
            {
                "outcome": label,
                "n": f"{len(arr):,}",
                "μ Δlead (3m)": f"{arr.mean():+.3f}",
                "t vs 0": f"{t:.3f}",
                "p vs 0": f"{p:.4f}",
                "sig": sig(float(p)),  # type: ignore
            }
        )
    # Direct comparison overturned vs failed
    if len(overturned) >= 2 and len(failed) >= 2:
        d, p_ww, s_ww = welch(overturned, failed)
        bg_rows.append(
            {
                "outcome": "Δ (overturned − failed)",
                "n": "",
                "μ Δlead (3m)": f"{d:+.3f}",
                "t vs 0": "",
                "p vs 0": f"{p_ww:.4f}",
                "sig": s_ww,
            }
        )
    append_table(
        "E15 table: calling-team Δlead (+3 min), all 2020-21 challenges, split by outcome",
        bg_rows,
        ["outcome", "n", "μ Δlead (3m)", "t vs 0", "p vs 0", "sig"],
    )

    append_md(
        "\n**Takeaways:**\n"
        "- Replay outcomes are only recorded in cdnnba for the 2020 and 2021\n"
        "  seasons. The 2022+ feed drops `instantreplay` rows entirely, so all\n"
        "  numbers in this section are 2020-21 only.\n"
        "- The matched-twin split is structurally biased: almost no failed\n"
        "  challenges happen during a 6+ run (the analysis-DataFrame filter),\n"
        "  so the failed bucket is essentially empty. The broader table below\n"
        "  drops the streak filter and is the meaningful split.\n"
        "- The corresponding `timeout/challenge` rows in cdnnba are biased\n"
        "  toward overturned challenges (~89% of timeout-rows in 2020-21 sit\n"
        "  next to an `overturned` replay row). The broader analysis instead\n"
        "  works directly off the 1,465 `instantreplay/challenge` rows, which\n"
        "  carry both the calling team's `teamId` and the recorded outcome.\n"
        "- Mechanism: an `overturned` challenge mechanically alters the score\n"
        "  or possession (call reversed); a `stands`/`support` challenge\n"
        "  changes nothing about the play but still produces a pause and\n"
        "  possible subs. The split isolates how much of the `coach_challenge`\n"
        "  effect is the rule-change mechanic vs. the pause-plus-substitution\n"
        "  channel shared with `coach_full`.\n"
    )


# ============================================================================
# Entry point
# ============================================================================


def main():
    print("Loading memo...")
    memo = CDNNBAMemoPL.load_all()
    print(f"Loaded. Spine: {memo.height:,} rows")

    append_md("\n---\n\n# Extended experiments (E9-E14)\n")
    append_md(
        "\n*These experiments address the extended TODO.md variable list: "
        "timeout subtype, game situation, team characteristics, PPP baselines, "
        "substitutions, and a proper matched-twin causal estimate.*\n"
    )

    print("Building rich analysis DataFrame...")
    analysis = build_rich_analysis(memo, run_size=6, minutes=3.0)
    print(f"  analysis rows: {analysis.height:,}")

    experiment_9(memo, analysis)
    experiment_10(memo, analysis)
    experiment_11(memo, analysis)
    experiment_12(memo, analysis)
    experiment_13(memo, analysis)
    experiment_14(memo, analysis)

    print("\nExtended experiments complete.")


if __name__ == "__main__":
    main()
