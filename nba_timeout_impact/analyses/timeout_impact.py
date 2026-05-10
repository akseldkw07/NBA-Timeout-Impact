"""Timeout-impact analysis on cdnnba play-by-play.

Answers the README research questions:
  Q1. Short-term point differential: pts in N possessions after a timeout vs N before,
      compared to season-long PPP and head-to-head differential.
  Q2. Short-term momentum (PPP): pts-per-possession after vs before, vs season PPP.
  Q3. Long-term: win-probability added (handled in wp_model.py).

The core entrypoint is :func:`build_timeout_events`, which returns one row per
timeout with pre/post window aggregates, context columns, and season baselines
already joined. Downstream summarizers (:func:`summarize_pre_post`,
:func:`summarize_ppp`) group those events and run Welch t-tests against the
appropriate baseline.
"""

from __future__ import annotations

import typing as t

import numpy as np
import polars as pl
from scipy import stats as sp_stats

from nba_timeout_impact.datasets.memo_cdnnba_pl import CDNNBAMemoPL

# --------------------------------------------------------------------------- #
#  Per-game helpers                                                            #
# --------------------------------------------------------------------------- #


def team_pairs_per_game(memo: CDNNBAMemoPL) -> pl.DataFrame:
    """Return one row per (gameId, teamA, teamB) ordered pair (2 per game)."""
    df = pl.DataFrame._from_pydf(memo.cdnnba._df)
    teams = df.filter(pl.col("teamId") > 0).select("gameId", "teamId").unique()
    pairs = (
        teams.rename({"teamId": "teamA"})
        .join(teams.rename({"teamId": "teamB"}), on="gameId")
        .filter(pl.col("teamA") != pl.col("teamB"))
    )
    return pairs


def home_away_per_game(memo: CDNNBAMemoPL) -> pl.DataFrame:
    """Return (gameId, home_teamId, away_teamId) for every game."""
    pairs = team_pairs_per_game(memo)
    home = memo.home_team_per_game  # gameId, home_teamId
    return pairs.rename({"teamA": "home_teamId", "teamB": "away_teamId"}).join(
        home, on=["gameId", "home_teamId"], how="inner"
    )


def final_score_per_game(memo: CDNNBAMemoPL) -> pl.DataFrame:
    """Final scoreHome/scoreAway per game plus home_won flag."""
    df = pl.DataFrame._from_pydf(memo.cdnnba._df)
    return (
        df.group_by("gameId", maintain_order=True)
        .agg(
            pl.col("scoreHome").last().alias("final_home"),
            pl.col("scoreAway").last().alias("final_away"),
            pl.col("game_date").first(),
            pl.col("season").first(),
            pl.col("season_type").first(),
        )
        .with_columns((pl.col("final_home") > pl.col("final_away")).alias("home_won"))
    )


# --------------------------------------------------------------------------- #
#  Season baselines                                                            #
# --------------------------------------------------------------------------- #


def compute_team_season_ppp(memo: CDNNBAMemoPL) -> pl.DataFrame:
    """Per-team season points-per-possession baseline.

    Returns: (season, season_type, teamId, n_poss, ppp).
    """
    poss = memo.possessions
    out = (
        poss.filter(pl.col("possession") > 0)
        .group_by("season", "season_type", pl.col("possession").alias("teamId"))
        .agg(
            pl.col("possession_points").sum().alias("total_pts"),
            pl.len().alias("n_poss"),
        )
        .with_columns((pl.col("total_pts") / pl.col("n_poss")).alias("ppp"))
        .sort("season", "season_type", "teamId")
    )
    return out


def compute_team_season_record(memo: CDNNBAMemoPL) -> pl.DataFrame:
    """Final regular-season win-loss record per (season, teamId).

    Playoff games downstream use the team's full-season RS record.
    Returns: (season, teamId, wins, losses, wpct).
    """
    df = pl.DataFrame._from_pydf(memo.cdnnba._df)
    final = (
        df.filter(pl.col("season_type") == "rg")
        .group_by("gameId", maintain_order=True)
        .agg(
            pl.col("scoreHome").last().alias("final_home"),
            pl.col("scoreAway").last().alias("final_away"),
            pl.col("season").first(),
        )
    )
    pairs = team_pairs_per_game(memo)
    home = memo.home_team_per_game

    team_games = (
        pairs.rename({"teamA": "teamId", "teamB": "opp_teamId"})
        .join(home, on="gameId")
        .join(final, on="gameId", how="inner")
        .with_columns((pl.col("teamId") == pl.col("home_teamId")).alias("is_home"))
        .with_columns(
            pl.when(pl.col("is_home"))
            .then(pl.col("final_home") > pl.col("final_away"))
            .otherwise(pl.col("final_away") > pl.col("final_home"))
            .alias("won")
        )
    )
    return (
        team_games.group_by("season", "teamId")
        .agg(
            pl.col("won").sum().alias("wins"),
            (~pl.col("won")).sum().alias("losses"),
        )
        .with_columns((pl.col("wins") / (pl.col("wins") + pl.col("losses"))).alias("wpct"))
        .sort("season", "teamId")
    )


def compute_h2h_pt_diff(memo: CDNNBAMemoPL) -> pl.DataFrame:
    """Season-long head-to-head differential per possession.

    For every season, season_type, ordered (teamA, teamB) pair, sums teamA's
    point differential across all matchups and divides by total possessions.

    Returns: (season, season_type, teamA, teamB, n_games, total_pt_diff,
              total_poss, pt_diff_per_poss).
    """
    df = pl.DataFrame._from_pydf(memo.cdnnba._df)
    home = memo.home_team_per_game
    pairs = team_pairs_per_game(memo)

    final = df.group_by("gameId", maintain_order=True).agg(
        pl.col("scoreHome").last().alias("final_home"),
        pl.col("scoreAway").last().alias("final_away"),
        pl.col("season").first(),
        pl.col("season_type").first(),
    )

    poss_per_game = memo.possessions.filter(pl.col("possession") > 0).group_by("gameId").agg(pl.len().alias("n_poss"))

    # Build per-team-per-game point diff (teamA - teamB perspective)
    team_games = (
        pairs.rename({"teamA": "teamId_a", "teamB": "teamId_b"})
        .join(home, on="gameId")
        .join(final, on="gameId")
        .join(poss_per_game, on="gameId")
        .with_columns(
            (pl.col("teamId_a") == pl.col("home_teamId")).alias("a_is_home"),
        )
        .with_columns(
            pl.when(pl.col("a_is_home"))
            .then(pl.col("final_home") - pl.col("final_away"))
            .otherwise(pl.col("final_away") - pl.col("final_home"))
            .alias("pt_diff_a"),
        )
    )

    return (
        team_games.group_by("season", "season_type", "teamId_a", "teamId_b")
        .agg(
            pl.len().alias("n_games"),
            pl.col("pt_diff_a").sum().alias("total_pt_diff"),
            pl.col("n_poss").sum().alias("total_poss"),
        )
        .with_columns((pl.col("total_pt_diff") / pl.col("total_poss")).alias("pt_diff_per_poss"))
        .rename({"teamId_a": "calling_team", "teamId_b": "opponent_team"})
        .sort("season", "season_type", "calling_team", "opponent_team")
    )


# --------------------------------------------------------------------------- #
#  Per-timeout event table                                                     #
# --------------------------------------------------------------------------- #


def build_timeout_events(memo: CDNNBAMemoPL, window: int = 6) -> pl.DataFrame:
    """One row per timeout with pre/post window aggregates and context.

    Window logic:
      Looks at the *window* possessions immediately preceding and the *window*
      possessions immediately following the timeout's possession. Each side
      mixes both teams (since possessions alternate); per-team subtotals come
      out of the lag/lead arithmetic.

    Calling-team convention:
      - Coach timeouts (full / challenge): teamId on the timeout row.
      - Exogenous (official_inferred): the team possessing at the timeout
        (since there is no team-of-record on the row). This means exogenous
        rows always have ``calling_team == possession`` at the timeout.

    Output columns:
      identity:    gameId, possession_id, period, game_seconds_elapsed,
                   seconds_remaining, season, season_type, IsPlayoff
      timeout:     timeout_subtype, calling_team, opponent_team,
                   is_home_calling, is_endogenous
      context:     score_margin, streak_signed (calling-team perspective),
                   time_bucket, margin_bucket, streak_bucket
      windows:     pts_for_pre, pts_against_pre, pts_for_post, pts_against_post
                   net_pre, net_post, net_change
                   n_for_pre, n_against_pre, n_for_post, n_against_post
                   ppp_for_pre, ppp_against_pre, ppp_for_post, ppp_against_post
      baselines:   calling_team_ppp, opponent_team_ppp, h2h_pt_diff_per_poss
                   expected_net_per_poss = (calling_ppp - opp_ppp)
                   expected_net_window = expected_net_per_poss * (window / 2 -- per side)
                   excess_net_post = net_post - expected_net_window
                   excess_ppp_post = ppp_for_post - calling_team_ppp
    """
    W = window
    if W % 2 != 0:
        raise ValueError(f"window must be even (got {W}); each team gets W/2 possessions per side.")

    df = pl.DataFrame._from_pydf(memo.cdnnba._df)
    poss = memo.possessions
    streak_s = memo.streak

    # ---- 1. Per-possession lag/lead arrays ----
    p = poss.select("gameId", "possession_id", "possession_points", "possession").sort("gameId", "possession_id")
    for i in range(1, W + 1):
        p = p.with_columns(
            pl.col("possession_points").shift(i).over("gameId").alias(f"pts_lag{i}"),
            pl.col("possession").shift(i).over("gameId").alias(f"tm_lag{i}"),
            pl.col("possession_points").shift(-i).over("gameId").alias(f"pts_lead{i}"),
            pl.col("possession").shift(-i).over("gameId").alias(f"tm_lead{i}"),
        )

    # ---- 2. Timeout rows with row-level context ----
    timeouts = (
        df.with_columns(streak_s.alias("_streak"))
        .filter(pl.col("actionType") == "timeout")
        .filter(pl.col("possession_id").is_not_null())
        .select(
            "gameId",
            "possession_id",
            "period",
            "game_seconds_elapsed",
            "seconds_remaining",
            "score_margin",
            "season",
            "season_type",
            "IsPlayoff",
            pl.col("subType").alias("timeout_subtype"),
            pl.col("teamId").alias("_teamId_raw"),
            pl.col("_streak").alias("_streak_home"),
        )
    )

    # ---- 3. Join + determine calling team ----
    events = timeouts.join(p, on=["gameId", "possession_id"], how="inner")
    events = events.with_columns(
        pl.when(pl.col("timeout_subtype").is_in(["full", "challenge"]))
        .then(pl.col("_teamId_raw"))
        .otherwise(pl.col("possession"))
        .alias("calling_team"),
        pl.col("timeout_subtype").is_in(["full", "challenge"]).alias("is_endogenous"),
    )

    # Attach opponent_team via team_pairs
    pairs = team_pairs_per_game(memo).rename({"teamA": "calling_team", "teamB": "opponent_team"})
    events = events.join(pairs, on=["gameId", "calling_team"], how="left")

    # is_home_calling
    home = memo.home_team_per_game
    events = events.join(home, on="gameId", how="left").with_columns(
        (pl.col("calling_team") == pl.col("home_teamId")).alias("is_home_calling")
    )

    # ---- 4. Pre/post point totals + counts (calling-team perspective) ----
    pts_for_pre = pl.lit(0)
    pts_against_pre = pl.lit(0)
    pts_for_post = pl.lit(0)
    pts_against_post = pl.lit(0)
    n_for_pre = pl.lit(0)
    n_against_pre = pl.lit(0)
    n_for_post = pl.lit(0)
    n_against_post = pl.lit(0)
    for i in range(1, W + 1):
        same_lag = pl.col(f"tm_lag{i}") == pl.col("calling_team")
        same_lead = pl.col(f"tm_lead{i}") == pl.col("calling_team")
        pts_for_pre = pts_for_pre + pl.when(same_lag).then(pl.col(f"pts_lag{i}")).otherwise(0)
        pts_against_pre = pts_against_pre + pl.when(~same_lag).then(pl.col(f"pts_lag{i}")).otherwise(0)
        pts_for_post = pts_for_post + pl.when(same_lead).then(pl.col(f"pts_lead{i}")).otherwise(0)
        pts_against_post = pts_against_post + pl.when(~same_lead).then(pl.col(f"pts_lead{i}")).otherwise(0)
        n_for_pre = n_for_pre + same_lag.cast(pl.Int32)
        n_against_pre = n_against_pre + (~same_lag).cast(pl.Int32)
        n_for_post = n_for_post + same_lead.cast(pl.Int32)
        n_against_post = n_against_post + (~same_lead).cast(pl.Int32)

    events = events.with_columns(
        pts_for_pre.alias("pts_for_pre"),
        pts_against_pre.alias("pts_against_pre"),
        pts_for_post.alias("pts_for_post"),
        pts_against_post.alias("pts_against_post"),
        n_for_pre.alias("n_for_pre"),
        n_against_pre.alias("n_against_pre"),
        n_for_post.alias("n_for_post"),
        n_against_post.alias("n_against_post"),
    )

    # Drop rows that don't have a full window on both sides
    events = events.drop_nulls(subset=[f"pts_lag{W}", f"pts_lead{W}"])

    events = events.with_columns(
        (pl.col("pts_for_pre") - pl.col("pts_against_pre")).alias("net_pre"),
        (pl.col("pts_for_post") - pl.col("pts_against_post")).alias("net_post"),
    ).with_columns(
        (pl.col("net_post") - pl.col("net_pre")).alias("net_change"),
        (pl.col("pts_for_pre") / pl.col("n_for_pre")).alias("ppp_for_pre"),
        (pl.col("pts_against_pre") / pl.col("n_against_pre")).alias("ppp_against_pre"),
        (pl.col("pts_for_post") / pl.col("n_for_post")).alias("ppp_for_post"),
        (pl.col("pts_against_post") / pl.col("n_against_post")).alias("ppp_against_post"),
    )

    # ---- 5. Streak from calling-team perspective (positive = calling team on a run) ----
    # streak_home is +ve when home is on a run. Flip if calling is away.
    events = events.with_columns(
        pl.when(pl.col("is_home_calling"))
        .then(pl.col("_streak_home"))
        .otherwise(-pl.col("_streak_home"))
        .alias("streak_signed"),
    )

    # ---- 6. Context buckets ----
    events = add_context_buckets(events)

    # ---- 7. Drop helper columns ----
    helper_cols = [c for c in events.columns if c.startswith(("pts_lag", "tm_lag", "pts_lead", "tm_lead"))]
    helper_cols += ["_teamId_raw", "_streak_home", "possession", "possession_points", "home_teamId"]
    events = events.drop([c for c in helper_cols if c in events.columns])

    return events


def add_context_buckets(events: pl.DataFrame) -> pl.DataFrame:
    """Attach categorical buckets used for slicing in summarizers.

    time_bucket   in {"Q1","Q2","Q3","Q4_early","Q4_clutch","OT"}
    margin_bucket in {"down15+","down6-15","down1-5","tied","up1-5","up6-15","up15+"}
                  (from the calling team's perspective; tied = |margin|<=0)
    streak_bucket in {"down10+","down6-9","down3-5","calm","up3-5","up6-9","up10+"}
                  (calling team perspective)
    """
    # Margin from calling-team perspective
    margin_calling = pl.when(pl.col("is_home_calling")).then(pl.col("score_margin")).otherwise(-pl.col("score_margin"))

    return events.with_columns(
        pl.when(pl.col("period") == 1)
        .then(pl.lit("Q1"))
        .when(pl.col("period") == 2)
        .then(pl.lit("Q2"))
        .when(pl.col("period") == 3)
        .then(pl.lit("Q3"))
        .when((pl.col("period") == 4) & (pl.col("seconds_remaining") > 300))
        .then(pl.lit("Q4_early"))
        .when(pl.col("period") == 4)
        .then(pl.lit("Q4_clutch"))
        .otherwise(pl.lit("OT"))
        .alias("time_bucket"),
        margin_calling.alias("margin_calling"),
        pl.when(margin_calling <= -16)
        .then(pl.lit("down15+"))
        .when(margin_calling <= -6)
        .then(pl.lit("down6-15"))
        .when(margin_calling <= -1)
        .then(pl.lit("down1-5"))
        .when(margin_calling == 0)
        .then(pl.lit("tied"))
        .when(margin_calling <= 5)
        .then(pl.lit("up1-5"))
        .when(margin_calling <= 15)
        .then(pl.lit("up6-15"))
        .otherwise(pl.lit("up15+"))
        .alias("margin_bucket"),
        pl.when(pl.col("streak_signed") <= -10)
        .then(pl.lit("down10+"))
        .when(pl.col("streak_signed") <= -6)
        .then(pl.lit("down6-9"))
        .when(pl.col("streak_signed") <= -3)
        .then(pl.lit("down3-5"))
        .when(pl.col("streak_signed").abs() <= 2)
        .then(pl.lit("calm"))
        .when(pl.col("streak_signed") <= 5)
        .then(pl.lit("up3-5"))
        .when(pl.col("streak_signed") <= 9)
        .then(pl.lit("up6-9"))
        .otherwise(pl.lit("up10+"))
        .alias("streak_bucket"),
    )


# --------------------------------------------------------------------------- #
#  Attaching season baselines                                                  #
# --------------------------------------------------------------------------- #


def attach_baselines(
    events: pl.DataFrame,
    team_ppp: pl.DataFrame,
    h2h: pl.DataFrame,
    *,
    window: int = 6,
) -> pl.DataFrame:
    """Left-join per-team season PPP, opponent PPP, and head-to-head margin/poss.

    Adds columns:
      calling_team_ppp, opponent_team_ppp,
      expected_net_per_poss = calling_ppp - opp_ppp,
      expected_net_window   = expected_net_per_poss * (window / 2),
      h2h_pt_diff_per_poss,
      h2h_expected_net_window,
      excess_net_post    = net_post - expected_net_window,
      excess_h2h_net_post = net_post - h2h_expected_net_window,
      excess_ppp_for_post = ppp_for_post - calling_team_ppp.
    """
    W_per_team = window // 2

    calling_ppp = team_ppp.select(
        "season",
        "season_type",
        pl.col("teamId").alias("calling_team"),
        pl.col("ppp").alias("calling_team_ppp"),
    )
    opp_ppp = team_ppp.select(
        "season",
        "season_type",
        pl.col("teamId").alias("opponent_team"),
        pl.col("ppp").alias("opponent_team_ppp"),
    )

    h2h_join = h2h.select(
        "season",
        "season_type",
        "calling_team",
        "opponent_team",
        pl.col("pt_diff_per_poss").alias("h2h_pt_diff_per_poss"),
    )

    out = (
        events.join(calling_ppp, on=["season", "season_type", "calling_team"], how="left")
        .join(opp_ppp, on=["season", "season_type", "opponent_team"], how="left")
        .join(h2h_join, on=["season", "season_type", "calling_team", "opponent_team"], how="left")
        .with_columns(
            (pl.col("calling_team_ppp") - pl.col("opponent_team_ppp")).alias("expected_net_per_poss"),
        )
        .with_columns(
            (pl.col("expected_net_per_poss") * W_per_team).alias("expected_net_window"),
            (pl.col("h2h_pt_diff_per_poss") * W_per_team).alias("h2h_expected_net_window"),
        )
        .with_columns(
            (pl.col("net_post") - pl.col("expected_net_window")).alias("excess_net_post"),
            (pl.col("net_post") - pl.col("h2h_expected_net_window")).alias("excess_h2h_net_post"),
            (pl.col("ppp_for_post") - pl.col("calling_team_ppp")).alias("excess_ppp_for_post"),
        )
    )
    return out


# --------------------------------------------------------------------------- #
#  Summarizers                                                                 #
# --------------------------------------------------------------------------- #


def _welch_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Welch t-test (two-sided). Returns (t_stat, p_value). NaN-safe."""
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return (float("nan"), float("nan"))
    t_stat, p = sp_stats.ttest_ind(a, b, equal_var=False)
    return float(t_stat), float(p)


def summarize_pre_post(
    events: pl.DataFrame,
    group_cols: t.Sequence[str] = (),
    *,
    metric: str = "net",
) -> pl.DataFrame:
    """Per-group summary of pre/post window outcomes with a Welch t-test.

    metric:
      - "net"  → compares ``net_pre`` to ``net_post`` (Q1 in README).
      - "ppp"  → compares ``ppp_for_pre`` to ``ppp_for_post`` (Q2 momentum).

    Adds an ``excess_vs_baseline`` column when baselines are present:
      - "net"  → mean(net_post - expected_net_window)
      - "ppp"  → mean(ppp_for_post - calling_team_ppp)
    """
    if metric == "net":
        pre, post = "net_pre", "net_post"
        excess_col = "excess_net_post" if "excess_net_post" in events.columns else None
        excess_h2h_col = "excess_h2h_net_post" if "excess_h2h_net_post" in events.columns else None
    elif metric == "ppp":
        pre, post = "ppp_for_pre", "ppp_for_post"
        excess_col = "excess_ppp_for_post" if "excess_ppp_for_post" in events.columns else None
        excess_h2h_col = None
    else:
        raise ValueError(f"unknown metric {metric!r}; use 'net' or 'ppp'.")

    group_cols = list(group_cols)
    df = events.filter(pl.col(pre).is_not_null() & pl.col(post).is_not_null())

    aggs = [
        pl.len().alias("n"),
        pl.col(pre).mean().alias(f"{pre}_mean"),
        pl.col(post).mean().alias(f"{post}_mean"),
        (pl.col(post) - pl.col(pre)).mean().alias("delta_mean"),
        (pl.col(post) - pl.col(pre)).std().alias("delta_std"),
    ]
    if excess_col:
        aggs.append(pl.col(excess_col).mean().alias("excess_vs_baseline"))
    if excess_h2h_col:
        aggs.append(pl.col(excess_h2h_col).mean().alias("excess_vs_h2h"))

    if group_cols:
        summary = df.group_by(group_cols).agg(*aggs).sort(group_cols)
    else:
        summary = df.select(*aggs)

    # Welch t-test per group (post vs pre)
    tstats: list[float] = []
    pvals: list[float] = []
    if group_cols:
        for row in summary.iter_rows(named=True):
            mask = pl.lit(True)
            for c in group_cols:
                mask = mask & (pl.col(c) == row[c])
            sub = df.filter(mask)
            t_stat, p = _welch_t(sub[pre].to_numpy(), sub[post].to_numpy())
            tstats.append(t_stat)
            pvals.append(p)
    else:
        t_stat, p = _welch_t(df[pre].to_numpy(), df[post].to_numpy())
        tstats.append(t_stat)
        pvals.append(p)

    summary = summary.with_columns(
        pl.Series("t_stat", tstats),
        pl.Series("p_value", pvals),
    )
    return summary


def summarize_overall_table(events: pl.DataFrame) -> pl.DataFrame:
    """One-row overall report: n, mean pre, mean post, delta, excess vs baselines.

    Splits by ``is_endogenous`` for convenience.
    """
    rows = []
    for label, sub in (
        ("all", events),
        ("endogenous (coach)", events.filter(pl.col("is_endogenous"))),
        ("exogenous (TV/inferred)", events.filter(~pl.col("is_endogenous"))),
    ):
        if sub.height == 0:
            continue
        net_t, net_p = _welch_t(sub["net_pre"].to_numpy(), sub["net_post"].to_numpy())
        rows.append(
            {
                "group": label,
                "n": sub.height,
                "net_pre_mean": float(sub["net_pre"].mean()),
                "net_post_mean": float(sub["net_post"].mean()),
                "delta_mean": float((sub["net_post"] - sub["net_pre"]).mean()),
                "ppp_for_pre_mean": float(sub["ppp_for_pre"].mean()),
                "ppp_for_post_mean": float(sub["ppp_for_post"].mean()),
                "excess_net_post_mean": (
                    float(sub["excess_net_post"].mean()) if "excess_net_post" in sub.columns else None
                ),
                "excess_h2h_net_post_mean": (
                    float(sub["excess_h2h_net_post"].mean()) if "excess_h2h_net_post" in sub.columns else None
                ),
                "excess_ppp_for_post_mean": (
                    float(sub["excess_ppp_for_post"].mean()) if "excess_ppp_for_post" in sub.columns else None
                ),
                "t_stat_post_vs_pre": net_t,
                "p_value_post_vs_pre": net_p,
            }
        )
    return pl.DataFrame(rows)
