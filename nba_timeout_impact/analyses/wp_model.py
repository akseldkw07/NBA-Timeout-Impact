"""Win-probability model for cdnnba possession-level snapshots.

Used to answer Q3 in the README: long-term impact of timeouts measured as
win-probability added.

Pipeline:
  1. :func:`build_wp_dataset` — assembles a possession-start feature matrix
     keyed by (gameId, possession_id) plus a binary home_won label and a
     train/test split. Features include in-game state (margin, period,
     seconds remaining, possessing team, streak, clutch flag), context
     (regular vs playoff), and team-strength priors (each team's full-season
     RS win-percentage, plus the diff).
  2. :func:`train_wp_model` — fits a sklearn ``Pipeline`` of StandardScaler
     plus LogisticRegression on the home-WP target.
  3. :func:`evaluate_wp_model` — log-loss / Brier / AUC on a held-out test
     set of *games* (not rows — we hold out whole games to avoid leakage
     across the same game).
  4. :func:`compute_wp_added` — for each timeout event, predicts home WP at
     the start of the pre-window and at the end of the post-window, signs
     the delta from the calling team's perspective, and joins the results
     onto the events dataframe.
"""

from __future__ import annotations

import typing as t

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_timeout_impact.analyses.timeout_impact import (
    compute_team_season_record,
    final_score_per_game,
    home_away_per_game,
)
from nba_timeout_impact.datasets.memo_cdnnba_pl import CDNNBAMemoPL

# Features the WP model consumes. Keep this list aligned across train/predict.
WP_FEATURES: tuple[str, ...] = (
    "score_margin",  # signed (home - away), home perspective
    "period",
    "seconds_remaining",  # in current period
    "is_clutch",  # bool, already on the spine
    "poss_is_home",  # bool, possession team == home
    "streak_home",  # signed, home perspective
    "is_playoff",  # bool
    "home_wpct",
    "away_wpct",
    "wpct_diff",  # home_wpct - away_wpct
)
LABEL_COL = "home_won"


# --------------------------------------------------------------------------- #
#  Snapshot / training data construction                                       #
# --------------------------------------------------------------------------- #


def _seconds_remaining_in_game(period: pl.Expr, sec_remaining: pl.Expr) -> pl.Expr:
    """Approximate seconds left in regulation. OT clamped to 0.

    Reg periods 1-4 are 720s each, OT periods are 300s each. Game time elapsed
    is already on the spine; this helper produces a single scalar useful as a
    WP feature even when ``period`` is one-hot.
    """
    return pl.when(period <= 4).then((4 - period) * 720 + sec_remaining).otherwise(pl.lit(0))


def _possession_snapshots(memo: CDNNBAMemoPL) -> pl.DataFrame:
    """Possession-start snapshots with all WP-feature columns.

    One row per (gameId, possession_id). Pulls features from the spine's
    first row of each possession plus joins to home/away identity, final
    outcome, and per-team season RS records.
    """
    df = pl.DataFrame._from_pydf(memo.cdnnba._df)
    streak_s = memo.streak

    # Take the first event of each possession as the snapshot
    full = df.with_columns(streak_s.alias("_streak_home"))
    snaps = (
        full.filter(pl.col("possession_id").is_not_null())
        .filter(pl.col("possession") > 0)
        .group_by("gameId", "possession_id", maintain_order=True)
        .agg(
            pl.col("period").first(),
            pl.col("seconds_remaining").first(),
            pl.col("game_seconds_elapsed").first(),
            pl.col("score_margin").first(),
            pl.col("is_clutch").first(),
            pl.col("possession").first().alias("poss_team"),
            pl.col("_streak_home").first().alias("streak_home"),
            pl.col("season").first(),
            pl.col("season_type").first(),
            pl.col("IsPlayoff").first(),
        )
    )

    final = final_score_per_game(memo).select("gameId", "home_won")
    ha = home_away_per_game(memo)
    records = compute_team_season_record(memo)

    home_wpct = records.select("season", pl.col("teamId").alias("home_teamId"), pl.col("wpct").alias("home_wpct"))
    away_wpct = records.select("season", pl.col("teamId").alias("away_teamId"), pl.col("wpct").alias("away_wpct"))

    snaps = (
        snaps.join(ha, on="gameId", how="left")
        .join(final, on="gameId", how="left")
        .join(home_wpct, on=["season", "home_teamId"], how="left")
        .join(away_wpct, on=["season", "away_teamId"], how="left")
        .with_columns(
            (pl.col("poss_team") == pl.col("home_teamId")).alias("poss_is_home"),
            pl.col("IsPlayoff").alias("is_playoff"),
            (pl.col("home_wpct") - pl.col("away_wpct")).alias("wpct_diff"),
        )
    )
    # Some 2024-25 playoff teams may not have an RS record row if data is partial
    # — fill with 0.5 so the model still scores them.
    snaps = snaps.with_columns(
        pl.col("home_wpct").fill_null(0.5),
        pl.col("away_wpct").fill_null(0.5),
        pl.col("wpct_diff").fill_null(0.0),
    )
    return snaps


def build_wp_dataset(
    memo: CDNNBAMemoPL,
    *,
    test_frac: float = 0.2,
    seed: int = 42,
    sample_n: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pl.DataFrame]:
    """Construct (X_train, X_test, y_train, y_test, snapshots).

    Holds out whole games (not rows) so no game contributes to both splits.
    Returns the full snapshots dataframe alongside so downstream callers can
    score per-event states without rebuilding.
    """
    snaps = _possession_snapshots(memo)
    snaps = snaps.drop_nulls(subset=[LABEL_COL])

    # Optional sub-sample to speed training
    if sample_n is not None and sample_n < snaps.height:
        snaps_sampled = snaps.sample(n=sample_n, seed=seed)
    else:
        snaps_sampled = snaps

    rng = np.random.default_rng(seed)
    games = snaps_sampled["gameId"].unique().to_numpy()
    rng.shuffle(games)
    cut = int(len(games) * (1 - test_frac))
    train_games = set(games[:cut].tolist())

    train_mask = snaps_sampled["gameId"].is_in(list(train_games))
    train = snaps_sampled.filter(train_mask)
    test = snaps_sampled.filter(~train_mask)

    feature_cols = list(WP_FEATURES)
    X_train = train.select(feature_cols).to_pandas()
    X_test = test.select(feature_cols).to_pandas()
    y_train = train[LABEL_COL].to_pandas().astype(int)
    y_test = test[LABEL_COL].to_pandas().astype(int)

    # Cast booleans to int for sklearn
    for col in ["is_clutch", "poss_is_home", "is_playoff"]:
        X_train[col] = X_train[col].astype(int)
        X_test[col] = X_test[col].astype(int)

    return X_train, X_test, y_train, y_test, snaps


# --------------------------------------------------------------------------- #
#  Training and evaluation                                                     #
# --------------------------------------------------------------------------- #


def train_wp_model(X_train: pd.DataFrame, y_train: pd.Series, *, C: float = 1.0) -> Pipeline:
    """Fit StandardScaler + LogisticRegression. Returns the fitted pipeline."""
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=C, max_iter=2000, n_jobs=-1)),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe


def evaluate_wp_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Compute log-loss, Brier score, and ROC-AUC on the held-out set."""
    p = model.predict_proba(X_test)[:, 1]
    return {
        "n_test": int(len(y_test)),
        "log_loss": float(log_loss(y_test, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y_test, p)),
        "auc": float(roc_auc_score(y_test, p)),
        "base_rate_home_won": float(y_test.mean()),
    }


def coef_table(model: Pipeline, feature_names: t.Sequence[str] = WP_FEATURES) -> pd.DataFrame:
    """Return a tidy table of standardized coefficients, sorted by |coef|."""
    lr: LogisticRegression = model.named_steps["lr"]
    coefs = lr.coef_[0]
    out = pd.DataFrame({"feature": list(feature_names), "coef": coefs})
    out["abs_coef"] = out["coef"].abs()
    return out.sort_values("abs_coef", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
#  WP added at timeouts                                                        #
# --------------------------------------------------------------------------- #


def _features_at(snaps: pl.DataFrame, keys: pl.DataFrame) -> pl.DataFrame:
    """Look up snapshot feature rows for a list of (gameId, possession_id) keys."""
    return keys.join(snaps, on=["gameId", "possession_id"], how="left")


def compute_wp_added(
    events: pl.DataFrame,
    snaps: pl.DataFrame,
    model: Pipeline,
    *,
    window: int = 6,
) -> pl.DataFrame:
    """For each timeout event, predict home WP at pre-window-start and
    post-window-end and report the WP delta from the calling team's perspective.

    Pre-window-start = possession (timeout_poss - window) (or the earliest
    available row if not enough lead-in).
    Post-window-end  = possession (timeout_poss + window).

    Adds columns:
      wp_pre_home, wp_post_home    — model output (home perspective)
      wp_pre_calling, wp_post_calling
      wp_added_calling             — wp_post_calling - wp_pre_calling
    """
    feature_cols = list(WP_FEATURES)
    snaps_keys = snaps.select("gameId", "possession_id", *feature_cols)

    pre_keys = events.select(
        "gameId",
        (pl.col("possession_id") - window).alias("possession_id"),
        pl.col("possession_id").alias("_event_poss_id"),
    )
    post_keys = events.select(
        "gameId",
        (pl.col("possession_id") + window).alias("possession_id"),
        pl.col("possession_id").alias("_event_poss_id"),
    )

    pre = _features_at(snaps_keys, pre_keys).rename({"possession_id": "_pre_poss_id"})
    post = _features_at(snaps_keys, post_keys).rename({"possession_id": "_post_poss_id"})

    pre_X = pre.select(feature_cols).to_pandas()
    post_X = post.select(feature_cols).to_pandas()
    for col in ["is_clutch", "poss_is_home", "is_playoff"]:
        pre_X[col] = pre_X[col].astype("Int64").astype(float)
        post_X[col] = post_X[col].astype("Int64").astype(float)

    # Predict only on rows that have all features; rows with any NaN keep NaN WP.
    pre_mask = pre_X.notna().all(axis=1).to_numpy()
    post_mask = post_X.notna().all(axis=1).to_numpy()

    wp_pre = np.full(len(pre_X), np.nan)
    wp_post = np.full(len(post_X), np.nan)
    if pre_mask.any():
        wp_pre[pre_mask] = model.predict_proba(pre_X[pre_mask])[:, 1]
    if post_mask.any():
        wp_post[post_mask] = model.predict_proba(post_X[post_mask])[:, 1]

    out = (
        events.with_columns(
            pl.Series("wp_pre_home", wp_pre),
            pl.Series("wp_post_home", wp_post),
        )
        .with_columns(
            pl.when(pl.col("is_home_calling"))
            .then(pl.col("wp_pre_home"))
            .otherwise(1 - pl.col("wp_pre_home"))
            .alias("wp_pre_calling"),
            pl.when(pl.col("is_home_calling"))
            .then(pl.col("wp_post_home"))
            .otherwise(1 - pl.col("wp_post_home"))
            .alias("wp_post_calling"),
        )
        .with_columns(
            (pl.col("wp_post_calling") - pl.col("wp_pre_calling")).alias("wp_added_calling"),
        )
    )
    return out


def summarize_wp_added(
    events_with_wp: pl.DataFrame,
    group_cols: t.Sequence[str] = (),
) -> pl.DataFrame:
    """Per-group mean WP added (calling team perspective) with a one-sample t-test against 0."""
    group_cols = list(group_cols)
    df = events_with_wp.filter(pl.col("wp_added_calling").is_not_null())

    aggs = [
        pl.len().alias("n"),
        pl.col("wp_pre_calling").mean().alias("wp_pre_mean"),
        pl.col("wp_post_calling").mean().alias("wp_post_mean"),
        pl.col("wp_added_calling").mean().alias("wp_added_mean"),
        pl.col("wp_added_calling").std().alias("wp_added_std"),
    ]

    if group_cols:
        summary = df.group_by(group_cols).agg(*aggs).sort(group_cols)
    else:
        summary = df.select(*aggs)

    # one-sample t-test (H0: mean wp_added == 0)
    tstats: list[float] = []
    pvals: list[float] = []
    rows = [{c: row[c] for c in group_cols} for row in summary.iter_rows(named=True)] if group_cols else [{}]
    for row in rows:
        sub = df
        for c, v in row.items():
            sub = sub.filter(pl.col(c) == v)
        vals = sub["wp_added_calling"].to_numpy()
        vals = vals[~np.isnan(vals)]
        if len(vals) >= 2:
            t_stat, p = sp_stats_ttest_1samp_zero(vals)
        else:
            t_stat, p = float("nan"), float("nan")
        tstats.append(t_stat)
        pvals.append(p)

    return summary.with_columns(pl.Series("t_stat", tstats), pl.Series("p_value", pvals))


def sp_stats_ttest_1samp_zero(vals: np.ndarray) -> tuple[float, float]:
    """One-sample t-test against 0, NaN-safe."""
    from scipy import stats as sp_stats

    t_stat, p = sp_stats.ttest_1samp(vals, 0.0)
    return float(t_stat), float(p)
