"""Validation harness for the rulebook-based TV-timeout injection.

Compares ``CDNNBADatasetPL.infer_tv_timeouts_rulebook`` predictions against
the nbastatsv3 ground-truth labels (``actionType=="Timeout"`` with
``subType in {"Official", "Official TV"}``) over the v3 labeled era
(1998-2016 RS, 1998-2015 PO).

Usage
-----
    from nba_timeout_impact.datasets.memo_nbastatsv3 import NBAMemoDF
    from nba_timeout_impact.analyses.tv_timeout_validation import (
        validate_injection, compare_configs, POST_2017, PRE_2017_CANDIDATES,
    )

    memo = NBAMemoDF.load_all()
    print(validate_injection(memo, **POST_2017))
    compare_configs(memo, [POST_2017, *PRE_2017_CANDIDATES])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterable

import pandas as pd
import polars as pl

from nba_timeout_impact.datasets.enriched_cdnnba import CDNNBADatasetPL
from nba_timeout_impact.datasets.memo_nbastatsv3 import NBAMemoDF

# v3-specific column conventions
V3_COACH_SUBTYPES = ["Regular", "Short", "Coach Challenge"]
V3_DEAD_BALL_ACTION_TYPES = {
    "Foul",
    "Free Throw",
    "Substitution",
    "Turnover",
    "Violation",
    "Jump Ball",
    "Timeout",
    "Ejection",
    "Instant Replay",
}

# --- Threshold presets ---------------------------------------------------

POST_2017: dict = {
    "label": "post-2017 (7:00, 3:00) x Q1-Q4",
    "thresholds": [420, 180],
    "periods": [1, 2, 3, 4],
}

# Pre-2017 rulebook predates a single clean source, so we sweep plausible
# configurations and let the harness pick the best.
PRE_2017_CANDIDATES: list[dict] = [
    {"label": "pre-2017 (9:00, 3:00) x Q1-Q4", "thresholds": [540, 180], "periods": [1, 2, 3, 4]},
    {"label": "pre-2017 (9:00, 6:00, 3:00) x Q1-Q4", "thresholds": [540, 360, 180], "periods": [1, 2, 3, 4]},
    {"label": "pre-2017 (9:00, 6:00, 3:00) x Q2/Q4 + (6:00) x Q1/Q3", "thresholds": [540, 360, 180], "periods": [2, 4]},
]


# --- Result types --------------------------------------------------------


@dataclass
class ValidationResult:
    label: str
    thresholds: list[int]
    periods: list[int]
    tolerance_s: int
    seasons: tuple[int, int]
    n_gt: int
    n_pred: int
    tp: int
    fp: int
    fn: int
    per_season: pd.DataFrame = field(repr=False)
    per_period: pd.DataFrame = field(repr=False)

    @property
    def precision(self) -> float:
        return self.tp / max(self.tp + self.fp, 1)

    @property
    def recall(self) -> float:
        return self.tp / max(self.tp + self.fn, 1)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / max(p + r, 1e-9)

    def summary(self) -> str:
        return (
            f"{self.label}: "
            f"n_gt={self.n_gt:,} n_pred={self.n_pred:,} "
            f"TP={self.tp:,} FP={self.fp:,} FN={self.fn:,} | "
            f"P={self.precision:.3f} R={self.recall:.3f} F1={self.f1:.3f}"
        )

    def __repr__(self) -> str:
        return f"<ValidationResult {self.summary()}>"


# --- Greedy clock-matching ------------------------------------------------


def _greedy_match(gt: list[int], pred: list[int], tol: int) -> tuple[int, int, int]:
    """Greedy nearest-clock match. Returns (tp, fp, fn).

    Each ground-truth row is matched to the closest unmatched prediction
    within ``tol`` seconds. Both lists are short (≤4 entries per
    (game, period)) so O(N*M) is fine.
    """
    remaining = list(pred)
    tp = 0
    for g in gt:
        best_i = -1
        best_d = tol + 1
        for i, p in enumerate(remaining):
            d = abs(p - g)
            if d < best_d:
                best_d = d
                best_i = i
        if best_i >= 0:
            tp += 1
            remaining.pop(best_i)
    fp = len(remaining)
    fn = len(gt) - tp
    return tp, fp, fn


# --- Main entry point -----------------------------------------------------


def _prep_v3_for_inference(memo: NBAMemoDF, seasons: tuple[int, int]) -> pl.DataFrame:
    """Pull v3 PBP for the season range, strip string columns, return polars."""
    v3 = memo.data
    sub = v3[(v3["season"] >= seasons[0]) & (v3["season"] <= seasons[1])][
        [
            "gameId",
            "actionNumber",
            "period",
            "clock",
            "actionType",
            "subType",
            "seconds_remaining",
            "season",
            "season_type",
        ]
    ].copy()
    # The raw parquet preserves trailing whitespace in some categoricals
    # ("Foul " vs "Foul"). Strip for clean string equality.
    for col in ("actionType", "subType"):
        sub[col] = sub[col].astype("string").str.strip()
    return pl.from_pandas(sub)


def validate_injection(
    memo: NBAMemoDF,
    thresholds: list[int],
    periods: list[int] | None = None,
    seasons: tuple[int, int] = (1998, 2016),
    tolerance_s: int = 60,
    label: str = "",
    multi_fire: bool = False,
) -> ValidationResult:
    """Run rulebook injection on v3 PBP and score against v3 labeled timeouts.

    Parameters
    ----------
    memo : NBAMemoDF
        Loaded NBA memo (with v3 data + derived clock columns).
    thresholds : list[int]
        Mandatory clock thresholds in *seconds remaining* (e.g. ``[420, 180]``
        for post-2017 7:00 / 3:00).
    periods : list[int] | None
        Periods to apply the rule to. Default ``[1, 2, 3, 4]``.
    seasons : (int, int)
        Inclusive season range to evaluate over.
    tolerance_s : int
        Game-clock tolerance for matching a predicted mandatory to a labeled
        one within the same (gameId, period).
    label : str
        Human-readable label for the config (filled if empty).
    """
    if periods is None:
        periods = [1, 2, 3, 4]
    if not label:
        label = f"thresholds={thresholds} periods={periods}"

    v3_pl = _prep_v3_for_inference(memo, seasons)

    # --- Predictions
    hits = CDNNBADatasetPL.infer_tv_timeouts_rulebook(
        v3_pl,
        thresholds=thresholds,
        periods=periods,
        coach_to_subtypes=V3_COACH_SUBTYPES,
        dead_ball_action_types=V3_DEAD_BALL_ACTION_TYPES,
        order_col="actionNumber",
        multi_fire=multi_fire,
    )
    pred = hits.select(
        pl.col("gameId"),
        pl.col("period"),
        pl.col("seconds_remaining").round().cast(pl.Int64).alias("pred_sr"),
    )

    # --- Ground truth: Official / Official TV in the validated period set
    gt = v3_pl.filter(
        (pl.col("actionType") == "Timeout")
        & pl.col("subType").is_in(["Official", "Official TV"])
        & pl.col("period").is_in(periods)
    ).select(
        "gameId",
        "period",
        "season",
        pl.col("seconds_remaining").round().cast(pl.Int64).alias("gt_sr"),
    )

    # season map for prediction rows (predictions don't carry season directly)
    season_map = v3_pl.select("gameId", "season").unique()
    pred = pred.join(season_map, on="gameId", how="left")

    pred_g = pred.group_by(["gameId", "period", "season"], maintain_order=True).agg(pl.col("pred_sr"))
    gt_g = gt.group_by(["gameId", "period", "season"], maintain_order=True).agg(pl.col("gt_sr"))
    buckets = pred_g.join(gt_g, on=["gameId", "period", "season"], how="full", coalesce=True)

    tp = fp = fn = 0
    by_season: dict[int, list[int]] = {}
    by_period: dict[int, list[int]] = {}
    for row in buckets.iter_rows(named=True):
        gt_list = [int(x) for x in (row["gt_sr"] or [])]
        pred_list = [int(x) for x in (row["pred_sr"] or [])]
        a, b, c = _greedy_match(gt_list, pred_list, tolerance_s)
        tp += a
        fp += b
        fn += c
        s = row["season"]
        p = row["period"]
        by_season.setdefault(s, [0, 0, 0])
        by_period.setdefault(p, [0, 0, 0])
        for arr, val in ((by_season[s], (a, b, c)), (by_period[p], (a, b, c))):
            arr[0] += val[0]
            arr[1] += val[1]
            arr[2] += val[2]

    def _scores(t: int, f: int, n: int) -> tuple[float, float, float]:
        p = t / max(t + f, 1)
        r = t / max(t + n, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        return p, r, f1

    per_season_df = pd.DataFrame(
        [
            {"season": s, "TP": v[0], "FP": v[1], "FN": v[2], **dict(zip(["precision", "recall", "f1"], _scores(*v)))}
            for s, v in sorted(by_season.items())
        ]
    )
    per_period_df = pd.DataFrame(
        [
            {"period": p, "TP": v[0], "FP": v[1], "FN": v[2], **dict(zip(["precision", "recall", "f1"], _scores(*v)))}
            for p, v in sorted(by_period.items())
        ]
    )

    return ValidationResult(
        label=label,
        thresholds=list(thresholds),
        periods=list(periods),
        tolerance_s=tolerance_s,
        seasons=seasons,
        n_gt=tp + fn,
        n_pred=tp + fp,
        tp=tp,
        fp=fp,
        fn=fn,
        per_season=per_season_df,
        per_period=per_period_df,
    )


def compare_configs(
    memo: NBAMemoDF,
    configs: Iterable[dict],
    seasons: tuple[int, int] = (1998, 2016),
    tolerance_s: int = 60,
) -> pd.DataFrame:
    """Run validate_injection over multiple configs and return a comparison table.

    Each config is a dict with keys ``label``, ``thresholds``, ``periods``,
    and optionally ``multi_fire`` (default ``False``).
    """
    rows = []
    for cfg in configs:
        kwargs = {k: v for k, v in cfg.items() if k != "label"}
        res = validate_injection(memo, seasons=seasons, tolerance_s=tolerance_s, label=cfg.get("label", ""), **kwargs)
        rows.append(
            {
                "config": res.label,
                "multi_fire": kwargs.get("multi_fire", False),
                "n_gt": res.n_gt,
                "n_pred": res.n_pred,
                "TP": res.tp,
                "FP": res.fp,
                "FN": res.fn,
                "precision": round(res.precision, 4),
                "recall": round(res.recall, 4),
                "f1": round(res.f1, 4),
            }
        )
    return pd.DataFrame(rows)
