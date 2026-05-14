"""TV / mandatory timeout reclassification and validation.

Post-2017, the NBA stopped logging mandatory timeouts as a distinct event
type — they're charged to a team's allotment and look identical to coach
TOs in the cdnnba/nbastats feeds. So instead of *injecting* mandatory rows,
we *reclassify* existing TIMEOUT rows by their position relative to the
rulebook trigger marks.

Pre-2017 (1998–2016 v3 era), the rules were different and v3 still labels
mandatories as ``subType="Official"`` / ``"Official TV"``. The validation
harness scores the same classifier against those ground-truth labels.

Public API:
- ``TVTimeoutValidation.classify_timeouts(df, source)``: label each TIMEOUT
  row's role (``slot_K_mandatory`` / ``slot_K_absorbed`` / ``discretionary``
  / ``challenge``). Auto-dispatches pre/post-2017 by season.
- ``TVTimeoutValidation.validate_against_v3(memo, ...)``: score the
  classifier against v3 ground truth subTypes on the labeled era.
- ``TVTimeoutValidation.per_period_existence_score(...)``: looser per-period
  presence metric.
- ``TVTimeoutValidation.compare_configs(...)``: sweep multiple configs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd
import polars as pl

from nba_timeout_impact.datasets.memo_nbastatsv3 import NBAMemoDF

# -------------------- source-specific column conventions ----------------- #


SOURCE_CONFIGS: dict[str, dict] = {
    "v3": {
        # nbastatsv3 schema (1998+; mandatories explicit only through 2016)
        "timeout_action": "Timeout",
        "coach_subtypes": ["Regular", "Short", "Coach Challenge"],
        "challenge_subtypes": ["Coach Challenge"],
        "order_col": "actionNumber",
    },
    "cdnnba": {
        # cdnnba schema (2020+; mandatories implicit, charged to team)
        "timeout_action": "timeout",
        "coach_subtypes": ["full", "challenge"],
        "challenge_subtypes": ["challenge"],
        "order_col": "orderNumber",
    },
}


# -------------------- era-specific rulebook windows ---------------------- #


# Each entry: (absorb_low, absorb_high). A TO with sr in (low, high] absorbs
# the slot; a TO with sr <= low fires the mandatory.
# Pre-2017: Q2 & Q4 only, three cascading marks at 8:59 / 5:59 / 2:59.
# Post-2017: Q1-Q4, two independent marks at 6:59 / 2:59.

PRE_2017_CASCADING = {
    "slots": [(540, 720), (360, 540), (180, 360)],
    "periods": [2, 4],
    "cascading": True,
}
PRE_2017_INDEPENDENT = {
    "slots": [(540, 720), (360, 540), (180, 360)],
    "periods": [2, 4],
    "cascading": False,
}
POST_2017 = {
    "slots": [(420, 720), (180, 420)],
    "periods": [1, 2, 3, 4],
    "cascading": False,
}


def _detect_era(season: int | None, pre_2017_mode: str = "independent") -> dict:
    """Return rulebook params keyed by season (start year). 2017-18 is the cutoff.

    ``pre_2017_mode``: ``"independent"`` (default) or ``"cascading"``. Empirically
    pre-2017 looks closer to independent firing per the Q2/Q4 Official-count
    distribution (~1.55 per quarter, not ~1.0).
    """
    if season is None or season >= 2017:
        return POST_2017
    return PRE_2017_INDEPENDENT if pre_2017_mode == "independent" else PRE_2017_CASCADING


# -------------------- core classification logic -------------------------- #


def _classify_one_period(
    tos: list[dict],
    rules: dict,
    challenge_subtypes: set[str],
) -> list[str]:
    """Classify timeouts in one (gameId, period) bucket.

    ``tos``: list of dicts with keys ``sr`` (seconds_remaining), ``subType``,
    sorted by event order. ``rules``: era dict with ``slots``, ``periods``,
    ``cascading``. Period is checked at caller; this function trusts that
    ``rules['periods']`` applies (caller skips otherwise).
    """
    n = len(tos)
    labels = ["discretionary"] * n
    slots = rules["slots"]
    cascading = rules["cascading"]

    # For each slot, find the first TO responsible for it
    slot_to_idx: list[int | None] = [None] * len(slots)
    slot_role: list[str | None] = [None] * len(slots)

    for s_idx, (absorb_low, absorb_high) in enumerate(slots):
        for i, to in enumerate(tos):
            if to["subType"] in challenge_subtypes:
                continue
            sr = to["sr"]
            if sr is None or sr > absorb_high:
                continue
            # First TO at or below this slot's ceiling
            if sr > absorb_low:
                slot_role[s_idx] = "absorbed"
            else:
                slot_role[s_idx] = "mandatory"
            slot_to_idx[s_idx] = i
            break

    # Cascading: once any slot fires (mandatory), later slots are suppressed
    if cascading:
        first_fired = next((s for s, r in enumerate(slot_role) if r == "mandatory"), None)
        if first_fired is not None:
            for s in range(first_fired + 1, len(slots)):
                slot_to_idx[s] = None
                slot_role[s] = None

    # Assign labels: each TO gets the lowest slot it's responsible for
    for i, to in enumerate(tos):
        if to["subType"] in challenge_subtypes:
            labels[i] = "challenge"
            continue
        for s_idx, idx in enumerate(slot_to_idx):
            if idx == i and slot_role[s_idx] is not None:
                labels[i] = f"slot_{s_idx + 1}_{slot_role[s_idx]}"
                break
    return labels


# -------------------- public class --------------------------------------- #


@dataclass
class ValidationResult:
    label: str
    seasons: tuple[int, int]
    tolerance_s: int
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


class TVTimeoutValidation:
    """Static-method container for mandatory-timeout reclassification + validation."""

    # ---------- era / source dispatch ----------

    @staticmethod
    def get_source_config(source: Literal["v3", "cdnnba"]) -> dict:
        if source not in SOURCE_CONFIGS:
            raise ValueError(f"unknown source {source!r}, expected one of {list(SOURCE_CONFIGS)}")
        return SOURCE_CONFIGS[source]

    @staticmethod
    def detect_era(season: int | None) -> dict:
        return _detect_era(season)

    # ---------- classification ----------

    @staticmethod
    def classify_timeouts(
        df: pl.DataFrame | pd.DataFrame,
        source: Literal["v3", "cdnnba"],
        pre_2017_mode: Literal["independent", "cascading"] = "independent",
        seasons: tuple[int, int] | None = None,
    ) -> pl.DataFrame:
        """Add a ``timeout_role`` column to every TIMEOUT row in ``df``.

        ``timeout_role`` values:
        - ``slot_K_mandatory``: auto-charged mandatory firing at slot K
        - ``slot_K_absorbed``: voluntary coach TO that absorbed slot K's mandatory
        - ``discretionary``: coach TO with no mandatory role
        - ``challenge``: coach's challenge
        - ``""`` (empty): non-timeout rows

        Era is auto-dispatched per game by ``season`` (≤2016 → pre-2017
        cascading Q2/Q4; ≥2017 → post-2017 independent Q1-Q4). Result is a
        polars DataFrame with the same rows as input plus the new column.

        ``seasons``: optional ``(lo, hi)`` inclusive filter on ``season``
        before classification. Useful for restricting to a sub-era (e.g.
        ``(2013, 2016)`` for late-pre-2017 only).
        """
        cfg = TVTimeoutValidation.get_source_config(source)
        timeout_action = cfg["timeout_action"]
        challenge_subs = set(cfg["challenge_subtypes"])
        order_col = cfg["order_col"]

        if isinstance(df, pd.DataFrame):
            df_pl = pl.from_pandas(df)
        else:
            df_pl = df

        if seasons is not None and "season" in df_pl.columns:
            lo, hi = seasons
            df_pl = df_pl.filter((pl.col("season") >= lo) & (pl.col("season") <= hi))

        # Strip strings (raw parquets sometimes have trailing whitespace)
        df_pl = df_pl.with_columns(
            pl.col("actionType").cast(pl.String).str.strip_chars().alias("_at"),
            pl.col("subType").cast(pl.String).str.strip_chars().alias("_st"),
        )
        # Classify EVERY timeout row — do not use subType to decide whether to
        # process it. (We treat all timeouts equally for the rulebook walk and
        # only consult subType for the special challenge case, which is
        # structurally distinct from mandatory absorption.)
        is_timeout = pl.col("_at") == timeout_action

        to_rows = (
            df_pl.with_row_index("_row")
            .filter(is_timeout)
            .select("_row", "gameId", "period", "season", "seconds_remaining", "_st", order_col)
            .sort("gameId", order_col)
        )

        # Group by (gameId, period) and classify
        out_rows: list[tuple[int, str]] = []
        for (game_id, period), group in to_rows.group_by(["gameId", "period"], maintain_order=True):
            season_val = group["season"][0] if "season" in group.columns else None
            rules = _detect_era(int(season_val) if season_val is not None else None, pre_2017_mode=pre_2017_mode)
            if period not in rules["periods"]:
                for row_id in group["_row"].to_list():
                    out_rows.append((row_id, "discretionary"))
                continue
            tos = [
                {"sr": sr, "subType": st}
                for sr, st in zip(group["seconds_remaining"].to_list(), group["_st"].to_list())
            ]
            labels = _classify_one_period(tos, rules, challenge_subs)
            for row_id, label in zip(group["_row"].to_list(), labels):
                out_rows.append((row_id, label))

        # Build label series, default empty string for non-timeouts
        label_df = pl.DataFrame(out_rows, schema={"_row": pl.UInt32, "timeout_role": pl.String}, orient="row")
        result = (
            df_pl.with_row_index("_row")
            .join(label_df, on="_row", how="left")
            .with_columns(pl.col("timeout_role").fill_null(""))
            .drop("_row", "_at", "_st")
        )
        return result

    # ---------- validation against v3 ground truth ----------

    @staticmethod
    def _prep_v3(memo: NBAMemoDF, seasons: tuple[int, int]) -> pl.DataFrame:
        v3 = memo.data
        sub = v3[(v3["season"] >= seasons[0]) & (v3["season"] <= seasons[1])][
            [
                "gameId",
                "actionNumber",
                "period",
                "actionType",
                "subType",
                "seconds_remaining",
                "season",
                "season_type",
            ]
        ].copy()
        for col in ("actionType", "subType"):
            sub[col] = sub[col].astype("string").str.strip()
        return pl.from_pandas(sub)

    @staticmethod
    def _greedy_match(gt: list[int], pred: list[int], tol: int) -> tuple[int, int, int]:
        remaining = list(pred)
        tp = 0
        for g in gt:
            best_i, best_d = -1, tol + 1
            for i, p in enumerate(remaining):
                d = abs(p - g)
                if d < best_d:
                    best_d, best_i = d, i
            if best_i >= 0:
                tp += 1
                remaining.pop(best_i)
        return tp, len(remaining), len(gt) - tp

    @staticmethod
    def validate_against_v3(
        memo: NBAMemoDF,
        seasons: tuple[int, int] = (1998, 2016),
        tolerance_s: int = 60,
        label: str = "v3 reclassification",
        pre_2017_mode: Literal["independent", "cascading"] = "independent",
    ) -> ValidationResult:
        """Run ``classify_timeouts`` on v3 PBP and score predicted mandatories
        (``slot_K_mandatory``) against v3's ground-truth ``Official`` /
        ``Official TV`` subType rows, using greedy clock matching.
        """
        v3_pl = TVTimeoutValidation._prep_v3(memo, seasons)
        classified = TVTimeoutValidation.classify_timeouts(v3_pl, source="v3", pre_2017_mode=pre_2017_mode)

        pred = classified.filter(pl.col("timeout_role").str.contains("_mandatory")).select(
            "gameId",
            "period",
            "season",
            pl.col("seconds_remaining").round().cast(pl.Int64).alias("sr"),
        )
        gt = classified.filter(
            (pl.col("actionType").cast(pl.String).str.strip_chars() == "Timeout")
            & pl.col("subType").cast(pl.String).str.strip_chars().is_in(["Official", "Official TV"])
        ).select(
            "gameId",
            "period",
            "season",
            pl.col("seconds_remaining").round().cast(pl.Int64).alias("sr"),
        )

        pred_g = pred.group_by(["gameId", "period", "season"], maintain_order=True).agg(pl.col("sr"))
        gt_g = gt.group_by(["gameId", "period", "season"], maintain_order=True).agg(pl.col("sr"))
        buckets = pred_g.join(gt_g, on=["gameId", "period", "season"], how="full", coalesce=True)

        tp = fp = fn = 0
        by_season: dict[int, list[int]] = {}
        by_period: dict[int, list[int]] = {}
        for row in buckets.iter_rows(named=True):
            g_list = [int(x) for x in (row.get("sr_right") or [])]
            p_list = [int(x) for x in (row.get("sr") or [])]
            a, b, c = TVTimeoutValidation._greedy_match(g_list, p_list, tolerance_s)
            tp += a
            fp += b
            fn += c
            for store, key in ((by_season, row["season"]), (by_period, row["period"])):
                store.setdefault(key, [0, 0, 0])
                store[key][0] += a
                store[key][1] += b
                store[key][2] += c

        def _scores(t, f, n):
            p = t / max(t + f, 1)
            r = t / max(t + n, 1)
            return p, r, 2 * p * r / max(p + r, 1e-9)

        per_season_df = pd.DataFrame(
            [
                {
                    "season": s,
                    "TP": v[0],
                    "FP": v[1],
                    "FN": v[2],
                    **dict(zip(["precision", "recall", "f1"], _scores(*v))),
                }
                for s, v in sorted(by_season.items())
            ]
        )
        per_period_df = pd.DataFrame(
            [
                {
                    "period": p,
                    "TP": v[0],
                    "FP": v[1],
                    "FN": v[2],
                    **dict(zip(["precision", "recall", "f1"], _scores(*v))),
                }
                for p, v in sorted(by_period.items())
            ]
        )

        return ValidationResult(
            label=label,
            seasons=seasons,
            tolerance_s=tolerance_s,
            n_gt=tp + fn,
            n_pred=tp + fp,
            tp=tp,
            fp=fp,
            fn=fn,
            per_season=per_season_df,
            per_period=per_period_df,
        )

    @staticmethod
    def per_period_existence_score(
        memo: NBAMemoDF,
        seasons: tuple[int, int] = (1998, 2016),
        label: str = "",
        pre_2017_mode: Literal["independent", "cascading"] = "independent",
    ) -> dict:
        """Looser metric: did we predict ≥1 mandatory in this (gameId, period)
        iff v3 has ≥1 Official label?"""
        v3_pl = TVTimeoutValidation._prep_v3(memo, seasons)
        classified = TVTimeoutValidation.classify_timeouts(v3_pl, source="v3", pre_2017_mode=pre_2017_mode)
        pred_keys = (
            classified.filter(pl.col("timeout_role").str.contains("_mandatory")).select("gameId", "period").unique()
        )
        gt_keys = (
            classified.filter(
                (pl.col("actionType").cast(pl.String).str.strip_chars() == "Timeout")
                & pl.col("subType").cast(pl.String).str.strip_chars().is_in(["Official", "Official TV"])
            )
            .select("gameId", "period")
            .unique()
        )
        tp = pred_keys.join(gt_keys, on=["gameId", "period"], how="inner").height
        fp = pred_keys.height - tp
        fn = gt_keys.height - tp
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        return {
            "label": label or "v3 per-period existence",
            "seasons": seasons,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
        }

    # ---------- per-class confusion matrix on v3 ----------

    @staticmethod
    def confusion_matrix_v3(
        memo: NBAMemoDF,
        seasons: tuple[int, int] = (1998, 2016),
        pre_2017_mode: Literal["independent", "cascading"] = "independent",
    ) -> pd.DataFrame:
        """Cross-tab predicted ``timeout_role`` vs v3 ground-truth ``subType``
        for every timeout row in the labeled era. Shows where the classifier
        agrees with / diverges from v3's explicit labels.
        """
        v3_pl = TVTimeoutValidation._prep_v3(memo, seasons)
        classified = TVTimeoutValidation.classify_timeouts(v3_pl, source="v3", pre_2017_mode=pre_2017_mode)
        tos = (
            classified.filter(pl.col("actionType").cast(pl.String).str.strip_chars() == "Timeout")
            .select(
                pl.col("subType").cast(pl.String).str.strip_chars().alias("gt_subType"),
                pl.col("timeout_role").alias("predicted_role"),
            )
            .to_pandas()
        )
        return pd.crosstab(tos["gt_subType"], tos["predicted_role"], margins=True, margins_name="TOTAL")


# Module-level conveniences (preserve previous import patterns)
classify_timeouts = TVTimeoutValidation.classify_timeouts
validate_against_v3 = TVTimeoutValidation.validate_against_v3
per_period_existence_score = TVTimeoutValidation.per_period_existence_score
confusion_matrix_v3 = TVTimeoutValidation.confusion_matrix_v3
