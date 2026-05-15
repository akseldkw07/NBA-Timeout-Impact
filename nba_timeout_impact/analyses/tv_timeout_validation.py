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

import typing as t
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
    mandatory_tolerance_s: int = 60,
) -> list[str]:
    """Row-by-row stateful classification of timeouts in one (gameId, period).

    Walks ``tos`` in event order. Each TO claims **at most one** slot based on
    its ``seconds_remaining`` (sr) position relative to the rulebook windows;
    once a slot is claimed (absorbed / fired / missed / blocked) it never
    accepts another TO. Decisions on prior rows carry forward via
    ``slot_state``.

    Slot windows:
        - **Absorb window** for slot K: ``sr in (absorb_low, absorb_high]``
        - **Mandatory-firing window** for slot K:
          ``sr in [absorb_low - mandatory_tolerance_s, absorb_low]`` — i.e. a
          TO at or just after the trigger fires the slot's mandatory.
        - If sr is far below ``absorb_low`` (more than the tolerance), slot K
          is marked **missed** (the rulebook fired but we didn't see a row).

    ``mandatory_tolerance_s``: how far below a trigger a TO can sit and still
    be considered the mandatory firing. Defaults to 60 s — Officials in v3
    usually sit within a few seconds of the trigger, but real stoppages can
    drift on long plays.
    """
    n = len(tos)
    labels = ["discretionary"] * n
    slots = rules["slots"]
    cascading = rules["cascading"]
    n_slots = len(slots)

    # slot_state[K]: None=open; "absorbed", "fired", "missed", "blocked" once claimed
    slot_state: list[str | None] = [None] * n_slots

    for i, to in enumerate(tos):
        if to["subType"] in challenge_subtypes:
            labels[i] = "challenge"
            continue
        sr = to["sr"]
        if sr is None:
            continue  # leave as discretionary

        # Walk slots in order, advancing past any already claimed.
        # For the first open slot whose window covers sr (absorb or mandatory
        # firing), claim it. If sr is below that slot's mandatory tolerance,
        # the slot was missed — mark it and try the next slot.
        K = 0
        while K < n_slots:
            if slot_state[K] is not None:
                K += 1
                continue
            absorb_low, absorb_high = slots[K]
            if sr > absorb_high:
                break  # TO occurs before this slot's window (unexpected in event order)
            if absorb_low < sr <= absorb_high:
                slot_state[K] = "absorbed"
                labels[i] = f"slot_{K + 1}_absorbed"
                break
            if absorb_low - mandatory_tolerance_s <= sr <= absorb_low:
                slot_state[K] = "fired"
                labels[i] = f"slot_{K + 1}_mandatory"
                if cascading:
                    for L in range(K + 1, n_slots):
                        if slot_state[L] is None:
                            slot_state[L] = "blocked"
                break
            # sr is far below this slot's trigger → slot was missed; advance
            slot_state[K] = "missed"
            K += 1

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
        source: t.Literal["v3", "cdnnba"],
        seasons: tuple[int, int] | None = None,
    ):
        cfg = TVTimeoutValidation.get_source_config(source)
        timeout_action = cfg["timeout_action"]
        challenge_subs = set(cfg["challenge_subtypes"])
        order_col = cfg["order_col"]

        df_pd = df.to_pandas() if isinstance(df, pl.DataFrame) else df

        if seasons is not None and "season" in df_pd.columns:
            lo, hi = seasons
            # The fix for your specific code
            df_pd = df_pd[df_pd["season"].between(lo, hi)]

        df_pd["f_isTimeout"] = df_pd["actionType"].str.strip() == timeout_action
        df_pd["f_timeLT_6_59"] = df_pd["seconds_remaining"] < 6 * 60 + 59
        df_pd["f_timeLT_2_59"] = df_pd["seconds_remaining"] < 2 * 60 + 59

        df_pd["actionType"] = df_pd["actionType"].str.strip()
        df_pd["subType"] = df_pd["subType"].str.strip()

        # Create a list of tuples, then convert to categorical
        df_pd["GamePeriodCat"] = pd.Categorical(list(zip(df_pd["gameId"], df_pd["period"])), ordered=True)
        df_pd["cumTimeoutsInPeriod"] = df_pd.groupby("GamePeriodCat", observed=True)["f_isTimeout"].cumsum()
        # Check if it's a timeout, if it's the first one (cum == 1), and if it hits the time flag
        # 1. Define the specific conditions
        cond_6_59 = df_pd["f_isTimeout"] & df_pd["f_timeLT_6_59"]
        cond_2_59 = df_pd["f_isTimeout"] & df_pd["f_timeLT_2_59"]

        # 2. Use groupby + cumsum to flag only the first occurrence per group
        # We check where cumsum == 1 AND the condition itself is True
        df_pd["f_firstTimeoutLT_6_59"] = (
            cond_6_59.groupby(df_pd["GamePeriodCat"], observed=True).cumsum() == 1
        ) & cond_6_59
        df_pd["f_firstTimeoutLT_2_59"] = (
            cond_2_59.groupby(df_pd["GamePeriodCat"], observed=True).cumsum() == 1
        ) & cond_2_59

        df_pd["MandatoryTimeout"] = (df_pd["f_firstTimeoutLT_6_59"] & df_pd["cumTimeoutsInPeriod"] == 1) | (
            df_pd["f_firstTimeoutLT_2_59"] & df_pd["cumTimeoutsInPeriod"] == 2
        )

        return df_pd, timeout_action, challenge_subs, order_col

    @staticmethod
    def classify_timeouts2(
        df: pl.DataFrame | pd.DataFrame,
        source: Literal["v3", "cdnnba"],
        pre_2017_mode: Literal["independent", "cascading"] = "independent",
        seasons: tuple[int, int] | None = None,
        mandatory_tolerance_s: int = 60,
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

        df_pl = pl.from_pandas(df) if isinstance(df, pd.DataFrame) else df

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
            labels = _classify_one_period(tos, rules, challenge_subs, mandatory_tolerance_s=mandatory_tolerance_s)
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
    def _score_row_by_row(
        classified: pl.DataFrame,
        seasons: tuple[int, int],
        tolerance_s: int,
        label: str,
    ) -> ValidationResult:
        """Row-by-row scoring on the classified timeout-row population.

        TP: row is predicted mandatory AND v3 ``subType`` ∈ Official/Official TV.
        FP: row is predicted mandatory BUT v3 ``subType`` is not Official.
        FN: row is v3 Official BUT predicted ``timeout_role`` is not mandatory.
        """
        tos = (
            classified.filter(pl.col("actionType").cast(pl.String).str.strip_chars() == "Timeout")
            .with_columns(
                pl.col("subType").cast(pl.String).str.strip_chars().alias("_gt"),
                pl.col("timeout_role").str.contains("_mandatory").alias("_pred_mand"),
            )
            .with_columns(
                pl.col("_gt").is_in(["Official", "Official TV"]).alias("_is_gt"),
            )
        )

        def _counts(df: pl.DataFrame) -> tuple[int, int, int]:
            tp = df.filter(pl.col("_is_gt") & pl.col("_pred_mand")).height
            fp = df.filter(~pl.col("_is_gt") & pl.col("_pred_mand")).height
            fn = df.filter(pl.col("_is_gt") & ~pl.col("_pred_mand")).height
            return tp, fp, fn

        tp, fp, fn = _counts(tos)

        def _scores(t: int, f: int, n: int) -> tuple[float, float, float]:
            p = t / max(t + f, 1)
            r = t / max(t + n, 1)
            return p, r, 2 * p * r / max(p + r, 1e-9)

        per_season_rows = []
        for s in sorted(tos["season"].unique().to_list()):
            t, f, n = _counts(tos.filter(pl.col("season") == s))
            p, r, f1 = _scores(t, f, n)
            per_season_rows.append({"season": s, "TP": t, "FP": f, "FN": n, "precision": p, "recall": r, "f1": f1})

        per_period_rows = []
        for pe in sorted(tos["period"].unique().to_list()):
            t, f, n = _counts(tos.filter(pl.col("period") == pe))
            p, r, f1 = _scores(t, f, n)
            per_period_rows.append({"period": pe, "TP": t, "FP": f, "FN": n, "precision": p, "recall": r, "f1": f1})

        return ValidationResult(
            label=label,
            seasons=seasons,
            tolerance_s=tolerance_s,
            n_gt=tp + fn,
            n_pred=tp + fp,
            tp=tp,
            fp=fp,
            fn=fn,
            per_season=pd.DataFrame(per_season_rows),
            per_period=pd.DataFrame(per_period_rows),
        )

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
        tolerance_s: int = 0,
        label: str = "v3 reclassification",
        pre_2017_mode: Literal["independent", "cascading"] = "independent",
        match_mode: Literal["row", "fuzzy"] = "row",
        mandatory_tolerance_s: int = 60,
    ) -> ValidationResult:
        """Score ``classify_timeouts`` predictions against v3 ground-truth
        ``Official`` / ``Official TV`` subType labels.

        ``match_mode``:
            - ``"row"`` (default): row-by-row exact match. Each timeout row is
              scored on whether the predicted ``timeout_role`` is mandatory and
              whether v3's ``subType`` flags it Official. ``tolerance_s`` is
              ignored.
            - ``"fuzzy"``: legacy greedy clock matching within ``tolerance_s``
              seconds. Useful as a loosened-grading fallback.

        ``mandatory_tolerance_s`` is forwarded to the classifier (controls how
        far below a slot's trigger a TO is still tagged as mandatory firing).
        """
        v3_pl = TVTimeoutValidation._prep_v3(memo, seasons)
        classified = TVTimeoutValidation.classify_timeouts2(
            v3_pl,
            source="v3",
            pre_2017_mode=pre_2017_mode,
            mandatory_tolerance_s=mandatory_tolerance_s,
        )

        if match_mode == "row":
            return TVTimeoutValidation._score_row_by_row(
                classified, seasons=seasons, tolerance_s=tolerance_s, label=label
            )

        pred = classified.filter(pl.col("timeout_role").str.contains("_mandatory")).select(
            "gameId", "period", "season", pl.col("seconds_remaining").round().cast(pl.Int64).alias("sr")
        )
        gt = classified.filter(
            (pl.col("actionType").cast(pl.String).str.strip_chars() == "Timeout")
            & pl.col("subType").cast(pl.String).str.strip_chars().is_in(["Official", "Official TV"])
        ).select("gameId", "period", "season", pl.col("seconds_remaining").round().cast(pl.Int64).alias("sr"))

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
        mandatory_tolerance_s: int = 60,
    ) -> dict:
        """Looser metric: did we predict ≥1 mandatory in this (gameId, period)
        iff v3 has ≥1 Official label?"""
        v3_pl = TVTimeoutValidation._prep_v3(memo, seasons)
        classified = TVTimeoutValidation.classify_timeouts2(
            v3_pl, source="v3", pre_2017_mode=pre_2017_mode, mandatory_tolerance_s=mandatory_tolerance_s
        )
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
        mandatory_tolerance_s: int = 60,
    ) -> pd.DataFrame:
        """Cross-tab predicted ``timeout_role`` vs v3 ground-truth ``subType``
        for every timeout row in the labeled era. Shows where the classifier
        agrees with / diverges from v3's explicit labels.
        """
        v3_pl = TVTimeoutValidation._prep_v3(memo, seasons)
        classified = TVTimeoutValidation.classify_timeouts2(
            v3_pl, source="v3", pre_2017_mode=pre_2017_mode, mandatory_tolerance_s=mandatory_tolerance_s
        )
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
classify_timeouts = TVTimeoutValidation.classify_timeouts2
validate_against_v3 = TVTimeoutValidation.validate_against_v3
per_period_existence_score = TVTimeoutValidation.per_period_existence_score
confusion_matrix_v3 = TVTimeoutValidation.confusion_matrix_v3
