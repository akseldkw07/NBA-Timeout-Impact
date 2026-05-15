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
- ``TVTimeoutValidation.confusion_matrix_v3(...)``: per-class confusion
  table.
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


# Trigger marks in seconds-remaining. Pre-2017: Q2/Q4 only, three triggers
# at 8:59 / 5:59 / 2:59. Post-2017: Q1-Q4, two triggers at 6:59 / 2:59.
PRE_2017_TRIGGERS = [540, 360, 180]
PRE_2017_PERIODS = [2, 4]
POST_2017_TRIGGERS = [420, 180]
POST_2017_PERIODS = [1, 2, 3, 4]


# -------------------- public result type --------------------------------- #


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

    # ---------- source dispatch ----------

    @staticmethod
    def get_source_config(source: Literal["v3", "cdnnba"]) -> dict:
        if source not in SOURCE_CONFIGS:
            raise ValueError(f"unknown source {source!r}, expected one of {list(SOURCE_CONFIGS)}")
        return SOURCE_CONFIGS[source]

    # ---------- classification ----------

    @staticmethod
    def classify_timeouts(
        df: pl.DataFrame | pd.DataFrame,
        source: Literal["v3", "cdnnba"],
        pre_2017_mode: Literal["independent", "cascading"] = "independent",
        seasons: tuple[int, int] | None = None,
        mandatory_tolerance_s: int = 60,
        mandatory_above_tolerance_s: int = 0,
    ) -> pl.DataFrame:
        """Vectorized labeler — adds a ``timeout_role`` column to every row.

        For each TIMEOUT row, sets one of:
        - ``slot_K_mandatory``: official auto-fired the slot K mandatory
          (sr ≤ trigger_K, within ``mandatory_tolerance_s`` below it).
        - ``slot_K_absorbed``: coach TO in slot K's pre-trigger absorb
          window (sr > trigger_K, sr ≤ upper_K).
        - ``challenge``: coach challenge subtype.
        - ``discretionary``: coach TO with no mandatory role.
        Non-TIMEOUT rows get ``""``.

        Era is auto-dispatched per row by ``season``:
        - ``season < 2017``: pre-2017 — Q2/Q4 only, triggers
          ``[540, 360, 180]`` (8:59 / 5:59 / 2:59). ``pre_2017_mode=
          "cascading"`` blocks later slots once any earlier slot fires.
        - ``season >= 2017``: post-2017 — Q1-Q4, triggers ``[420, 180]``
          (6:59 / 2:59). Always independent.

        Within each (gameId, period), the *first* coach TO in slot K's
        claim window (sr ∈ (trigger_K - mand_tol, upper_K]) claims the
        slot — later TOs in the same region stay discretionary. Slots
        K=1..N are resolved in order, so the slot-K-vs-slot-K+1 overlap
        zone (size ``mand_tol``) gives priority to the earlier trigger.
        """
        cfg = TVTimeoutValidation.get_source_config(source)
        timeout_action = cfg["timeout_action"]
        challenge_subs = set(cfg["challenge_subtypes"])
        order_col = cfg["order_col"]

        df_pd = df.to_pandas() if isinstance(df, pl.DataFrame) else df.copy()
        if seasons is not None and "season" in df_pd.columns:
            lo, hi = seasons
            df_pd = df_pd[df_pd["season"].between(lo, hi)].copy()

        # Sort by event order within each game so cumsum-based "first per
        # (gameId, period)" semantics line up with the play-by-play.
        df_pd = df_pd.sort_values(["gameId", order_col]).reset_index(drop=True)
        df_pd["actionType"] = df_pd["actionType"].astype(str).str.strip()
        df_pd["subType"] = df_pd["subType"].astype(str).str.strip()

        df_pd["timeout_role"] = ""
        is_timeout = df_pd["actionType"] == timeout_action
        is_challenge = is_timeout & df_pd["subType"].isin(challenge_subs)
        is_coach_to = is_timeout & ~is_challenge
        df_pd.loc[is_challenge, "timeout_role"] = "challenge"
        df_pd.loc[is_coach_to, "timeout_role"] = "discretionary"

        if "season" in df_pd.columns:
            pre_mask = (df_pd["season"] < 2017).fillna(False).astype(bool)
        else:
            pre_mask = pd.Series(False, index=df_pd.index)

        TVTimeoutValidation._apply_slot_labels(
            df_pd,
            eligible=pre_mask & is_coach_to,
            triggers=PRE_2017_TRIGGERS,
            periods_ok=PRE_2017_PERIODS,
            mand_tol=mandatory_tolerance_s,
            mand_tol_above=mandatory_above_tolerance_s,
            cascading=(pre_2017_mode == "cascading"),
        )
        TVTimeoutValidation._apply_slot_labels(
            df_pd,
            eligible=(~pre_mask) & is_coach_to,
            triggers=POST_2017_TRIGGERS,
            periods_ok=POST_2017_PERIODS,
            mand_tol=mandatory_tolerance_s,
            mand_tol_above=mandatory_above_tolerance_s,
            cascading=False,
        )

        return pl.from_pandas(df_pd)

    @staticmethod
    def _apply_slot_labels(
        df_pd: pd.DataFrame,
        *,
        eligible: pd.Series,
        triggers: list[int],
        periods_ok: list[int],
        mand_tol: int,
        mand_tol_above: int = 0,
        cascading: bool,
    ) -> None:
        """In-place: claim slots K=1..N sequentially.

        For each K, the first eligible coach TO per (gameId, period) whose
        ``seconds_remaining`` lands in ``(trigger_K - mand_tol, upper_K]``
        (where ``upper_K`` = previous trigger, or 720 for slot 1) claims
        the slot. Label is ``slot_K_mandatory`` if sr ∈ [trigger_K -
        mand_tol, trigger_K + mand_tol_above], else ``slot_K_absorbed``.

        Slot windows overlap by ``mand_tol`` between consecutive slots; the
        sequential iteration + ``claimed`` mask gives slot K priority in
        the overlap zone (the rulebook fires the earlier trigger first).
        Cascading mode additionally blocks all later slots in any
        (gameId, period) where slot K fired (mandatory).
        """
        sr = df_pd["seconds_remaining"]
        in_period = df_pd["period"].isin(periods_ok)
        claimed = pd.Series(False, index=df_pd.index)
        blocked = pd.Series(False, index=df_pd.index)
        group_keys = [df_pd["gameId"], df_pd["period"]]

        for K, trigger in enumerate(triggers, start=1):
            upper = triggers[K - 2] if K >= 2 else 720
            lower = trigger - mand_tol
            mand_upper = trigger + mand_tol_above

            slot_eligible = eligible & in_period & (sr > lower) & (sr <= upper) & ~claimed & ~blocked
            cum = slot_eligible.astype(int).groupby(group_keys).cumsum()
            is_first = (cum == 1) & slot_eligible

            is_mand = is_first & (sr <= mand_upper)
            is_absorb = is_first & (sr > mand_upper)

            df_pd.loc[is_absorb, "timeout_role"] = f"slot_{K}_absorbed"
            df_pd.loc[is_mand, "timeout_role"] = f"slot_{K}_mandatory"

            claimed = claimed | is_first

            if cascading:
                fired_per_group = is_mand.astype(int).groupby(group_keys).transform("max").astype(bool)
                blocked = blocked | fired_per_group

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
        mandatory_above_tolerance_s: int = 0,
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

        ``mandatory_tolerance_s`` is forwarded to the classifier (controls
        how far below a slot's trigger a TO is still tagged as mandatory
        firing).
        """
        v3_pl = TVTimeoutValidation._prep_v3(memo, seasons)
        classified = TVTimeoutValidation.classify_timeouts(
            v3_pl,
            source="v3",
            pre_2017_mode=pre_2017_mode,
            mandatory_tolerance_s=mandatory_tolerance_s,
            mandatory_above_tolerance_s=mandatory_above_tolerance_s,
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
        classified = TVTimeoutValidation.classify_timeouts(
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
        classified = TVTimeoutValidation.classify_timeouts(
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
classify_timeouts = TVTimeoutValidation.classify_timeouts
validate_against_v3 = TVTimeoutValidation.validate_against_v3
per_period_existence_score = TVTimeoutValidation.per_period_existence_score
confusion_matrix_v3 = TVTimeoutValidation.confusion_matrix_v3
