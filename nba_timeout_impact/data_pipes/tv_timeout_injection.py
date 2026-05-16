"""TV / mandatory timeout reclassification + validation.

Each play-by-play feed carries its own *structural* signal for league-fired
mandatories — we don't have to infer them from sr position. Both signals
were discovered by inspecting raw rows (see notebook
``tv-validation-injection-auto-claude.ipynb`` for the derivation):

- **Pre-2017 v3 (nbastatsv3)**: every ``Official`` / ``Official TV`` row
  has ``personId == 0`` (league-charged, no team). Every Regular/Short
  has a real team-shaped ``personId``. → ``personId == 0`` IS the
  mandatory flag. F1 ≈ 0.93 against v3's ``subType`` ground truth.

- **Post-2017 cdnnba**: the ``qualifiers`` column directly tags
  mandatories as ``"team, mandatory"`` (vs ``"team"`` for purely
  discretionary coach TOs).

``classify_timeouts(df, source)`` returns the input frame with a
``timeout_role`` column:
    ``slot_K_mandatory`` — league-charged mandatory at slot K
    ``challenge``        — coach's challenge timeout
    ``discretionary``    — coach TO that didn't satisfy a mandatory slot
    ``""``               — non-timeout row

Slot K is assigned by ``seconds_remaining`` relative to era-appropriate
trigger marks:
    Pre-2017 Q2/Q4: 3 triggers at 8:59 / 5:59 / 2:59 (sr 540 / 360 / 180)
    Pre-2017 Q1/Q3: 2 triggers at 5:59 / 2:59
    Post-2017 Q1-Q4: 2 triggers at 6:59 / 2:59 (sr 420 / 180)
    OT (any era):    1 trigger at 2:59
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd
import polars as pl

from nba_timeout_impact.datasets.memo_nbastatsv3 import NBAMemoDF

Source = Literal["v3", "cdnnba"]


# --------------------------------------------------------------------------- #
#  Source configs                                                             #
# --------------------------------------------------------------------------- #

SOURCE_CONFIGS: dict[str, dict] = {
    "v3": {
        "timeout_action": "Timeout",
        "challenge_subtypes": {"Coach Challenge"},
        "order_col": "actionNumber",
        # personId == 0 means the league charged the TO to no team (i.e.,
        # an Official TV mandatory). Discovered by inspecting raw rows.
        "mandatory_signal": "personId_zero",
    },
    "cdnnba": {
        "timeout_action": "timeout",
        "challenge_subtypes": {"challenge"},
        "order_col": "orderNumber",
        # qualifiers field tags mandatories explicitly as "team, mandatory".
        "mandatory_signal": "qualifier_mandatory",
    },
}


# --------------------------------------------------------------------------- #
#  Rulebook trigger marks: (period_filter, triggers_descending)               #
# --------------------------------------------------------------------------- #

PRE_2017_SLOTS = [
    ([2, 4], [540, 360, 180]),  # 8:59, 5:59, 2:59
    ([1, 3], [360, 180]),  # 5:59, 2:59
    ([5, 6, 7, 8, 9, 10], [180]),  # OT: 2:59
]
POST_2017_SLOTS = [
    ([1, 2, 3, 4], [420, 180]),  # 6:59, 2:59
    ([5, 6, 7, 8, 9, 10], [180]),  # OT: 2:59
]


# --------------------------------------------------------------------------- #
#  Cause classification — rulebook-faithful approach                          #
# --------------------------------------------------------------------------- #

# We classify a mandatory-qualified TO as ``tv_mandatory`` vs ``coach_absorb``
# entirely from rulebook structure:
#
#   slot K = the K-th mandatory-qualified TO in the (gameId, period)
#   trigger_K = the K-th rulebook trigger in descending sr (e.g., post-2017
#               Q1-Q4 has triggers [420, 180] = 6:59, 2:59)
#
#   - If this TO's ``seconds_remaining`` ≤ trigger_K, the league trigger
#     had already expired (or expired right at the TO) — this is the
#     "first dead ball after the trigger" event → ``tv_mandatory``.
#   - If sr > trigger_K, the coach called the TO BEFORE the league
#     trigger fired, absorbing the slot pre-emptively → ``coach_absorb``.
#
# This makes only one assumption: that mandatory-qualified TOs fill slots
# 1..N in event order. The cdnnba feed enforces this — slot 1 is always
# the first mandatory-qualified row in the period, slot 2 the second.
# No duration filter, no sr buffer, no preempt sub-bucket.

# Sanity bounds for ``timeout_duration_s``. cdnnba's ``timeActual`` has rare
# data glitches (rows where the wall-clock jumps backwards by a full day or
# similar). Anything outside this range is set to null in the persisted
# ``timeout_duration_s`` column.
DURATION_SANITY_MIN_S = 0
DURATION_SANITY_MAX_S = 600  # 10 minutes — well above any real TO


# --------------------------------------------------------------------------- #
#  Result type                                                                #
# --------------------------------------------------------------------------- #


@dataclass
class ValidationResult:
    label: str
    seasons: tuple[int, int] | None
    n_gt: int
    n_pred: int
    tp: int
    fp: int
    fn: int
    per_season: pd.DataFrame = field(repr=False)
    per_period: pd.DataFrame = field(repr=False)
    # Kept for back-compat with the original plotting code's "row-by-row vs fuzzy"
    # title heuristic. The new classifier never does fuzzy clock matching, so
    # always 0.
    tolerance_s: int = 0

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
            f"{self.label}: n_gt={self.n_gt:,} n_pred={self.n_pred:,} "
            f"TP={self.tp:,} FP={self.fp:,} FN={self.fn:,} | "
            f"P={self.precision:.3f} R={self.recall:.3f} F1={self.f1:.3f}"
        )

    def __repr__(self) -> str:
        return f"<ValidationResult {self.summary()}>"


# --------------------------------------------------------------------------- #
#  Main classifier                                                            #
# --------------------------------------------------------------------------- #


class TVTimeoutValidation:
    """Static-method container for mandatory-timeout reclassification + validation."""

    @staticmethod
    def get_source_config(source: Source) -> dict:
        if source not in SOURCE_CONFIGS:
            raise ValueError(f"unknown source {source!r}, expected one of {list(SOURCE_CONFIGS)}")
        return SOURCE_CONFIGS[source]

    # ---------- shared duration computer ----------

    @staticmethod
    def compute_timeout_duration_s(df: pl.DataFrame) -> pl.Series:
        """Wall-clock seconds from each ``timeout`` row to the next event that
        represents game resumption.

        "Resumption" excludes ``substitution`` / ``stoppage`` / ``instantreplay``
        rows — those are logged DURING the TV break itself. Returns null on
        non-timeout rows AND on rows where ``timeActual`` glitches produce
        durations outside ``[DURATION_SANITY_MIN_S, DURATION_SANITY_MAX_S]``.

        Called by ``CDNNBADatasetPL._inject_timeout_columns`` at load time;
        the result is persisted as the ``timeout_duration_s`` column.
        Caller's input must contain ``actionType``, ``timeActual``, ``gameId``.
        """
        excluded = ["substitution", "stoppage", "instantreplay"]
        next_resume = (
            pl.when(~pl.col("actionType").is_in(excluded))
            .then(pl.col("timeActual"))
            .otherwise(None)
            .shift(-1)
            .fill_null(strategy="backward")
            .over("gameId")
        )
        raw_delta = (next_resume - pl.col("timeActual")).dt.total_seconds()
        result = df.select(
            pl.when(
                (pl.col("actionType") == "timeout")
                & (raw_delta >= DURATION_SANITY_MIN_S)
                & (raw_delta <= DURATION_SANITY_MAX_S)
            )
            .then(raw_delta)
            .otherwise(None)
            .alias("timeout_duration_s")
        )
        return result["timeout_duration_s"]

    # ---------- sequence-position helpers ----------

    @staticmethod
    def compute_cum_timeouts_period(df: pl.DataFrame) -> pl.Series:
        """Cumulative count of timeout rows within each ``(gameId, period)``,
        evaluated AT each row (inclusive).

        For a timeout row, this is its 1-indexed rank among all TOs in the
        period. For a non-timeout row, this is the number of TOs that came
        before this row in the period.

        Caller's input must contain ``actionType``, ``gameId``, ``period``.
        """
        is_to = (pl.col("actionType") == "timeout").cast(pl.Int64)
        return df.select(
            is_to.cum_sum().over(["gameId", "period"]).alias("cumTimeoutsPeriod"),
        )["cumTimeoutsPeriod"]

    @staticmethod
    def _compute_cum_mandatory_period(df: pl.DataFrame, mandatory_signal: str) -> pl.Series:
        """Cumulative count of mandatory-qualified timeout rows within each
        ``(gameId, period)``.

        For a mandatory-qualified TO, this equals its slot K under the
        rulebook (1 = first mandatory in period = slot 1, 2 = second =
        slot 2, etc.).
        """
        if mandatory_signal == "personId_zero":
            is_mand = (pl.col("actionType").is_in(["timeout", "Timeout"])) & (pl.col("personId").fill_null(-1) == 0)
        elif mandatory_signal == "qualifier_mandatory":
            is_mand = (pl.col("actionType") == "timeout") & pl.col("qualifiers").cast(pl.String).str.contains(
                "mandatory"
            ).fill_null(False)
        else:
            raise ValueError(f"unknown mandatory_signal {mandatory_signal!r}")
        return df.select(
            is_mand.cast(pl.Int64).cum_sum().over(["gameId", "period"]).alias("_cum_mand"),
        )["_cum_mand"]

    # ---------- classification ----------

    @staticmethod
    def classify_timeouts(
        df: pl.DataFrame | pd.DataFrame, source: Source, seasons: tuple[int, int] | None = None
    ) -> pl.DataFrame:
        """Add a ``timeout_role`` column to every row.

        Source-specific mandatory signal:
            ``"v3"``     → ``personId == 0`` flags Official / Official TV.
            ``"cdnnba"`` → ``qualifiers`` contains ``"mandatory"``.

        Slot K is assigned by ``seconds_remaining`` and ``period`` against
        the era's rulebook trigger marks. A mandatory at sr=X belongs to
        the slot whose trigger most recently fired (i.e., the highest K
        whose trigger sr ≥ X). Mandatories above all triggers (pre-trigger
        absorbs) default to slot 1.

        ``seasons``: optional ``(lo, hi)`` inclusive filter on ``season``.
        """
        cfg = TVTimeoutValidation.get_source_config(source)
        df_pd = df.to_pandas() if isinstance(df, pl.DataFrame) else df.copy()
        if seasons is not None and "season" in df_pd.columns:
            lo, hi = seasons
            df_pd = df_pd[df_pd["season"].between(lo, hi)].copy()

        # Keep the input row order — every label decision below is row-local
        # (``personId == 0``, ``qualifiers`` contains ``"mandatory"``, sr
        # position). Resorting would break alignment with ``memo.cdnnba``
        # and any other memo series the caller wants to join row-wise.
        df_pd["actionType"] = df_pd["actionType"].astype(str).str.strip()
        df_pd["subType"] = df_pd["subType"].astype(str).str.strip()

        df_pd["timeout_role"] = ""
        is_timeout = df_pd["actionType"] == cfg["timeout_action"]
        is_challenge = is_timeout & df_pd["subType"].isin(cfg["challenge_subtypes"])
        is_full_to = is_timeout & ~is_challenge
        df_pd.loc[is_challenge, "timeout_role"] = "challenge"
        df_pd.loc[is_full_to, "timeout_role"] = "discretionary"

        is_mandatory = is_full_to & _detect_mandatory(df_pd, cfg["mandatory_signal"])

        sr = df_pd["seconds_remaining"]
        slot_table = PRE_2017_SLOTS if source == "v3" else POST_2017_SLOTS

        # Slot identity = rank of this mandatory among period mandatories.
        # i.e. ``slot_K_mandatory`` literally means "this is the K-th
        # mandatory-qualified TO in the (gameId, period)." This matches the
        # rulebook (slots fill in event order) and stays consistent with
        # ``timeout_cause`` below. NOT a position-by-sr assignment — a TO at
        # sr=300 that's the 2nd mandatory of the period is ``slot_2_mandatory``,
        # NOT slot_1, even though sr=300 sits in slot 1's sr range.
        cum_mand = TVTimeoutValidation._compute_cum_mandatory_period(
            pl.from_pandas(df_pd), cfg["mandatory_signal"]
        ).to_pandas()

        for periods_ok, triggers in slot_table:
            in_periods = df_pd["period"].isin(periods_ok)
            n_slots = len(triggers)
            for K in range(1, n_slots + 1):
                mask = is_mandatory & in_periods & (cum_mand == K)
                df_pd.loc[mask, "timeout_role"] = f"slot_{K}_mandatory"
            # Anomalous: more mandatory-qualified rows than slots in the period.
            # Tag them with the last slot label (rare; cause logic flags absorb).
            extras = is_mandatory & in_periods & (cum_mand > n_slots)
            df_pd.loc[extras, "timeout_role"] = f"slot_{n_slots}_mandatory"

        # ------------------------------------------------------------------ #
        #  timeout_cause — rulebook-faithful taxonomy                         #
        # ------------------------------------------------------------------ #
        # Categories:
        #   ""                    — non-timeout
        #   "challenge"           — coach challenge
        #   "coach_discretionary" — coach TO that didn't fill a mandatory slot
        #   "tv_mandatory"        — league-forced TV commercial break (the
        #                           K-th mandatory in the period at sr ≤
        #                           trigger_K — the trigger had already
        #                           expired so this row IS the first dead
        #                           ball / Official-TV firing).
        #   "coach_absorb"        — coach TO that absorbed the K-th mandatory
        #                           slot pre-emptively (sr > trigger_K).
        #
        # CAUTION: ``tv_mandatory`` rows carry ``teamTricode`` / ``teamId``
        # per the NBA's structural charge-to-home (slot 1) / charge-to-road
        # (slot 2) convention — the team label is a bookkeeping artifact for
        # true auto-fires, NOT the coach's decision.
        #
        # For causal analysis filter to ``timeout_cause == "tv_mandatory"``.
        # ``cum_mand`` (computed above for role assignment) is reused here.

        df_pd["timeout_cause"] = ""
        df_pd.loc[is_challenge, "timeout_cause"] = "challenge"
        df_pd.loc[is_full_to & ~is_mandatory, "timeout_cause"] = "coach_discretionary"

        # Rulebook classification, identical for both sources: slot K = the
        # K-th mandatory-qualified TO in the period; sr ≤ trigger_K means the
        # trigger had already fired (tv_mandatory), else the coach absorbed
        # the slot pre-emptively (coach_absorb).
        for periods_ok, triggers in slot_table:
            in_periods = df_pd["period"].isin(periods_ok)
            for K, trigger in enumerate(triggers, start=1):
                mask = is_full_to & is_mandatory & in_periods & (cum_mand == K)
                df_pd.loc[mask & (sr <= trigger), "timeout_cause"] = "tv_mandatory"
                df_pd.loc[mask & (sr > trigger), "timeout_cause"] = "coach_absorb"
            # Anomalous: more mandatory-qualified rows than slots in the period.
            # Default to coach_absorb (rare; treat the extras as coach calls).
            extras = is_full_to & is_mandatory & in_periods & (cum_mand > len(triggers))
            df_pd.loc[extras, "timeout_cause"] = "coach_absorb"

        # Also expose ``cumTimeoutsPeriod`` (cum of ALL TOs in period) on the
        # output — the loader persists it as a column for downstream use.
        df_pd["cumTimeoutsPeriod"] = (
            TVTimeoutValidation.compute_cum_timeouts_period(pl.from_pandas(df_pd)).to_pandas().values
        )

        return pl.from_pandas(df_pd)

    # ---------- v3-era validation ----------

    @staticmethod
    def _prep_v3(memo: NBAMemoDF, seasons: tuple[int, int] | None = None) -> pl.DataFrame:
        v3 = memo.data
        cols = [
            "gameId",
            "actionNumber",
            "period",
            "actionType",
            "subType",
            "seconds_remaining",
            "season",
            "season_type",
            "personId",
        ]
        cols = [c for c in cols if c in v3.columns]
        if seasons is None:
            sub = v3[cols].copy()
        else:
            sub = v3[(v3["season"] >= seasons[0]) & (v3["season"] <= seasons[1])][cols].copy()
        for col in ("actionType", "subType"):
            sub[col] = sub[col].astype("string").str.strip()
        return pl.from_pandas(sub)

    @staticmethod
    def validate_against_v3(
        memo: NBAMemoDF,
        seasons: tuple[int, int] | None = None,
        label: str = "v3 reclassification",
    ) -> ValidationResult:
        """Score ``classify_timeouts`` predictions against v3 ground-truth
        ``Official`` / ``Official TV`` subType labels.
        """
        v3_pl = TVTimeoutValidation._prep_v3(memo, seasons)
        classified = TVTimeoutValidation.classify_timeouts(v3_pl, source="v3", seasons=seasons)
        return _score_generic(classified, _v3_gt_mask(), seasons=seasons, label=label, action="Timeout")

    @staticmethod
    def confusion_matrix_v3(memo: NBAMemoDF, seasons: tuple[int, int] | None = None) -> pd.DataFrame:
        v3_pl = TVTimeoutValidation._prep_v3(memo, seasons)
        classified = TVTimeoutValidation.classify_timeouts(v3_pl, source="v3", seasons=seasons)
        tos = (
            classified.filter(pl.col("actionType").cast(pl.String).str.strip_chars() == "Timeout")
            .select(
                pl.col("subType").cast(pl.String).str.strip_chars().alias("gt_subType"),
                pl.col("timeout_role").alias("predicted_role"),
            )
            .to_pandas()
        )
        return pd.crosstab(tos["gt_subType"], tos["predicted_role"], margins=True, margins_name="TOTAL")

    # ---------- cdnnba-era validation ----------

    @staticmethod
    def _prep_cdnnba(memo, seasons: tuple[int, int] | None = None) -> pl.DataFrame:
        """Prep cdnnba feed for classification. Accepts a polars memo
        (``CDNNBAMemoPL`` subclasses ``pl.DataFrame``), a wrapper with
        ``.data``, a raw polars DataFrame, or a pandas DataFrame.
        Keeps ``qualifiers`` (mandatory signal) and ``timeActual``
        (wall-clock, for diagnostic plots). ``seasons=None`` skips the
        season filter entirely.
        """
        # MemoDataFramePL inherits from pl.DataFrame, so check pl FIRST —
        # the ``hasattr(memo, "data")`` path is only for pandas-style memos.
        if isinstance(memo, pl.DataFrame):
            df = memo
        elif hasattr(memo, "data"):
            df = memo.data
        else:
            df = memo

        # ``timeout_duration_s`` is persisted on the frame at load time
        # (see ``CDNNBADatasetPL._inject_timeout_columns``). If the caller
        # bypassed the loader and passed a raw parquet, the column won't
        # be present and the cause classifier falls back to position-only.

        wanted = [
            "gameId",
            "orderNumber",
            "period",
            "actionType",
            "subType",
            "seconds_remaining",
            "season",
            "season_type",
            "personId",
            "teamId",
            "teamTricode",
            "qualifiers",
            "timeActual",
            "timeout_duration_s",
        ]

        if isinstance(df, pl.DataFrame):
            cols = [c for c in wanted if c in df.columns]
            if seasons is None:
                return df.select(cols)
            return df.filter((pl.col("season") >= seasons[0]) & (pl.col("season") <= seasons[1])).select(cols)

        # pandas path
        cols = [c for c in wanted if c in df.columns]
        if seasons is None:
            sub = df[cols].copy()
        else:
            sub = df[(df["season"] >= seasons[0]) & (df["season"] <= seasons[1])][cols].copy()
        for col in ("actionType", "subType"):
            sub[col] = sub[col].astype("string").str.strip()
        return pl.from_pandas(sub)

    @staticmethod
    def validate_against_cdnnba(
        memo,
        seasons: tuple[int, int] | None = None,
        label: str = "cdnnba reclassification",
    ) -> ValidationResult:
        """Score ``classify_timeouts`` predictions against cdnnba's
        ``qualifiers`` mandatory tag. Should be ≈ 1.0 since we predict
        from the same signal. Provided for harness symmetry with the v3
        validator.
        """
        cdn_pl = TVTimeoutValidation._prep_cdnnba(memo, seasons)
        classified = TVTimeoutValidation.classify_timeouts(cdn_pl, source="cdnnba", seasons=seasons)
        return _score_generic(classified, _cdnnba_gt_mask(), seasons=seasons, label=label, action="timeout")

    # Back-compat shim for the older auto-claude notebook, which calls
    # ``TVTimeoutValidation._score_row_by_row(classified, seasons=..., tolerance_s=0, label=...)``.
    # ``tolerance_s`` is no longer meaningful (we don't do fuzzy clock matching anymore)
    # — accepted and ignored.
    @staticmethod
    def _score_row_by_row(
        classified: pl.DataFrame,
        seasons: tuple[int, int] | None = None,
        tolerance_s: int = 0,  # noqa: ARG004 — accepted for back-compat
        label: str = "v3 reclassification",
    ) -> ValidationResult:
        return _score_generic(classified, _v3_gt_mask(), seasons=seasons, label=label, action="Timeout")

    @staticmethod
    def confusion_matrix_cdnnba(memo, seasons: tuple[int, int] | None = None) -> pd.DataFrame:
        cdn_pl = TVTimeoutValidation._prep_cdnnba(memo, seasons)
        classified = TVTimeoutValidation.classify_timeouts(cdn_pl, source="cdnnba", seasons=seasons)
        tos = (
            classified.filter(pl.col("actionType").cast(pl.String).str.strip_chars() == "timeout")
            .with_columns(
                pl.col("qualifiers").cast(pl.String).str.contains("mandatory").alias("_is_mand_gt"),
            )
            .select(
                pl.when(pl.col("_is_mand_gt")).then(pl.lit("mandatory")).otherwise(pl.lit("not_mandatory")).alias("gt"),
                pl.col("timeout_role").alias("predicted_role"),
            )
            .to_pandas()
        )
        return pd.crosstab(tos["gt"], tos["predicted_role"], margins=True, margins_name="TOTAL")


# --------------------------------------------------------------------------- #
#  Internals                                                                  #
# --------------------------------------------------------------------------- #


def _detect_mandatory(df_pd: pd.DataFrame, signal: str) -> pd.Series:
    """Boolean Series marking rows that are league-fired mandatories."""
    if signal == "personId_zero":
        if "personId" not in df_pd.columns:
            raise ValueError("v3 classifier requires 'personId' column in input")
        return df_pd["personId"].fillna(-1) == 0
    if signal == "qualifier_mandatory":
        if "qualifiers" not in df_pd.columns:
            raise ValueError("cdnnba classifier requires 'qualifiers' column in input")
        q = df_pd["qualifiers"]
        # cdnnba ships qualifiers as either a comma-joined string
        # ("team, mandatory") or a list-of-strings (["team", "mandatory"])
        # depending on whether it came from the polars memo or the pandas
        # enriched parquet. Handle both.
        if q.dtype == "object" and len(q) and any(isinstance(v, (list, tuple)) for v in q.head(50)):
            return q.map(lambda v: bool(v) and "mandatory" in v)  # type: ignore
        return q.astype("string").str.contains("mandatory", na=False)
    raise ValueError(f"unknown mandatory_signal {signal!r}")


def _slot_by_position(sr: pd.Series, triggers: list[int]) -> pd.Series:
    """For each row, the slot K = highest K such that ``triggers[K-1] ≥ sr``.

    Rows above all triggers (pre-trigger absorbs) default to slot 1.
    Triggers must be in descending sr order.
    """
    slot = pd.Series(1, index=sr.index, dtype="int8")
    for K, t in enumerate(triggers, start=1):
        slot = slot.where(sr > t, K)  # if sr <= t, this slot's trigger has fired → upgrade to K
    return slot


def _v3_gt_mask() -> pl.Expr:
    return pl.col("subType").cast(pl.String).str.strip_chars().is_in(["Official", "Official TV"])


def _cdnnba_gt_mask() -> pl.Expr:
    return pl.col("qualifiers").cast(pl.String).str.contains("mandatory")


def _score_generic(
    classified: pl.DataFrame,
    gt_mask: pl.Expr,
    seasons: tuple[int, int] | None,
    label: str,
    action: str,
) -> ValidationResult:
    """Row-by-row TP/FP/FN against a polars ground-truth boolean expression."""
    tos = classified.filter(pl.col("actionType").cast(pl.String).str.strip_chars() == action).with_columns(
        gt_mask.alias("_is_gt"),
        pl.col("timeout_role").str.contains("_mandatory").alias("_pred_mand"),
    )

    def _counts(d: pl.DataFrame) -> tuple[int, int, int]:
        tp = d.filter(pl.col("_is_gt") & pl.col("_pred_mand")).height
        fp = d.filter(~pl.col("_is_gt") & pl.col("_pred_mand")).height
        fn = d.filter(pl.col("_is_gt") & ~pl.col("_pred_mand")).height
        return tp, fp, fn

    def _scores(t: int, f: int, n: int) -> tuple[float, float, float]:
        p = t / max(t + f, 1)
        r = t / max(t + n, 1)
        return p, r, 2 * p * r / max(p + r, 1e-9)

    tp, fp, fn = _counts(tos)

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
        n_gt=tp + fn,
        n_pred=tp + fp,
        tp=tp,
        fp=fp,
        fn=fn,
        per_season=pd.DataFrame(per_season_rows),
        per_period=pd.DataFrame(per_period_rows),
    )


# --------------------------------------------------------------------------- #
#  Module-level conveniences                                                  #
# --------------------------------------------------------------------------- #

classify_timeouts = TVTimeoutValidation.classify_timeouts
validate_against_v3 = TVTimeoutValidation.validate_against_v3
validate_against_cdnnba = TVTimeoutValidation.validate_against_cdnnba
confusion_matrix_v3 = TVTimeoutValidation.confusion_matrix_v3
confusion_matrix_cdnnba = TVTimeoutValidation.confusion_matrix_cdnnba
