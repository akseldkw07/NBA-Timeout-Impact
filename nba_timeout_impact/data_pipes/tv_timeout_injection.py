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

        df_pd = df_pd.sort_values(["gameId", cfg["order_col"]]).reset_index(drop=True)
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
        for periods_ok, triggers in slot_table:
            in_periods = df_pd["period"].isin(periods_ok)
            slot = _slot_by_position(sr, triggers)
            for K in range(1, len(triggers) + 1):
                mask = is_mandatory & in_periods & (slot == K)
                df_pd.loc[mask, "timeout_role"] = f"slot_{K}_mandatory"

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

        # If the caller passed a ``CDNNBAMemoPL`` (or any memo exposing the
        # ``timeout_duration_s`` ``@memo_series``), materialize it as a
        # regular column so downstream plotting can read it without having
        # to recompute. Plain ``pl.DataFrame`` inputs skip this — they
        # don't carry the memo. ``getattr`` (not attribute access) because
        # the type of ``memo`` is not statically known to declare this.
        if (
            hasattr(type(memo), "timeout_duration_s")
            and isinstance(df, pl.DataFrame)
            and "timeout_duration_s" not in df.columns
        ):
            try:
                dur = getattr(memo, "timeout_duration_s")
                if isinstance(dur, pl.Series) and len(dur) == df.height:
                    df = df.with_columns(dur.alias("timeout_duration_s"))
            except Exception:  # noqa: BLE001 — memo may not be ready; degrade gracefully
                pass

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
