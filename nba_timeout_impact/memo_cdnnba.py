"""
Enriched DataFrame and MemoDataFrame for cleaned cdn.nba.com play-by-play data.

Usage:
    from nba_timeout_impact.memo_cdnnba import CDNNBADataset, CDNNBAMemoDF

    ds = CDNNBADataset.load_from_parquet()
    memo = CDNNBAMemoDF({"data": ds})
"""

from pathlib import Path

import numpy as np
import pandas as pd
from kret_np_pd.enriched_df import Enriched_DF
from kret_np_pd.memo_df import InputTypedDict, MemoDataFrame, memo_array

from nba_timeout_impact.constants import NBAConstants


class CDNNBADatasetInput_TypedDict(InputTypedDict):
    data: "CDNNBADataset"


class CDNNBAMemoDF(MemoDataFrame[CDNNBADatasetInput_TypedDict]):
    @property
    def data(self) -> "CDNNBADataset":
        return self.inputs["data"]

    @memo_array
    def f_clock_reversal(self) -> np.ndarray:
        """Boolean mask: True for rows where game_seconds_elapsed decreases
        relative to the previous row within the same game.

        These are caused by:
        - ``instantreplay`` requests logged with a slightly stale clock
        - ``memo`` events (stat corrections / annotations) carrying the
          clock of the original play they reference

        Both are non-play metadata — safe to exclude from time-based analysis.
        """
        df = self.data
        game_boundary = df["gameId"].ne(df["gameId"].shift(fill_value=-1))
        gse_diff = df["game_seconds_elapsed"].diff()
        return np.array(((gse_diff < 0) & ~game_boundary).values, dtype=bool)


class CDNNBADataset(Enriched_DF):
    # -- key columns (sorted to front by pipeline) --
    game_date: pd.Series  # datetime64[s]
    gameId: pd.Series  # int64
    orderNumber: pd.Series  # int64
    actionType: pd.Series  # category
    subType: pd.Series  # category
    description: pd.Series  # str
    scoreHome: pd.Series  # int64
    scoreAway: pd.Series  # int64
    pointsTotal: pd.Series  # Int64 (nullable)
    possession: pd.Series  # int64
    period: pd.Series  # int64
    game_seconds_elapsed: pd.Series  # float64
    seconds_remaining: pd.Series  # float64
    seconds_elapsed: pd.Series  # float64
    IsPlayoff: pd.Series  # bool

    # -- context columns --
    periodType: pd.Series  # category
    qualifiers: pd.Series  # str
    edited: pd.Series  # datetime64[us, UTC]
    isFieldGoal: pd.Series  # int64
    side: pd.Series  # category
    personIdsFilter: pd.Series  # str
    descriptor: pd.Series  # str
    isTargetScoreLastPeriod: pd.Series  # object (mixed bool/str)
    teamId: pd.Series  # Int64 (nullable)
    teamTricode: pd.Series  # category

    # -- shot detail --
    area: pd.Series  # category
    areaDetail: pd.Series  # category
    shotDistance: pd.Series  # float64
    shotResult: pd.Series  # category
    shotActionNumber: pd.Series  # Int64 (nullable)
    x: pd.Series  # float64
    y: pd.Series  # float64
    xLegacy: pd.Series  # Int64 (nullable)
    yLegacy: pd.Series  # Int64 (nullable)

    # -- player / assist / block / steal / rebound --
    personId: pd.Series  # int64
    playerName: pd.Series  # category
    playerNameI: pd.Series  # category
    assistPlayerNameInitial: pd.Series  # str
    assistPersonId: pd.Series  # Int64 (nullable)
    assistTotal: pd.Series  # Int64 (nullable)
    blockPlayerName: pd.Series  # str
    blockPersonId: pd.Series  # Int64 (nullable)
    stealPlayerName: pd.Series  # str
    stealPersonId: pd.Series  # Int64 (nullable)
    reboundTotal: pd.Series  # Int64 (nullable)
    reboundDefensiveTotal: pd.Series  # Int64 (nullable)
    reboundOffensiveTotal: pd.Series  # Int64 (nullable)
    turnoverTotal: pd.Series  # Int64 (nullable)

    # -- foul detail --
    foulPersonalTotal: pd.Series  # Int64 (nullable)
    foulTechnicalTotal: pd.Series  # Int64 (nullable)
    foulDrawnPlayerName: pd.Series  # str
    foulDrawnPersonId: pd.Series  # Int64 (nullable)

    # -- jump ball --
    jumpBallRecoveredName: pd.Series  # str
    jumpBallRecoverdPersonId: pd.Series  # Int64 (nullable)
    jumpBallWonPlayerName: pd.Series  # str
    jumpBallWonPersonId: pd.Series  # Int64 (nullable)
    jumpBallLostPlayerName: pd.Series  # str
    jumpBallLostPersonId: pd.Series  # Int64 (nullable)

    # -- official --
    officialId: pd.Series  # Int64 (nullable)

    # -- enriched columns --
    season: pd.Series  # int64
    season_type: pd.Series  # str
    shot_value: pd.Series  # Int64 (nullable)
    points_scored: pd.Series  # int64
    score_margin: pd.Series  # int64
    is_clutch: pd.Series  # bool
    prev_action_type: pd.Series  # category
    x_court: pd.Series  # Int32 (nullable)
    y_court: pd.Series  # Int32 (nullable)
    possession_id: pd.Series  # Int64 (nullable)
    possession_points: pd.Series  # Int64 (nullable)
    possession_outcome: pd.Series  # str

    # -- tail columns --
    actionNumber: pd.Series  # int64
    clock: pd.Series  # str
    timeActual: pd.Series  # datetime64[us, UTC]

    @classmethod
    def load_from_parquet(cls, path: str | Path | None = None) -> "CDNNBADataset":
        path = path or NBAConstants.NBA_DATA_DIR / "cdnnba_enriched.parquet"
        df = pd.read_parquet(path)
        ret = cls(df)
        ret.validate_data()
        return ret

    # Valid possession outcome values from the enrichment pipeline
    _VALID_OUTCOMES = {
        "made_2pt",
        "made_3pt",
        "made_2pt_and1",
        "made_3pt_and1",
        "miss",
        "miss_def_reb",
        "turnover_live",
        "turnover_dead",
        "end_of_period",
        "violation",
        "other",
    }

    def validate_data(self):
        print("Validating cdnnba data...")
        errors: list[str] = []
        warnings: list[str] = []

        # --- required columns ---
        required = list(self.__class__.__annotations__.keys())
        missing = [c for c in required if c not in self.columns]
        if missing:
            errors.append(f"Missing columns: {missing}")

        # --- sorting ---
        if not self["game_date"].is_monotonic_increasing:
            errors.append("game_date is not sorted.")

        if (self["gameId"].diff().ne(0) & self["gameId"].duplicated()).sum() > 0:
            errors.append("gameId is not contiguously grouped (games interleaved).")

        # --- no-NaN columns ---
        no_nan_cols = [
            "gameId",
            "orderNumber",
            "period",
            "scoreHome",
            "scoreAway",
            "game_seconds_elapsed",
            "seconds_remaining",
            "seconds_elapsed",
            "game_date",
            "actionType",
            "personId",
            "possession",
            "IsPlayoff",
            "season",
            "season_type",
            "points_scored",
            "score_margin",
            "is_clutch",
        ]
        for col in no_nan_cols:
            if col in self.columns and self[col].isna().any():
                errors.append(f"NaN values in non-nullable column '{col}' ({self[col].isna().sum():,})")

        # --- monotonicity within games ---
        for col in ["orderNumber", "scoreHome", "scoreAway"]:
            if col not in self.columns:
                continue
            check = self.groupby("gameId")[col].is_monotonic_increasing
            if not check.all():
                bad = check.index[~check].tolist()[:5]
                errors.append(f"{col} not non-decreasing within gameId, violations: {bad}")

        # game_seconds_elapsed — warn only (known data quirk)
        if "game_seconds_elapsed" in self.columns:
            gse_check = self.groupby("gameId")["game_seconds_elapsed"].is_monotonic_increasing
            if not gse_check.all():
                n_bad = (~gse_check).sum()
                warnings.append(
                    f"game_seconds_elapsed has minor reversals in {n_bad} games "
                    "(instant replay / memo events — not a pipeline bug)."
                )

        # --- value ranges ---
        if "period" in self.columns:
            if (self["period"] < 1).any() or (self["period"] > 10).any():
                errors.append(f"period out of range [1, 10]: min={self['period'].min()}, max={self['period'].max()}")

        if "seconds_remaining" in self.columns:
            if (self["seconds_remaining"] < 0).any():
                errors.append("seconds_remaining has negative values.")

        if "game_seconds_elapsed" in self.columns:
            if (self["game_seconds_elapsed"] < 0).any():
                errors.append("game_seconds_elapsed has negative values.")

        if "scoreHome" in self.columns and "scoreAway" in self.columns:
            if (self["scoreHome"] < 0).any() or (self["scoreAway"] < 0).any():
                errors.append("Negative scores found.")

        # --- enriched column checks ---
        if "season" in self.columns:
            valid_seasons = set(range(2020, 2030))
            actual = set(self["season"].unique())
            bad_seasons = actual - valid_seasons
            if bad_seasons:
                errors.append(f"Invalid season values: {bad_seasons}")

        if "season_type" in self.columns:
            actual_st = set(self["season_type"].unique())
            if not actual_st.issubset({"rg", "po"}):
                errors.append(f"Invalid season_type values: {actual_st - {'rg', 'po'}}")

        if "IsPlayoff" in self.columns and "season_type" in self.columns:
            mismatch = (self["IsPlayoff"] & (self["season_type"] != "po")) | (
                ~self["IsPlayoff"] & (self["season_type"] != "rg")
            )
            if mismatch.any():
                errors.append(f"IsPlayoff/season_type mismatch: {mismatch.sum():,} rows")

        if "shot_value" in self.columns:
            {1, 2, 3, pd.NA}
            actual_sv = set(self["shot_value"].dropna().unique())
            if not actual_sv.issubset({1, 2, 3}):
                errors.append(f"Invalid shot_value values: {actual_sv - {1, 2, 3}}")

        if "points_scored" in self.columns:
            actual_ps = set(self["points_scored"].unique())
            if not actual_ps.issubset({0, 1, 2, 3}):
                errors.append(f"Invalid points_scored values: {actual_ps - {0, 1, 2, 3}}")

        if "score_margin" in self.columns:
            expected_margin = self["scoreHome"] - self["scoreAway"]
            if not (self["score_margin"] == expected_margin).all():
                errors.append("score_margin != scoreHome - scoreAway")

        if "is_clutch" in self.columns:
            expected_clutch = (self["score_margin"].abs() <= 5) & (self["game_seconds_elapsed"] >= 2400.0)
            if not (self["is_clutch"] == expected_clutch).all():
                errors.append("is_clutch does not match definition (|margin| <= 5 AND elapsed >= 2400).")

        if "possession_outcome" in self.columns:
            outcomes = set(self["possession_outcome"].dropna().unique())
            # ft outcomes are dynamic (ft_X_of_Y), check the fixed ones
            fixed_outcomes = {o for o in outcomes if not o.startswith("ft_")}
            unexpected = fixed_outcomes - self._VALID_OUTCOMES
            if unexpected:
                warnings.append(f"Unexpected possession_outcome values: {unexpected}")

        if "possession_id" in self.columns and "possession_points" in self.columns:
            # Rows with possession_id should have possession_points and vice versa
            has_pid = self["possession_id"].notna()
            has_ppts = self["possession_points"].notna()
            if (has_pid != has_ppts).any():
                n = (has_pid != has_ppts).sum()
                warnings.append(f"possession_id/possession_points nullity mismatch: {n:,} rows")

        # --- report ---
        for w in warnings:
            print(f"  Warning: {w}")
        if errors:
            msg = "\n  ".join(errors)
            raise AssertionError(f"Validation failed:\n  {msg}")
        print(f"  Passed ({len(self):,} rows, {len(self.columns)} cols, {len(warnings)} warnings).")
