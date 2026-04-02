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

    # -- tail columns --
    actionNumber: pd.Series  # int64
    clock: pd.Series  # str
    timeActual: pd.Series  # datetime64[us, UTC]

    @classmethod
    def load_from_parquet(cls, path: str | Path | None = None) -> "CDNNBADataset":
        path = path or NBAConstants.NBA_DATA_DIR / "cdnnba_clean.parquet"
        df = pd.read_parquet(path)
        ret = cls(df)
        ret.validate_data()
        return ret

    def validate_data(self):
        print("Validating cdnnba data...")
        assert self["game_date"].is_monotonic_increasing, "game_date must be sorted."

        assert (
            self["gameId"].diff().ne(0) & self["gameId"].duplicated()
        ).sum() == 0, "gameId must be contiguously grouped."

        for col in ["orderNumber", "scoreHome", "scoreAway", "game_seconds_elapsed"]:
            assert col in self.columns, f"Missing column: {col}"
            assert self[col].isna().sum() == 0, f"NaN values in {col}"

        # orderNumber and scores must be strictly non-decreasing
        for col in ["orderNumber", "scoreHome", "scoreAway"]:
            check = self.groupby("gameId")[col].is_monotonic_increasing
            assert check.all(), (
                f"{col} must be non-decreasing within each gameId, "
                f"violations at gameIds: {check.index[~check].tolist()[:5]}"
            )

        # game_seconds_elapsed can have minor clock reversals (e.g. instant replay
        # events logged at a fractionally earlier clock than the preceding event).
        # Warn but don't fail.
        gse_check = self.groupby("gameId")["game_seconds_elapsed"].is_monotonic_increasing
        if not gse_check.all():
            n_bad = (~gse_check).sum()
            print(
                f"  Warning: game_seconds_elapsed has minor reversals in {n_bad} games (data quirk, not a pipeline bug)."
            )
