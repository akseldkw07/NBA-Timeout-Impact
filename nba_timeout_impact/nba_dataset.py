from pathlib import Path

import pandas as pd
from kret_np_pd.enriched_df import Enriched_DF
from kret_np_pd.memo_df import InputTypedDict, MemoDataFrame, memo_array

from nba_timeout_impact.constants import NBAConstants


class NBADatasetInput_TypedDict(InputTypedDict):
    data: "NBADataset"


class NBAMemoDF(MemoDataFrame[NBADatasetInput_TypedDict]):
    @property
    def data(self) -> "NBADataset":
        return self.inputs["data"]

    @memo_array
    def f_timeout(self):
        return self.data.actionType == "Timeout"


class NBADataset(Enriched_DF):
    actionNumber: pd.Series  # int64
    clock: pd.Series  # str
    period: pd.Series  # int64
    teamId: pd.Series  # int64
    teamTricode: pd.Series  # category
    personId: pd.Series  # int64
    playerName: pd.Series  # category
    playerNameI: pd.Series  # category
    xLegacy: pd.Series  # int64
    yLegacy: pd.Series  # int64
    shotDistance: pd.Series  # int64
    shotResult: pd.Series  # category
    isFieldGoal: pd.Series  # int64
    scoreHome: pd.Series  # float64
    scoreAway: pd.Series  # float64
    pointsTotal: pd.Series  # int64
    location: pd.Series  # category
    description: pd.Series  # str
    actionType: pd.Series  # category
    subType: pd.Series  # category
    videoAvailable: pd.Series  # int64
    actionId: pd.Series  # int64
    gameId: pd.Series  # int64
    shotValue: pd.Series  # float64
    IsPlayoff: pd.Series  # bool
    seconds_remaining: pd.Series  # float64
    seconds_elapsed: pd.Series  # float64
    game_date: pd.Series  # datetime64[ms]
    game_date_ffill: pd.Series  # datetime64[ms]

    @classmethod
    def load_from_parquet(cls, path: str | Path | None = None) -> "NBADataset":
        path = path or NBAConstants.NBA_DATA_DIR / "nba_statsv3_clean.parquet"
        df = pd.read_parquet(path)
        return cls(df)
