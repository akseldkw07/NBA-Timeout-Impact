import typing as t
from pathlib import Path

import numpy as np
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

    @memo_array
    def f_timeout_exogenous(self):
        return self.f_timeout & (self.data.subType.isin(["Official", "Official TV"]))

    @memo_array
    def f_timeout_endogenous(self):
        return self.f_timeout & self.data.subType.isin(["Regular", "Short", "Coach Challenge"])

    @memo_array
    def streak(self):
        df = self.data
        game_start = df["gameId"].ne(df["gameId"].shift(fill_value=-1))

        # Zero out at game boundaries — no NaN, no bleed from previous game
        home_pts = df["scoreHome"].diff().clip(lower=0).where(~game_start, 0).fillna(0).astype(int)
        away_pts = df["scoreAway"].diff().clip(lower=0).where(~game_start, 0).fillna(0).astype(int)
        net = home_pts - away_pts

        scorer = t.cast(pd.Series, np.sign(net).replace(0, np.nan)).ffill().fillna(0)  # type: ignore

        # game_start acts as a hard reset even if scorer bleeds across boundary
        new_segment = game_start | ((net != 0) & (scorer != scorer.shift(fill_value=0)))
        segment = new_segment.cumsum()

        ret = net.groupby(segment).cumsum()
        ret.name = "streak"
        return ret

    @memo_array
    def f_streak_6(self):
        return np.abs(self.streak) >= 6

    @memo_array
    def f_streak_9(self):
        return np.abs(self.streak) >= 9

    @memo_array
    def f_streak_12(self):
        return np.abs(self.streak) >= 12


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
        ret = cls(df)
        ret.validate_data()
        return ret

    def validate_data(self):
        print("Validating data...")
        assert (
            self.game_date_ffill.is_monotonic_increasing
        ), "game_date_ffill must be sorted, but found non-monotonic values at indices "

        assert (
            self["gameId"].diff().ne(0) & self["gameId"].duplicated()
        ).sum() == 0, f"gameId must be sorted and non-duplicated, but found duplicates at indices {self['gameId'].index[self['gameId'].duplicated()]} "

        for col in ["actionId", "scoreHome", "scoreAway"]:
            assert col in self.columns, f"Expected column '{col}' not found in DataFrame."
            assert self[col].dtype in [int], f"Column '{col}' must be int, but found dtype {self[col].dtype}."
            assert (
                self[col].isna().sum() == 0
            ), f"Column '{col}' must not contain NaN values, but found {self[col].isna().sum()} NaNs."

            check = self.groupby("gameId")[col].is_monotonic_increasing
            assert (
                check.all()
            ), f"{col} must be non-decreasing within each gameId, but found decreases at gameIds {check.index[~check]}"
