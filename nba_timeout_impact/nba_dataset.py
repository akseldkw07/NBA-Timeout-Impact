import typing as t
from pathlib import Path

import numpy as np
import pandas as pd
from kret_np_pd.enriched_df import Enriched_DF
from kret_np_pd.memo_df import InputTypedDict, MemoDataFrame, memo_array, memo_fn

from nba_timeout_impact.constants import NBAConstants


class NBADatasetInput_TypedDict(InputTypedDict):
    data: "NBADataset"


class NBAMemoDF(MemoDataFrame[NBADatasetInput_TypedDict]):
    @property
    def data(self) -> "NBADataset":
        return self.inputs["data"]

    """PTRS"""

    @property
    def data_sorted_time_elapsed(self):
        key = "_data_sorted_time_elapsed"
        if key not in self._df_dict:
            df = self.data
            pos = np.arange(len(df), dtype=np.intp)
            self._df_dict[key] = pd.DataFrame(
                {
                    "gameId": df["gameId"].values,
                    "game_seconds_elapsed": df["game_seconds_elapsed"].values,
                    "ptr": pos,
                }
            ).sort_values("game_seconds_elapsed")
        return self._df_dict[key]

    @memo_fn
    def ptr_n_minutes(self, n: float) -> np.ndarray:
        """
        For every event, return the positional index (0-based row number) of
        the closest event ``n`` minutes in the future (n > 0) or past (n < 0)
        within the same game.

        Uses ``game_seconds_elapsed`` — robust to OT.
        Returns -1 where no match exists (e.g. looking back before tip-off).

        Example
        -------
        >>> ptrs = nba_memo.ptr_n_minutes(3)
        >>> nba_memo.data.iloc[ptrs[ptrs != -1]]   # events 3 min ahead
        """
        window = n * 60.0
        lookup = self.data_sorted_time_elapsed  # sorted by game_seconds_elapsed, never mutate

        # separate queries table — copy avoids mutating the cached lookup
        queries = lookup.copy()
        queries["target_time"] = queries["game_seconds_elapsed"] + window
        queries = queries.sort_values("target_time")

        merged = pd.merge_asof(
            queries,
            lookup,
            left_on="target_time",
            right_on="game_seconds_elapsed",
            by="gameId",
            direction="nearest",
            suffixes=("_query", "_result"),
        )

        # ptr_query = original row position, ptr_result = matched row position
        ret = merged.sort_values("ptr_query")["ptr_result"].fillna(-1).astype(np.intp)
        return ret.to_numpy()

    """TIMEOUTS"""

    @memo_array
    def f_timeout(self):
        return self.data.actionType == "Timeout"

    @memo_array
    def f_timeout_exogenous(self):
        return self.f_timeout & (self.data.subType.isin(["Official", "Official TV"]))

    @memo_array
    def f_timeout_endogenous(self):
        return self.f_timeout & self.data.subType.isin(["Regular", "Short", "Coach Challenge"])

    """LEAD & LEAD CHANGE"""

    @memo_array
    def lead(self):
        return self.data.scoreHome - self.data.scoreAway

    @memo_fn
    def lead_change_n_minutes(self, n: float):
        diff = self.lead[self.ptr_n_minutes(n)] - self.lead
        return diff

    """STREAKS"""

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

    @memo_fn
    def f_streak_n(self, n: int, direction: t.Literal["home", "away", "either"] = "either"):
        if direction == "home":
            return self.streak >= n
        elif direction == "away":
            return self.streak <= -n
        elif direction == "either":
            return np.abs(self.streak) >= n
        else:
            raise ValueError(f"Invalid direction: {direction}")


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
    game_seconds_elapsed: pd.Series  # float64
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

        for col in ["actionId", "scoreHome", "scoreAway", "game_seconds_elapsed"]:
            assert col in self.columns, f"Expected column '{col}' not found in DataFrame."
            assert self[col].dtype in [
                int,
                float,
            ], f"Column '{col}' must be int or float, but found dtype {self[col].dtype}."
            assert (
                self[col].isna().sum() == 0
            ), f"Column '{col}' must not contain NaN values, but found {self[col].isna().sum()} NaNs."

            check = self.groupby("gameId")[col].is_monotonic_increasing
            assert (
                check.all()
            ), f"{col} must be non-decreasing within each gameId, but found decreases at gameIds {check.index[~check]}"
