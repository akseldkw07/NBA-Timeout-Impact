import typing as t
from pathlib import Path

import numpy as np
import pandas as pd
from kret_np_pd.enriched_df import Enriched_DF
from kret_np_pd.memo_df import InputTypedDict, MemoDataFrame, memo_array, memo_fn

from nba_timeout_impact.constants import NBAConstants


class NBADatasetInput_TypedDict(InputTypedDict):
    data: "NBADataset"
    v2: "NBAStatsV2Dataset"


class NBAMemoDF(MemoDataFrame[NBADatasetInput_TypedDict]):
    @property
    def data(self) -> "NBADataset":
        return self.inputs["data"]

    @property
    def v2(self) -> "NBAStatsV2Dataset":
        """nbastats (v2 nba_api format). Has WCTIMESTRING (wall-clock time of
        day, minute precision) which v3 lacks.
        """
        return self.inputs["v2"]

    # -- load_all --

    @classmethod
    def load_all(cls, v3_path: str | Path | None = None, v2_path: str | Path | None = None) -> "NBAMemoDF":
        """Load nbastatsv3 + nbastats (v2) parquets, ready for joining on
        (gameId, period, pc_seconds). v2 contributes WCTIMESTRING (wall-clock
        time-of-day) which v3 doesn't carry.
        """
        return cls(
            {
                "data": NBADataset.load_from_parquet(v3_path),
                "v2": NBAStatsV2Dataset.load_from_parquet(v2_path),
            }
        )

    # -- v3 <-> v2 alignment --

    @property
    def _v2_minute_spine(self) -> pd.DataFrame:
        """Cached one-row-per-(gameId, period, pc_seconds) lookup from v2."""
        key = "_v2_minute_spine"
        if key not in self._df_dict:
            v2 = self.v2
            spine = (
                v2[v2["wc_minute"].notna()]
                .groupby(["gameId", "period", "pc_seconds"], as_index=False, observed=True)
                .agg(WCTIMESTRING=("WCTIMESTRING", "first"), wc_minute=("wc_minute", "first"))
            )
            self._df_dict[key] = spine
        return self._df_dict[key]

    @memo_array
    def wc_minute(self):
        """Wall-clock minute-of-day for each v3 row, joined from v2 on
        (gameId, period, pc_seconds). Null where no v2 row matches.

        Encoded as ``(hour % 12) * 60 + minute`` so deltas within a game are
        meaningful (games don't span noon/midnight in practice).
        """
        v3 = self.data
        spine = self._v2_minute_spine
        keys = pd.DataFrame(
            {
                "gameId": v3["gameId"].to_numpy(),
                "period": v3["period"].to_numpy(),
                "pc_seconds": v3["pc_seconds"].to_numpy(),
            }
        )
        merged = keys.merge(
            spine[["gameId", "period", "pc_seconds", "wc_minute"]],
            on=["gameId", "period", "pc_seconds"],
            how="left",
        )
        return pd.Series(merged["wc_minute"].to_numpy(), index=v3.index, name="wc_minute")

    @memo_array
    def wctimestring(self):
        """Raw WCTIMESTRING ("H:MM PM") joined from v2, same key as wc_minute."""
        v3 = self.data
        spine = self._v2_minute_spine
        keys = pd.DataFrame(
            {
                "gameId": v3["gameId"].to_numpy(),
                "period": v3["period"].to_numpy(),
                "pc_seconds": v3["pc_seconds"].to_numpy(),
            }
        )
        merged = keys.merge(
            spine[["gameId", "period", "pc_seconds", "WCTIMESTRING"]],
            on=["gameId", "period", "pc_seconds"],
            how="left",
        )
        return pd.Series(merged["WCTIMESTRING"].to_numpy(), index=v3.index, name="WCTIMESTRING")

    """TIME"""

    @memo_fn
    def bin_sr(self, width: int = 60):
        arr = (self.data["seconds_remaining"] // width) * width
        return pd.Categorical(arr)

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
    def ptr_n_mins(self, n: float) -> np.ndarray:
        """
        For every event, return the positional index (0-based row number) of
        the closest event ``n`` minutes in the future (n > 0) or past (n < 0)
        within the same game.

        Uses ``game_seconds_elapsed`` — robust to OT.
        Returns -1 where no match exists (e.g. looking back before tip-off).

        Example
        -------
        >>> ptrs = nba_memo.ptr_n_mins(3)
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
    def lead_change_n_mins(self, n: float):
        """
        If value is positive, home team lead increased (or deficit decreased) after n minutes.

        If value is negative, home team lead decreased (or deficit increased) after n minutes.
        """
        diff = self.lead[self.ptr_n_mins(n)] - self.lead
        return diff

    @memo_fn
    def score_diff_n_mins(self, n: float):
        """
        Alias for lead_change_n_minutes, since score diff and lead are the same thing — just different sign conventions.
        """
        return self.lead_change_n_mins(n)

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
    def load_from_parquet(cls, path: str | Path | None = None, enrich: bool = True) -> "NBADataset":
        path = path or NBAConstants.NBA_DATA_DIR / "nbastatsv3.parquet"
        df = pd.read_parquet(path)
        if enrich:
            df = _derive_v3_columns(df)
        # Parquet is persisted pre-sorted by (gameId, actionNumber); skip sort on load.
        # If you re-save the parquet, sort first to maintain the invariant.
        return cls(df)

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


# ---------------------------------------------------------------------------
# v2 (nbastats / nba_api) dataset wrapper + shared parsing helpers
# ---------------------------------------------------------------------------


_PERIOD_LEN_REG = 720.0  # regulation period length in seconds
_PERIOD_LEN_OT = 300.0


def _parse_pt_clock(clock: pd.Series) -> pd.Series:
    """Parse v3 game clock 'PT12M00.00S' -> seconds remaining in period (float)."""
    s = clock.astype("string")
    m = pd.to_numeric(s.str.extract(r"PT(\d+)M", expand=False), errors="coerce")
    sec = pd.to_numeric(s.str.extract(r"M([\d.]+)S", expand=False), errors="coerce")
    return (m * 60 + sec).astype("float64")


def _parse_mmss_clock(clock: pd.Series) -> pd.Series:
    """Parse v2 PCTIMESTRING 'MM:SS' -> seconds remaining in period (Int32)."""
    parts = clock.astype("string").str.split(":", n=1, expand=True)
    mm = pd.to_numeric(parts[0], errors="coerce")
    ss = pd.to_numeric(parts[1], errors="coerce")
    return (mm * 60 + ss).astype("Int32")


def _derive_v3_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns NBADataset declares but the raw nbastatsv3 parquet lacks.

    Idempotent: only fills in what's missing.
    """
    if "seconds_remaining" not in df.columns:
        df["seconds_remaining"] = _parse_pt_clock(df["clock"])
    if "seconds_elapsed" not in df.columns:
        period_len = np.where(df["period"].to_numpy() <= 4, _PERIOD_LEN_REG, _PERIOD_LEN_OT)
        df["seconds_elapsed"] = (period_len - df["seconds_remaining"].to_numpy()).astype("float64")
    if "game_seconds_elapsed" not in df.columns:
        per = df["period"].to_numpy()
        prior = np.where(
            per <= 1,
            0.0,
            np.where(per <= 4, (per - 1) * _PERIOD_LEN_REG, 4 * _PERIOD_LEN_REG + (per - 5) * _PERIOD_LEN_OT),
        )
        df["game_seconds_elapsed"] = (prior + df["seconds_elapsed"].to_numpy()).astype("float64")
    # pc_seconds is the join key against v2 (integer-typed for clean equality)
    if "pc_seconds" not in df.columns:
        df["pc_seconds"] = df["seconds_remaining"].round().astype("Int32")
    if "IsPlayoff" not in df.columns and "season_type" in df.columns:
        df["IsPlayoff"] = df["season_type"].astype(str).eq("po")
    if "shotValue" not in df.columns:
        df["shotValue"] = 0.0
    if "game_date" not in df.columns:
        df["game_date"] = pd.NaT
    if "game_date_ffill" not in df.columns:
        df["game_date_ffill"] = pd.NaT
    return df


def _derive_v2_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add gameId (int), period, pc_seconds, and wc_minute to a raw v2 frame."""
    if "gameId" not in df.columns:
        df["gameId"] = pd.to_numeric(df["GAME_ID"], errors="coerce").astype("Int64")
    if "period" not in df.columns:
        df["period"] = df["PERIOD"].astype("Int32")
    if "pc_seconds" not in df.columns:
        df["pc_seconds"] = _parse_mmss_clock(df["PCTIMESTRING"])
    if "wc_minute" not in df.columns:
        wc = df["WCTIMESTRING"].astype("string")
        h = pd.to_numeric(wc.str.extract(r"^(\d+):", expand=False), errors="coerce")
        m = pd.to_numeric(wc.str.extract(r":(\d+)", expand=False), errors="coerce")
        df["wc_minute"] = ((h.mod(12)) * 60 + m).astype("Int32")
    return df


class NBAStatsV2Dataset(Enriched_DF):
    """nbastats (v2 nba_api format). Saved by ``NBADataLoader.load_season('nbastats', ...)``
    and persisted to ``NBA_DATA_DIR / 'nbastats.parquet'``.

    The key add over v3 is ``WCTIMESTRING`` — wall-clock time of day at
    minute precision (e.g. ``"7:12 PM"``). v2 itself uses ``GAME_ID`` (string,
    zero-padded) and ``PERIOD`` (uppercase); ``load_from_parquet`` normalizes
    these to ``gameId`` (int) and ``period`` for joining with v3.

    Wall-clock data is only reliable for seasons 2009+ (earlier seasons have
    corrupted WCTIMESTRING values).
    """

    GAME_ID: pd.Series  # str ("0021300001")
    EVENTNUM: pd.Series  # int
    EVENTMSGTYPE: pd.Series  # int (9 = Timeout)
    EVENTMSGACTIONTYPE: pd.Series  # int (1=Regular, 2=Short, 4=Official)
    PERIOD: pd.Series  # int
    WCTIMESTRING: pd.Series  # str like "7:12 PM"
    PCTIMESTRING: pd.Series  # str like "12:00"
    HOMEDESCRIPTION: pd.Series
    VISITORDESCRIPTION: pd.Series
    NEUTRALDESCRIPTION: pd.Series
    SCORE: pd.Series
    SCOREMARGIN: pd.Series
    season: pd.Series
    season_type: pd.Series
    # Derived
    gameId: pd.Series  # int
    period: pd.Series  # int
    pc_seconds: pd.Series  # Int32, seconds remaining in period
    wc_minute: pd.Series  # Int32, (hour % 12) * 60 + minute

    @classmethod
    def load_from_parquet(cls, path: str | Path | None = None) -> "NBAStatsV2Dataset":
        path = path or NBAConstants.NBA_DATA_DIR / "nbastats.parquet"
        df = pd.read_parquet(path)
        df = _derive_v2_columns(df)
        return cls(df)
