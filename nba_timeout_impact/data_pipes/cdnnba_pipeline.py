"""
Pipeline for loading, stacking, and cleaning cdn.nba.com play-by-play data.

Two-step workflow:
    1. load_and_stack_all  – load all cdnnba seasons (regular + playoff),
                             add an IsPlayoff column, save to parquet.
    2. clean_stacked       – load the stacked parquet, run the cleaning
                             pipeline, save the cleaned result.

Usage:
    from nba_timeout_impact.data_pipes.cdnnba_pipeline import CDNNBAPipelineHelper

    CDNNBAPipelineHelper.load_and_stack_all()
    CDNNBAPipelineHelper.clean_stacked()
"""

from pathlib import Path

import numpy as np
import pandas as pd
from kret_np_pd.UTILS_np_pd import NP_PD_Utils
from kret_sklearn.custom_transformers import PandasColumnOrderBase
from kret_sklearn.pd_pipeline import PipelinePD
from sklearn.preprocessing import FunctionTransformer

from nba_timeout_impact.constants import NBAConstants
from nba_timeout_impact.data_pipes.load_data_utils import NBADataLoader

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OUT_DIR = NBAConstants.NBA_DATA_DIR
_STACKED_PATH = _OUT_DIR / "cdnnba_stacked.parquet"
_CLEAN_PATH = _OUT_DIR / "cdnnba_clean.parquet"

# Columns that only appear in some seasons — drop for consistency
_DROP_COLS = {"value", "jerseyNumber", "shortFormattedClock"}

# Float columns that are actually whole numbers — cast to (nullable) int
_FLOAT_TO_INT_COLS = [
    "teamId",
    "pointsTotal",
    "jumpBallWonPersonId",
    "jumpBallLostPersonId",
    "jumpBallRecoverdPersonId",
    "officialId",
    "turnoverTotal",
    "foulPersonalTotal",
    "foulTechnicalTotal",
    "foulDrawnPersonId",
    "assistPersonId",
    "assistTotal",
    "shotActionNumber",
    "reboundTotal",
    "reboundDefensiveTotal",
    "reboundOffensiveTotal",
    "blockPersonId",
    "stealPersonId",
    "xLegacy",
    "yLegacy",
]

# String columns that are ISO 8601 datetimes
_DATETIME_COLS = ["timeActual", "edited"]

# Columns to convert to category
_CAT_COLS = [
    "actionType",
    "subType",
    "periodType",
    "playerName",
    "playerNameI",
    "teamTricode",
    "shotResult",
    "side",
    "area",
    "areaDetail",
]

# Desired column ordering after cleaning
_START_COLS = [
    "game_date",
    "gameId",
    "orderNumber",
    "actionType",
    "subType",
    "description",
    "scoreHome",
    "scoreAway",
    "pointsTotal",
    "possession",
    "period",
    "game_seconds_elapsed",
    "seconds_remaining",
    "seconds_elapsed",
    "IsPlayoff",
]

_END_COLS = [
    "playerNameI",
    "actionNumber",
    "clock",
    "timeActual",
    "teamId",
    "personId",
    "playerName",
    "xLegacy",
    "yLegacy",
    "x",
    "y",
    "shotDistance",
    "videoAvailable",
]


# ---------------------------------------------------------------------------
# Transformers
# ---------------------------------------------------------------------------


class StripStringColumns(PandasColumnOrderBase):
    """Strip leading/trailing whitespace from every object/string column."""

    def _fit(self, X: pd.DataFrame, y=None) -> "StripStringColumns":
        # Only strip columns whose non-null values are actually strings
        self._str_cols = [
            col
            for col in X.select_dtypes(include=["object", "string"]).columns
            if X[col].dropna().head(100).apply(type).eq(str).all()  # type: ignore
        ]
        return self

    def _transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        for col in self._str_cols:
            X[col] = X[col].str.strip()
        self.new_columns = list(X.columns)
        return X


class CDNNBAClockParser(PandasColumnOrderBase):
    """
    Parse the ISO 8601 ``clock`` column (e.g. 'PT04M56.00S') into:

    seconds_remaining  – time left in the period
    seconds_elapsed    – time played so far in the period
    """

    REG_PERIOD_SECS = 720
    OT_PERIOD_SECS = 300

    def _fit(self, X: pd.DataFrame, y=None) -> "CDNNBAClockParser":
        assert "clock" in X.columns
        assert "period" in X.columns
        return self

    @staticmethod
    def _parse_clock(series: pd.Series) -> pd.Series:
        parsed = series.str.extract(r"PT(\d+)M([\d.]+)S").astype(float)
        return parsed[0] * 60 + parsed[1]

    def _transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        period_len = np.where(X["period"] <= 4, self.REG_PERIOD_SECS, self.OT_PERIOD_SECS)
        X["seconds_remaining"] = self._parse_clock(X["clock"])
        X["seconds_elapsed"] = period_len - X["seconds_remaining"]
        self.new_columns = list(X.columns)
        return X


class GameTimeElapsed(PandasColumnOrderBase):
    """
    Add ``game_seconds_elapsed`` — absolute seconds from tip-off.

    Regulation periods 1-4 : each 720 s  (12 min)
    OT periods 5, 6, 7 …  : each 300 s  (5 min)
    """

    REG_PERIOD_SECS = 720
    OT_PERIOD_SECS = 300
    REG_PERIODS = 4

    def _fit(self, X: pd.DataFrame, y=None) -> "GameTimeElapsed":
        assert "period" in X.columns
        assert "seconds_elapsed" in X.columns
        return self

    def _transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        reg_offset = np.minimum(X["period"] - 1, self.REG_PERIODS) * self.REG_PERIOD_SECS
        ot_offset = np.maximum(X["period"] - self.REG_PERIODS - 1, 0) * self.OT_PERIOD_SECS
        X["game_seconds_elapsed"] = reg_offset + ot_offset + X["seconds_elapsed"]
        self.new_columns = list(X.columns)
        return X


class DeriveGameDate(PandasColumnOrderBase):
    """
    Derive ``game_date`` from the ``timeActual`` wall-clock timestamp.
    No external date file needed — cdnnba embeds real timestamps.
    """

    def _fit(self, X: pd.DataFrame, y=None) -> "DeriveGameDate":
        assert "timeActual" in X.columns
        return self

    def _transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        ts = pd.to_datetime(X["timeActual"], format="ISO8601", utc=True)
        # NBA games that tip off late evening (ET) can cross midnight UTC.
        # Use the date of the first event per game to tag every row.
        first_date = ts.groupby(X["gameId"]).transform("first").dt.date
        X["game_date"] = pd.to_datetime(pd.Series(first_date, index=X.index))
        self.new_columns = list(X.columns)
        return X


class SortByDate(PandasColumnOrderBase):
    """Sort by game_date, gameId, orderNumber."""

    sort_cols = ["game_date", "gameId", "orderNumber"]

    def _fit(self, X: pd.DataFrame, y=None) -> "SortByDate":
        for col in self.sort_cols:
            assert col in X.columns, f"Missing column: {col}"
        return self

    def _transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.sort_values(self.sort_cols).reset_index(drop=True)
        self.new_columns = list(X.columns)
        return X


class CategoricalConversion(PandasColumnOrderBase):
    cols: list[str]

    def __init__(self, cols: list[str]) -> None:
        super().__init__()
        self.cols = cols

    def _fit(self, X: pd.DataFrame, y=None) -> "CategoricalConversion":
        return self

    def _transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        for col in self.cols:
            if col in X.columns:
                X[col] = pd.Categorical(X[col])
        self.new_columns = list(X.columns)
        return X


class EnforceMonotonicity(PandasColumnOrderBase):
    """Enforce non-decreasing scores within each game via cummax."""

    cols: list[str]

    def __init__(self, cols: list[str]) -> None:
        super().__init__()
        self.cols = cols

    def _fit(self, X: pd.DataFrame, y=None) -> "EnforceMonotonicity":
        return self

    def _transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        for col in self.cols:
            X[col] = X.groupby("gameId")[col].cummax()
        self.new_columns = list(X.columns)
        return X


class CastFloatToInt(PandasColumnOrderBase):
    """Cast float64 columns that only contain whole numbers to nullable Int64.

    Columns with NaN are cast to pd.Int64Dtype() (nullable integer).
    Columns without NaN are cast to plain int64.
    """

    cols: list[str]

    def __init__(self, cols: list[str]) -> None:
        super().__init__()
        self.cols = cols

    def _fit(self, X: pd.DataFrame, y=None) -> "CastFloatToInt":
        return self

    def _transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        for col in self.cols:
            if col not in X.columns:
                continue
            if X[col].isna().any():
                X[col] = X[col].astype(pd.Int64Dtype())
            else:
                X[col] = X[col].astype(np.int64)
        self.new_columns = list(X.columns)
        return X


class ParseDatetimeColumns(PandasColumnOrderBase):
    """Parse string columns to datetime."""

    cols: list[str]

    def __init__(self, cols: list[str]) -> None:
        super().__init__()
        self.cols = cols

    def _fit(self, X: pd.DataFrame, y=None) -> "ParseDatetimeColumns":
        return self

    def _transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        for col in self.cols:
            if col not in X.columns:
                continue
            X[col] = pd.to_datetime(X[col], format="ISO8601", utc=True)
        self.new_columns = list(X.columns)
        return X


# ---------------------------------------------------------------------------
# Pipeline helper
# ---------------------------------------------------------------------------


class CDNNBAPipelineHelper:

    @classmethod
    def load_and_stack_all(cls, out_path: str | Path = _STACKED_PATH) -> pd.DataFrame:
        """Load all cdnnba seasons (regular + playoff), stack, save to parquet."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        frames: list[pd.DataFrame] = []
        for playoffs in (False, True):
            seasons = NBADataLoader.available_seasons("cdnnba", playoffs=playoffs)
            for s in seasons:
                df = NBADataLoader.load_season("cdnnba", s, playoffs=playoffs)
                df["IsPlayoff"] = playoffs
                frames.append(df)
                print(f"  Loaded {'PO' if playoffs else 'RS'} {s}: {len(df):,} rows, {len(df.columns)} cols")

        stacked = pd.concat(frames, ignore_index=True)

        # Drop columns that only appear in some seasons
        drop = _DROP_COLS & set(stacked.columns)
        if drop:
            stacked = stacked.drop(columns=list(drop))
            print(f"  Dropped inconsistent columns: {drop}")

        stacked.to_parquet(out_path, index=False, compression="zstd")
        print(f"\nSaved stacked: {len(stacked):,} rows x {len(stacked.columns)} cols -> {out_path}")
        return stacked

    @classmethod
    def clean_stacked(cls, in_path: str | Path = _STACKED_PATH, out_path: str | Path = _CLEAN_PATH) -> pd.DataFrame:
        """Load stacked parquet, run cleaning pipeline, save cleaned result."""
        in_path, out_path = Path(in_path), Path(out_path)

        score_cols = ["scoreHome", "scoreAway"]
        # Only include end cols that will actually be present
        end_cols = [c for c in _END_COLS if c != "videoAvailable"]

        sort_cols_fn = FunctionTransformer(
            func=NP_PD_Utils.move_columns, validate=False, kw_args={"start": _START_COLS, "end": end_cols}
        )

        pipeline = PipelinePD(
            steps=[
                ("strip_strings", StripStringColumns()),
                ("parse_clock", CDNNBAClockParser()),
                ("game_time", GameTimeElapsed()),
                ("derive_date", DeriveGameDate()),
                ("sort", SortByDate()),
                ("cats", CategoricalConversion(_CAT_COLS)),
                ("enforce_monotonicity", EnforceMonotonicity(score_cols)),
                ("cast_int", CastFloatToInt(_FLOAT_TO_INT_COLS)),
                ("parse_datetimes", ParseDatetimeColumns(_DATETIME_COLS)),
                ("sort_cols", sort_cols_fn),
            ]
        )

        print(f"Loading stacked data from {in_path}...")
        df = pd.read_parquet(in_path)
        print(f"  {len(df):,} rows x {len(df.columns)} cols")

        print("Running cleaning pipeline...")
        clean = pipeline.fit_transform_df(df)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        clean.to_parquet(out_path, index=False, compression="zstd")
        print(f"\nSaved clean: {len(clean):,} rows x {len(clean.columns)} cols -> {out_path}")
        return clean
