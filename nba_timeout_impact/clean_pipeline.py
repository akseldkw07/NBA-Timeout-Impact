"""
Cleaning pipeline for nbastatsv3 DataFrames.

Steps
-----
1. StripStringColumns   – strip leading/trailing whitespace from every str/object column
2. NBAStatsV3ClockParser – parse ISO 8601 clock strings into seconds_remaining / seconds_elapsed

Usage
-----
    from nba_timeout_impact.clean_pipeline import NBAStatsV3CleanPipeline

    pipeline = NBAStatsV3CleanPipeline()
    clean = pipeline.fit_transform_df(df)
    pipeline.save(clean, path / "nbastatsv3_clean.parquet")
"""

import typing as t
from pathlib import Path

import numpy as np
import pandas as pd
from kret_np_pd.UTILS_np_pd import NP_PD_Utils
from kret_sklearn.custom_transformers import PandasColumnOrderBase
from kret_sklearn.pd_pipeline import PipelinePD
from sklearn.preprocessing import FunctionTransformer

# ---------------------------------------------------------------------------
# Transformers
# ---------------------------------------------------------------------------


class StripStringColumns(PandasColumnOrderBase):
    """Strip leading/trailing whitespace from every object/string column."""

    def _fit(self, X: pd.DataFrame, y=None) -> "StripStringColumns":
        self._str_cols = X.select_dtypes(include=["str"]).columns.tolist()
        return self

    def _transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        for col in self._str_cols:
            X[col] = X[col].str.strip()
        self.new_columns = list(X.columns)
        return X


class NBAStatsV3ClockParser(PandasColumnOrderBase):
    """
    Parse the ISO 8601 ``clock`` column (e.g. 'PT04M56.00S') into two new
    numeric columns appended at the end:

    seconds_remaining  – time left in the period
    seconds_elapsed    – time played so far in the period
                         (720 s for regulation quarters, 300 s for OT)
    """

    REG_PERIOD_SECS = 720  # 12 minutes
    OT_PERIOD_SECS = 300  # 5 minutes

    def _fit(self, X: pd.DataFrame, y=None) -> "NBAStatsV3ClockParser":
        assert "clock" in X.columns, "Expected a 'clock' column in the DataFrame."
        assert "period" in X.columns, "Expected a 'period' column in the DataFrame."
        return self

    @staticmethod
    def _parse_clock(series: pd.Series) -> pd.Series:
        parsed = series.str.extract(r"PT(\d+)M([\d.]+)S").astype(float)
        return parsed[0] * 60 + parsed[1]

    def _transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        # period_len = X["period"].apply(lambda p: self.REG_PERIOD_SECS if p <= 4 else self.OT_PERIOD_SECS) # TOO SLOW
        period_len = np.where(X["period"] <= 4, self.REG_PERIOD_SECS, self.OT_PERIOD_SECS)
        X["seconds_remaining"] = self._parse_clock(X["clock"])
        X["seconds_elapsed"] = period_len - X["seconds_remaining"]
        self.new_columns = list(X.columns)
        return X


class MergeDateTime(PandasColumnOrderBase):
    """
    Load in date file, map dates to gameId, merge as new columns at the end of the DataFrame. This is a separate step because it requires an additional input file and is not strictly necessary for cleaning the raw data.
    """

    dates_df: pd.DataFrame
    filename: str | Path

    def __init__(self, filename: str | Path) -> None:
        super().__init__()
        self.filename = filename

    def _fit(self, X: pd.DataFrame, y=None):
        self.dates_df = pd.read_parquet(self.filename)
        return self

    def _transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:

        game_date_map = self.dates_df.set_index("GAME_ID")["game_date"]
        X["game_date"] = pd.to_datetime(X["gameId"].map(game_date_map))
        X["game_date_ffill"] = X["game_date"].ffill()

        self.new_columns = list(X.columns)
        return X


class SortByDate(PandasColumnOrderBase):
    """
    NOTE: this must occur after MergeDateTime for correct sorting
    """

    sort_col = ["game_date_ffill", "gameId", "actionId"]

    def _fit(self, X: pd.DataFrame, y=None) -> "SortByDate":
        assert all(
            col in X.columns for col in self.sort_col
        ), f"Expected columns {self.sort_col} not found in DataFrame."
        assert all(
            X[col].isna().sum() == 0 for col in self.sort_col
        ), f"Expected no missing values in sort columns {self.sort_col}."
        return self

    def _transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.sort_values(self.sort_col, ascending=[True, True, True]).reset_index(drop=True)
        self.new_columns = list(X.columns)
        return X


class CategoricalConversion(PandasColumnOrderBase):
    cols: list[str]

    def __init__(self, cols: list[str]) -> None:
        self.cols = cols

    def _fit(self, X: pd.DataFrame, y=None) -> "CategoricalConversion":
        for col in self.cols:
            assert col in X.columns, f"Expected column '{col}' not found in DataFrame."
        return self

    def _transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        for col in self.cols:
            X[col] = pd.Categorical(X[col])
        self.new_columns = list(X.columns)
        return X


class ConvertToInt(PandasColumnOrderBase):
    """
    NOTE: this MUST occur after sort, for ffill to work correctly.
    """

    cols: list[str]

    def __init__(self, cols: list[str]) -> None:
        self.cols = cols

    def _fit(self, X: pd.DataFrame, y=None) -> "ConvertToInt":
        for col in self.cols:
            assert col in X.columns, f"Expected column '{col}' not found in DataFrame."
        return self

    def _transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        for col in self.cols:
            X[col] = X[col].ffill().fillna(0).astype(int)
        self.new_columns = list(X.columns)
        return X


class EnforceMonotonicity(PandasColumnOrderBase):
    """
    NOTE: this must convert after ConvertToInt
    """

    cols: list[str]

    def __init__(self, cols: list[str]) -> None:
        self.cols = cols

    def _fit(self, X: pd.DataFrame, y=None) -> "EnforceMonotonicity":
        for col in self.cols:
            assert col in X.columns, f"Expected column '{col}' not found in DataFrame."
        return self

    def _transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        for col in self.cols:
            X[col] = X.groupby("gameId")[col].cummax()
        self.new_columns = list(X.columns)
        return X


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class NBAStatsV3CleanPipeline:
    """
    Wraps PipelinePD with the two cleaning steps for nbastatsv3 data.

    Methods
    -------
    fit_transform_df(df)       – fit and transform, returns cleaned DataFrame
    transform_df(df)           – transform only (after fitting)
    save(df, path)             – save cleaned DataFrame to parquet
    """

    path_root: Path

    def __init__(self, path: str | Path) -> None:
        self.path_root = Path(path)
        df_load = FunctionTransformer(func=pd.read_parquet, validate=False, kw_args={})
        cats = CategoricalConversion(
            cols=[
                "actionType",
                "playerName",
                "playerNameI",
                "teamTricode",
                "subType",
                "location",
                "shotResult",
                "teamId",
                "shotResult",
            ]
        )
        score_cols = ["scoreHome", "scoreAway", "pointsTotal"]
        start_cols = [
            "game_date_ffill",
            "gameId",
            "actionId",
            "actionType",
            "subType",
            "scoreHome",
            "scoreAway",
            "pointsTotal",
        ]
        end_cols = [
            "playerNameI",
            "actionNumber",
            "clock",
            "teamId",
            "personId",
            "playerName",
            "xLegacy",
            "yLegacy",
            "shotDistance",
            "videoAvailable",
        ]
        sort_cols = FunctionTransformer(
            func=NP_PD_Utils.move_columns, validate=False, kw_args={"start": start_cols, "end": end_cols}
        )

        self._pipeline = PipelinePD(
            steps=[
                ("load_df", df_load),
                ("strip_strings", StripStringColumns()),
                ("parse_clock", NBAStatsV3ClockParser()),
                ("merge_datetime", MergeDateTime(self.path_root / "nba_game_dates.parquet")),
                ("sort", SortByDate()),
                ("cats", cats),
                ("convert_int", ConvertToInt(score_cols)),
                ("enforce_monotonicity", EnforceMonotonicity(score_cols)),
                ("sort_cols", sort_cols),
            ]
        )

    def run(self) -> pd.DataFrame:
        path = self.path_root / "nba_statsv3.parquet"
        return self._pipeline.fit_transform_df(path)

    @staticmethod
    def save(
        df: pd.DataFrame,
        path: Path | str,
        compression: t.Literal["snappy", "gzip", "brotli", "lz4", "zstd"] = "snappy",
    ) -> Path:
        """Save cleaned DataFrame to parquet. Returns the resolved path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False, compression=compression)
        print(f"Saved {len(df):,} rows x {len(df.columns)} cols -> {path}")
        return path
