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

import pandas as pd
from kret_sklearn.custom_transformers import PandasColumnOrderBase
from kret_sklearn.pd_pipeline import PipelinePD

# ---------------------------------------------------------------------------
# Transformers
# ---------------------------------------------------------------------------


class StripStringColumns(PandasColumnOrderBase):
    """Strip leading/trailing whitespace from every object/string column."""

    def _fit(self, X: pd.DataFrame, y=None) -> "StripStringColumns":
        self._str_cols = [c for c in X.columns if X[c].dtype == object]
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
        period_len = X["period"].apply(lambda p: self.REG_PERIOD_SECS if p <= 4 else self.OT_PERIOD_SECS)
        X["seconds_remaining"] = self._parse_clock(X["clock"])
        X["seconds_elapsed"] = period_len - X["seconds_remaining"]
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

    def __init__(self) -> None:
        self._pipeline = PipelinePD(
            steps=[
                ("strip_strings", StripStringColumns()),
                ("parse_clock", NBAStatsV3ClockParser()),
            ]
        )

    def fit_transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._pipeline.fit_transform_df(df)

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._pipeline.transform_df(df)

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
