import typing as t
from pathlib import Path

import polars as pl
from kret_polars.enriched_df_pl import Enriched_DF_PL
from nba_timeout_impact.constants import NBAConstants


class RotationsDatasetPL(Enriched_DF_PL):
    # -- key columns --
    GAME_ID: pl.Series  # String (zero-padded, e.g. "0021900001")
    gameId: pl.Series  # Int64 (numeric, e.g. 21900001)
    PERSON_ID: pl.Series  # Int64
    TEAM_ID: pl.Series  # Int64
    season: pl.Series  # Int64
    season_type: pl.Series  # String

    # -- time range (tenths of seconds elapsed) --
    IN_TIME_REAL: pl.Series  # Float64
    OUT_TIME_REAL: pl.Series  # Float64

    # -- player info --
    PLAYER_FIRST: pl.Series  # String
    PLAYER_LAST: pl.Series  # String
    TEAM_CITY: pl.Series  # String
    TEAM_NAME: pl.Series  # String
    location: pl.Series  # String ("home" / "away")

    # -- stint stats --
    PLAYER_PTS: pl.Series  # String
    PT_DIFF: pl.Series  # String
    USG_PCT: pl.Series  # Float64

    target_dtypes: t.ClassVar[dict[str, pl.DataType | type[pl.DataType]]] = {
        "gameId": pl.Int64(),
        "PERSON_ID": pl.Int64(),
        "TEAM_ID": pl.Int64(),
        "season": pl.Int64(),
        "season_type": pl.String(),
        "IN_TIME_REAL": pl.Float64(),
        "OUT_TIME_REAL": pl.Float64(),
    }

    @classmethod
    def load_from_parquet(cls, path: str | Path | None = None) -> "RotationsDatasetPL":
        path = path or NBAConstants.NBA_DATA_DIR / "rotations.parquet"
        df = pl.read_parquet(path)
        ret = cls(df)
        ret.validate_data()
        return ret

    def validate_data(self):
        print("Validating rotations data (Polars)...")
        errors: list[str] = []
        warnings: list[str] = []

        # --- required columns ---
        base_attrs = set(vars(Enriched_DF_PL).keys())
        required = [k for k in self.__class__.__annotations__.keys() if k not in base_attrs]
        missing = [c for c in required if c not in self.columns]
        if missing:
            errors.append(f"Missing columns: {missing}")

        # --- no-null key columns ---
        for col in ["gameId", "PERSON_ID", "TEAM_ID", "season", "season_type", "IN_TIME_REAL", "OUT_TIME_REAL"]:
            if col in self.columns:
                n_null = self[col].null_count()
                if n_null > 0:
                    errors.append(f"Null values in key column '{col}' ({n_null:,})")

        # --- time range validity ---
        if "IN_TIME_REAL" in self.columns and "OUT_TIME_REAL" in self.columns:
            bad_range = (self["OUT_TIME_REAL"] < self["IN_TIME_REAL"]).sum()
            if bad_range > 0:
                warnings.append(f"OUT_TIME_REAL < IN_TIME_REAL in {bad_range:,} rows (zero-minute / data quirks).")

            if (self["IN_TIME_REAL"] < 0).any():
                errors.append("Negative IN_TIME_REAL values.")

        # --- value ranges ---
        if "season_type" in self.columns:
            actual_st = set(self["season_type"].unique().to_list())
            if not actual_st.issubset({"rg", "po"}):
                errors.append(f"Invalid season_type values: {actual_st - {'rg', 'po'}}")

        if "location" in self.columns:
            actual_loc = set(self["location"].unique().to_list())
            if not actual_loc.issubset({"home", "away"}):
                warnings.append(f"Unexpected location values: {actual_loc - {'home', 'away'}}")

        # --- report ---
        for w in warnings:
            print(f"  Warning: {w}")
        if errors:
            msg = "\n  ".join(errors)
            raise AssertionError(f"Validation failed:\n  {msg}")
        print(f"  Passed ({self.height:,} rows, {len(self.columns)} cols, {len(warnings)} warnings).")
