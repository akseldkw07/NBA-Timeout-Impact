import typing as t
from pathlib import Path

import polars as pl
from kret_polars.enriched_df_pl import Enriched_DF_PL

from nba_timeout_impact.constants import NBAConstants


class StintsDatasetPL(Enriched_DF_PL):
    # -- key columns --
    gameId: pl.Series  # Int64
    personId: pl.Series  # Int64
    teamId: pl.Series  # Int64
    season: pl.Series  # Int64
    season_type: pl.Series  # String
    stint_id: pl.Series  # UInt32

    # -- time range (seconds elapsed) --
    in_game_seconds: pl.Series  # Float64
    out_game_seconds: pl.Series  # Float64
    stint_duration_minutes: pl.Series  # Float64

    # -- player info --
    playerFirst: pl.Series  # String
    playerLast: pl.Series  # String
    location: pl.Series  # String ("home" / "away")

    # -- stint stats --
    player_pts: pl.Series  # Float64
    pt_diff: pl.Series  # Float64

    target_dtypes: t.ClassVar[dict[str, pl.DataType | type[pl.DataType]]] = {
        "gameId": pl.Int64(),
        "personId": pl.Int64(),
        "teamId": pl.Int64(),
        "season": pl.Int64(),
        "season_type": pl.String(),
        "stint_id": pl.UInt32(),
        "in_game_seconds": pl.Float64(),
        "out_game_seconds": pl.Float64(),
        "stint_duration_minutes": pl.Float64(),
    }

    @classmethod
    def load_from_parquet(cls, path: str | Path | None = None) -> "StintsDatasetPL":
        path = path or NBAConstants.NBA_DATA_DIR / "stints.parquet"
        df = pl.read_parquet(path)
        ret = cls(df)
        ret.validate_data()
        return ret

    def validate_data(self):
        print("Validating stints data (Polars)...")
        errors: list[str] = []
        warnings: list[str] = []

        # --- required columns ---
        base_attrs = set(vars(Enriched_DF_PL).keys())
        required = [k for k in self.__class__.__annotations__.keys() if k not in base_attrs]
        missing = [c for c in required if c not in self.columns]
        if missing:
            errors.append(f"Missing columns: {missing}")

        # --- no-null key columns ---
        for col in [
            "gameId",
            "personId",
            "teamId",
            "season",
            "season_type",
            "stint_id",
            "in_game_seconds",
            "out_game_seconds",
        ]:
            if col in self.columns:
                n_null = self[col].null_count()
                if n_null > 0:
                    errors.append(f"Null values in key column '{col}' ({n_null:,})")

        # --- time range validity ---
        if "in_game_seconds" in self.columns and "out_game_seconds" in self.columns:
            bad_range = (self["out_game_seconds"] < self["in_game_seconds"]).sum()
            if bad_range > 0:
                warnings.append(
                    f"out_game_seconds < in_game_seconds in {bad_range:,} rows (zero-minute / data quirks)."
                )

            if (self["in_game_seconds"] < 0).any():
                errors.append("Negative in_game_seconds values.")

        # --- stint_duration consistency ---
        if all(c in self.columns for c in ["in_game_seconds", "out_game_seconds", "stint_duration_minutes"]):
            expected = (self["out_game_seconds"] - self["in_game_seconds"]) / 60.0
            diff = (self["stint_duration_minutes"] - expected).abs()
            bad = (diff.drop_nulls() > 0.01).sum()
            if bad > 0:
                warnings.append(f"stint_duration_minutes inconsistent with in/out times: {bad:,} rows")

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
