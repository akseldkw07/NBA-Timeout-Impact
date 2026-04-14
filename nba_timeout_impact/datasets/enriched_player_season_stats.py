import typing as t
from pathlib import Path

import polars as pl
from kret_polars.enriched_df_pl import Enriched_DF_PL

from nba_timeout_impact.constants import NBAConstants


class PlayerSeasonStatsDatasetPL(Enriched_DF_PL):
    # -- key columns --
    PLAYER_ID: pl.Series  # Int64
    PLAYER_NAME: pl.Series  # String
    TEAM_ID: pl.Series  # Int64
    TEAM_ABBREVIATION: pl.Series  # String
    season_int: pl.Series  # Int64
    season_type: pl.Series  # String
    SEASON_ID: pl.Series  # String
    LEAGUE_ID: pl.Series  # String

    # -- player info --
    PLAYER_AGE: pl.Series  # Float64
    player_name_ascii: pl.Series  # String

    # -- traditional stats --
    GP: pl.Series  # Int64
    GS: pl.Series  # Int64
    MIN: pl.Series  # Int64
    FGM: pl.Series  # Int64
    FGA: pl.Series  # Int64
    FG_PCT: pl.Series  # Float64
    FG3M: pl.Series  # Int64
    FG3A: pl.Series  # Int64
    FG3_PCT: pl.Series  # Float64
    FTM: pl.Series  # Int64
    FTA: pl.Series  # Int64
    FT_PCT: pl.Series  # Float64
    OREB: pl.Series  # Int64
    DREB: pl.Series  # Int64
    REB: pl.Series  # Int64
    AST: pl.Series  # Int64
    STL: pl.Series  # Int64
    BLK: pl.Series  # Int64
    TOV: pl.Series  # Int64
    PF: pl.Series  # Int64
    PTS: pl.Series  # Int64

    target_dtypes: t.ClassVar[dict[str, pl.DataType | type[pl.DataType]]] = {
        "PLAYER_ID": pl.Int64(),
        "TEAM_ID": pl.Int64(),
        "season_int": pl.Int64(),
        "season_type": pl.String(),
        "GP": pl.Int64(),
        "GS": pl.Int64(),
        "MIN": pl.Int64(),
        "PTS": pl.Int64(),
    }

    @classmethod
    def load_from_parquet(cls, path: str | Path | None = None) -> "PlayerSeasonStatsDatasetPL":
        path = path or NBAConstants.NBA_DATA_DIR / "player_season_stats.parquet"
        df = pl.read_parquet(path)
        ret = cls(df)
        ret.validate_data()
        return ret

    def validate_data(self):
        print("Validating player_season_stats data (Polars)...")
        errors: list[str] = []

        # --- required columns ---
        base_attrs = set(vars(Enriched_DF_PL).keys())
        required = [k for k in self.__class__.__annotations__.keys() if k not in base_attrs]
        missing = [c for c in required if c not in self.columns]
        if missing:
            errors.append(f"Missing columns: {missing}")

        # --- no-null key columns ---
        for col in ["PLAYER_ID", "TEAM_ID", "season_int", "season_type"]:
            if col in self.columns:
                n_null = self[col].null_count()
                if n_null > 0:
                    errors.append(f"Null values in key column '{col}' ({n_null:,})")

        # --- uniqueness: one row per player per season per season_type per team ---
        # Players traded mid-season have per-team rows plus a "TOT" total row.
        if all(c in self.columns for c in ["PLAYER_ID", "season_int", "season_type", "TEAM_ABBREVIATION"]):
            dupes = (
                self.group_by(["PLAYER_ID", "season_int", "season_type", "TEAM_ABBREVIATION"])
                .len()
                .filter(pl.col("len") > 1)
            )
            if dupes.height > 0:
                errors.append(
                    f"Duplicate (PLAYER_ID, season_int, season_type, TEAM_ABBREVIATION) rows: {dupes.height:,}"
                )

        # --- value ranges ---
        if "season_type" in self.columns:
            actual_st = set(self["season_type"].unique().to_list())
            if not actual_st.issubset({"rg", "po"}):
                errors.append(f"Invalid season_type values: {actual_st - {'rg', 'po'}}")

        if "season_int" in self.columns:
            valid_seasons = set(range(2015, 2030))
            actual = set(self["season_int"].unique().to_list())
            bad = actual - valid_seasons
            if bad:
                errors.append(f"Invalid season_int values: {bad}")

        for col in ["GP", "PTS", "MIN"]:
            if col in self.columns and self[col].null_count() == 0:
                if (self[col] < 0).any():
                    errors.append(f"Negative values in '{col}'.")

        # --- report ---
        if errors:
            msg = "\n  ".join(errors)
            raise AssertionError(f"Validation failed:\n  {msg}")
        print(f"  Passed ({self.height:,} rows, {len(self.columns)} cols).")
