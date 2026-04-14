import typing as t
from pathlib import Path

import polars as pl
from kret_polars.enriched_df_pl import Enriched_DF_PL

from nba_timeout_impact.constants import NBAConstants


class BoxscoresDatasetPL(Enriched_DF_PL):
    # -- key columns --
    gameId: pl.Series  # Int64
    personId: pl.Series  # Int64
    teamId: pl.Series  # Int64
    season: pl.Series  # Int64
    season_type: pl.Series  # String

    # -- player info --
    firstName: pl.Series  # String
    familyName: pl.Series  # String
    nameI: pl.Series  # String
    playerSlug: pl.Series  # String
    position: pl.Series  # String
    comment: pl.Series  # String
    jerseyNum: pl.Series  # String

    # -- team info --
    teamCity: pl.Series  # String
    teamName: pl.Series  # String
    teamTricode: pl.Series  # String
    teamSlug: pl.Series  # String

    # -- stats --
    minutes: pl.Series  # String
    fieldGoalsMade: pl.Series  # Int64
    fieldGoalsAttempted: pl.Series  # Int64
    fieldGoalsPercentage: pl.Series  # Float64
    threePointersMade: pl.Series  # Int64
    threePointersAttempted: pl.Series  # Int64
    threePointersPercentage: pl.Series  # Float64
    freeThrowsMade: pl.Series  # Int64
    freeThrowsAttempted: pl.Series  # Int64
    freeThrowsPercentage: pl.Series  # Float64
    reboundsOffensive: pl.Series  # Int64
    reboundsDefensive: pl.Series  # Int64
    reboundsTotal: pl.Series  # Int64
    assists: pl.Series  # Int64
    steals: pl.Series  # Int64
    blocks: pl.Series  # Int64
    turnovers: pl.Series  # Int64
    foulsPersonal: pl.Series  # Int64
    points: pl.Series  # Int64
    plusMinusPoints: pl.Series  # Int64

    target_dtypes: t.ClassVar[dict[str, pl.DataType | type[pl.DataType]]] = {
        "gameId": pl.Int64(),
        "personId": pl.Int64(),
        "teamId": pl.Int64(),
        "season": pl.Int64(),
        "season_type": pl.String(),
        "fieldGoalsMade": pl.Int64(),
        "fieldGoalsAttempted": pl.Int64(),
        "threePointersMade": pl.Int64(),
        "threePointersAttempted": pl.Int64(),
        "freeThrowsMade": pl.Int64(),
        "freeThrowsAttempted": pl.Int64(),
        "points": pl.Int64(),
    }

    @classmethod
    def load_from_parquet(cls, path: str | Path | None = None) -> "BoxscoresDatasetPL":
        path = path or NBAConstants.NBA_DATA_DIR / "boxscores.parquet"
        df = pl.read_parquet(path)
        ret = cls(df)
        ret.validate_data()
        return ret

    def validate_data(self):
        print("Validating boxscores data (Polars)...")
        errors: list[str] = []

        # --- required columns ---
        base_attrs = set(vars(Enriched_DF_PL).keys())
        required = [k for k in self.__class__.__annotations__.keys() if k not in base_attrs]
        missing = [c for c in required if c not in self.columns]
        if missing:
            errors.append(f"Missing columns: {missing}")

        # --- no-null key columns ---
        for col in ["gameId", "personId", "teamId", "season", "season_type"]:
            if col in self.columns:
                n_null = self[col].null_count()
                if n_null > 0:
                    errors.append(f"Null values in key column '{col}' ({n_null:,})")

        # --- uniqueness: one row per player per game ---
        if "gameId" in self.columns and "personId" in self.columns:
            dupes = self.group_by(["gameId", "personId"]).len().filter(pl.col("len") > 1)
            if dupes.height > 0:
                errors.append(f"Duplicate (gameId, personId) rows: {dupes.height:,}")

        # --- value ranges ---
        if "season_type" in self.columns:
            actual_st = set(self["season_type"].unique().to_list())
            if not actual_st.issubset({"rg", "po"}):
                errors.append(f"Invalid season_type values: {actual_st - {'rg', 'po'}}")

        for col in ["points", "fieldGoalsMade", "fieldGoalsAttempted"]:
            if col in self.columns and self[col].null_count() == 0:
                if (self[col] < 0).any():
                    errors.append(f"Negative values in '{col}'.")

        # --- report ---
        if errors:
            msg = "\n  ".join(errors)
            raise AssertionError(f"Validation failed:\n  {msg}")
        print(f"  Passed ({self.height:,} rows, {len(self.columns)} cols).")
