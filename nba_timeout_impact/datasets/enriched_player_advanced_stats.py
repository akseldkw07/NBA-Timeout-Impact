import typing as t
from pathlib import Path

import polars as pl
from kret_polars.enriched_df_pl import Enriched_DF_PL

from nba_timeout_impact.constants import NBAConstants


class PlayerAdvancedStatsDatasetPL(Enriched_DF_PL):
    # -- key columns --
    PLAYER_ID: pl.Series  # Int64
    PLAYER_NAME: pl.Series  # String
    TEAM_ID: pl.Series  # Int64
    TEAM_ABBREVIATION: pl.Series  # String
    season_int: pl.Series  # Int64

    # -- demographics --
    AGE: pl.Series  # Float64
    PLAYER_HEIGHT: pl.Series  # String
    PLAYER_HEIGHT_INCHES: pl.Series  # Int64
    PLAYER_WEIGHT: pl.Series  # String
    COLLEGE: pl.Series  # String
    COUNTRY: pl.Series  # String
    DRAFT_YEAR: pl.Series  # String
    DRAFT_ROUND: pl.Series  # String
    DRAFT_NUMBER: pl.Series  # String

    # -- traditional stats --
    GP: pl.Series  # Int64
    PTS: pl.Series  # Float64
    REB: pl.Series  # Float64
    AST: pl.Series  # Float64
    NET_RATING: pl.Series  # Float64
    OREB_PCT: pl.Series  # Float64
    DREB_PCT: pl.Series  # Float64
    USG_PCT: pl.Series  # Float64
    TS_PCT: pl.Series  # Float64
    AST_PCT: pl.Series  # Float64

    # -- estimated stats --
    PLAYER_NAME_em: pl.Series  # String
    E_OFF_RATING: pl.Series  # Float64
    E_DEF_RATING: pl.Series  # Float64
    E_NET_RATING: pl.Series  # Float64
    E_AST_RATIO: pl.Series  # Float64
    E_OREB_PCT: pl.Series  # Float64
    E_DREB_PCT: pl.Series  # Float64
    E_REB_PCT: pl.Series  # Float64
    E_TOV_PCT: pl.Series  # Float64
    E_USG_PCT: pl.Series  # Float64
    E_PACE: pl.Series  # Float64

    # -- hustle stats --
    PLAYER_NAME_hustle: pl.Series  # String
    CONTESTED_SHOTS: pl.Series  # Float64
    CONTESTED_SHOTS_2PT: pl.Series  # Float64
    CONTESTED_SHOTS_3PT: pl.Series  # Float64
    DEFLECTIONS: pl.Series  # Float64
    CHARGES_DRAWN: pl.Series  # Float64
    SCREEN_ASSISTS: pl.Series  # Float64
    SCREEN_AST_PTS: pl.Series  # Float64
    OFF_LOOSE_BALLS_RECOVERED: pl.Series  # Float64
    DEF_LOOSE_BALLS_RECOVERED: pl.Series  # Float64
    LOOSE_BALLS_RECOVERED: pl.Series  # Float64
    PCT_LOOSE_BALLS_RECOVERED_OFF: pl.Series  # Float64
    PCT_LOOSE_BALLS_RECOVERED_DEF: pl.Series  # Float64
    OFF_BOXOUTS: pl.Series  # Float64
    DEF_BOXOUTS: pl.Series  # Float64
    BOX_OUTS: pl.Series  # Float64
    BOX_OUT_PLAYER_TEAM_REBS: pl.Series  # Float64
    BOX_OUT_PLAYER_REBS: pl.Series  # Float64
    PCT_BOX_OUTS_OFF: pl.Series  # Float64
    PCT_BOX_OUTS_DEF: pl.Series  # Float64
    PCT_BOX_OUTS_TEAM_REB: pl.Series  # Float64
    PCT_BOX_OUTS_REB: pl.Series  # Float64

    # -- drive stats --
    PLAYER_NAME_drives: pl.Series  # String
    DRIVES: pl.Series  # Float64
    DRIVE_FGM: pl.Series  # Float64
    DRIVE_FGA: pl.Series  # Float64
    DRIVE_FG_PCT: pl.Series  # Float64
    DRIVE_FTM: pl.Series  # Float64
    DRIVE_FTA: pl.Series  # Float64
    DRIVE_FT_PCT: pl.Series  # Float64
    DRIVE_PTS: pl.Series  # Float64
    DRIVE_PTS_PCT: pl.Series  # Float64
    DRIVE_PASSES: pl.Series  # Float64
    DRIVE_PASSES_PCT: pl.Series  # Float64
    DRIVE_AST: pl.Series  # Float64
    DRIVE_AST_PCT: pl.Series  # Float64
    DRIVE_TOV: pl.Series  # Float64
    DRIVE_TOV_PCT: pl.Series  # Float64
    DRIVE_PF: pl.Series  # Float64
    DRIVE_PF_PCT: pl.Series  # Float64

    # -- catch and shoot --
    PLAYER_NAME_catchshoot: pl.Series  # String
    CATCH_SHOOT_FGM: pl.Series  # Float64
    CATCH_SHOOT_FGA: pl.Series  # Float64
    CATCH_SHOOT_FG_PCT: pl.Series  # Float64
    CATCH_SHOOT_PTS: pl.Series  # Float64
    CATCH_SHOOT_FG3M: pl.Series  # Float64
    CATCH_SHOOT_FG3A: pl.Series  # Float64
    CATCH_SHOOT_FG3_PCT: pl.Series  # Float64
    CATCH_SHOOT_EFG_PCT: pl.Series  # Float64

    # -- pull up shooting --
    PLAYER_NAME_pullupshot: pl.Series  # String
    PULL_UP_FGM: pl.Series  # Float64
    PULL_UP_FGA: pl.Series  # Float64
    PULL_UP_FG_PCT: pl.Series  # Float64
    PULL_UP_PTS: pl.Series  # Float64
    PULL_UP_FG3M: pl.Series  # Float64
    PULL_UP_FG3A: pl.Series  # Float64
    PULL_UP_FG3_PCT: pl.Series  # Float64
    PULL_UP_EFG_PCT: pl.Series  # Float64

    # -- passing --
    PLAYER_NAME_passing: pl.Series  # String
    PASSES_MADE: pl.Series  # Float64
    PASSES_RECEIVED: pl.Series  # Float64
    FT_AST: pl.Series  # Float64
    SECONDARY_AST: pl.Series  # Float64
    POTENTIAL_AST: pl.Series  # Float64
    AST_POINTS_CREATED: pl.Series  # Float64
    AST_ADJ: pl.Series  # Float64
    AST_TO_PASS_PCT: pl.Series  # Float64
    AST_TO_PASS_PCT_ADJ: pl.Series  # Float64

    # -- defense --
    PLAYER_NAME_defense: pl.Series  # String
    STL: pl.Series  # Float64
    BLK: pl.Series  # Float64
    DREB: pl.Series  # Float64
    DEF_RIM_FGM: pl.Series  # Float64
    DEF_RIM_FGA: pl.Series  # Float64
    DEF_RIM_FG_PCT: pl.Series  # Float64

    # -- speed and distance --
    PLAYER_NAME_speeddistance: pl.Series  # String
    MIN1: pl.Series  # Float64
    DIST_FEET: pl.Series  # Float64
    DIST_MILES: pl.Series  # Float64
    DIST_MILES_OFF: pl.Series  # Float64
    DIST_MILES_DEF: pl.Series  # Float64
    AVG_SPEED: pl.Series  # Float64
    AVG_SPEED_OFF: pl.Series  # Float64
    AVG_SPEED_DEF: pl.Series  # Float64

    target_dtypes: t.ClassVar[dict[str, pl.DataType | type[pl.DataType]]] = {
        "PLAYER_ID": pl.Int64(),
        "TEAM_ID": pl.Int64(),
        "season_int": pl.Int64(),
        "GP": pl.Int64(),
        "PLAYER_HEIGHT_INCHES": pl.Int64(),
        "AGE": pl.Float64(),
        "NET_RATING": pl.Float64(),
        "USG_PCT": pl.Float64(),
        "TS_PCT": pl.Float64(),
    }

    @classmethod
    def load_from_parquet(cls, path: str | Path | None = None) -> "PlayerAdvancedStatsDatasetPL":
        path = path or NBAConstants.NBA_DATA_DIR / "player_advanced_stats.parquet"
        df = pl.read_parquet(path)
        ret = cls(df)
        ret.validate_data()
        return ret

    def validate_data(self):
        print("Validating player_advanced_stats data (Polars)...")
        errors: list[str] = []

        # --- required columns ---
        base_attrs = set(vars(Enriched_DF_PL).keys())
        required = [k for k in self.__class__.__annotations__.keys() if k not in base_attrs]
        missing = [c for c in required if c not in self.columns]
        if missing:
            errors.append(f"Missing columns: {missing}")

        # --- no-null key columns ---
        for col in ["PLAYER_ID", "TEAM_ID", "season_int"]:
            if col in self.columns:
                n_null = self[col].null_count()
                if n_null > 0:
                    errors.append(f"Null values in key column '{col}' ({n_null:,})")

        # --- uniqueness: one row per player per season ---
        if "PLAYER_ID" in self.columns and "season_int" in self.columns:
            dupes = self.group_by(["PLAYER_ID", "season_int"]).len().filter(pl.col("len") > 1)
            if dupes.height > 0:
                errors.append(f"Duplicate (PLAYER_ID, season_int) rows: {dupes.height:,}")

        # --- season range ---
        if "season_int" in self.columns:
            valid_seasons = set(range(2015, 2030))
            actual = set(self["season_int"].unique().to_list())
            bad = actual - valid_seasons
            if bad:
                errors.append(f"Invalid season_int values: {bad}")

        # --- report ---
        if errors:
            msg = "\n  ".join(errors)
            raise AssertionError(f"Validation failed:\n  {msg}")
        print(f"  Passed ({self.height:,} rows, {len(self.columns)} cols).")
