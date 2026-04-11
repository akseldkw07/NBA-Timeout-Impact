import typing as t
from pathlib import Path

import polars as pl
from kret_polars.enriched_df_pl import Enriched_DF_PL

from nba_timeout_impact.constants import NBAConstants


class CDNNBADatasetPL(Enriched_DF_PL):
    # -- key columns (sorted to front by pipeline) --
    game_date: pl.Series  # Datetime(ms)
    gameId: pl.Series  # Int64
    orderNumber: pl.Series  # Int64
    actionType: pl.Series  # Categorical
    subType: pl.Series  # Categorical
    description: pl.Series  # String
    scoreHome: pl.Series  # Int64
    scoreAway: pl.Series  # Int64
    pointsTotal: pl.Series  # Int64
    possession: pl.Series  # Int64
    period: pl.Series  # Int64
    game_seconds_elapsed: pl.Series  # Float64
    seconds_remaining: pl.Series  # Float64
    seconds_elapsed: pl.Series  # Float64
    IsPlayoff: pl.Series  # Boolean

    # -- context columns --
    periodType: pl.Series  # Categorical
    qualifiers: pl.Series  # String
    edited: pl.Series  # Datetime(us, UTC)
    isFieldGoal: pl.Series  # Int64
    side: pl.Series  # Categorical
    personIdsFilter: pl.Series  # String
    teamTricode: pl.Series  # Categorical
    descriptor: pl.Series  # String
    jumpBallRecoveredName: pl.Series  # String
    jumpBallWonPlayerName: pl.Series  # String
    jumpBallWonPersonId: pl.Series  # Int64
    jumpBallLostPlayerName: pl.Series  # String
    jumpBallLostPersonId: pl.Series  # Int64
    officialId: pl.Series  # Int64
    turnoverTotal: pl.Series  # Int64
    foulPersonalTotal: pl.Series  # Int64
    foulTechnicalTotal: pl.Series  # Int64
    foulDrawnPlayerName: pl.Series  # String
    foulDrawnPersonId: pl.Series  # Int64
    shotResult: pl.Series  # Categorical
    assistPlayerNameInitial: pl.Series  # String
    assistPersonId: pl.Series  # Int64
    assistTotal: pl.Series  # Int64
    shotActionNumber: pl.Series  # Int64
    reboundTotal: pl.Series  # Int64
    reboundDefensiveTotal: pl.Series  # Int64
    reboundOffensiveTotal: pl.Series  # Int64
    blockPlayerName: pl.Series  # String
    blockPersonId: pl.Series  # Int64
    jumpBallRecoverdPersonId: pl.Series  # Int64
    stealPlayerName: pl.Series  # String
    stealPersonId: pl.Series  # Int64
    area: pl.Series  # Categorical
    areaDetail: pl.Series  # Categorical
    isTargetScoreLastPeriod: pl.Series  # Boolean

    # -- player --
    playerNameI: pl.Series  # Categorical
    actionNumber: pl.Series  # Int64
    clock: pl.Series  # String
    timeActual: pl.Series  # Datetime(us, UTC)
    teamId: pl.Series  # Int64
    personId: pl.Series  # Int64
    playerName: pl.Series  # Categorical
    xLegacy: pl.Series  # Int64
    yLegacy: pl.Series  # Int64
    x: pl.Series  # Float64
    y: pl.Series  # Float64
    shotDistance: pl.Series  # Float64

    # -- enriched columns --
    season_type: pl.Series  # String
    season: pl.Series  # Int64
    shot_value: pl.Series  # Int64
    points_scored: pl.Series  # Int64
    score_margin: pl.Series  # Int64
    is_clutch: pl.Series  # Boolean
    prev_action_type: pl.Series  # Categorical
    x_court: pl.Series  # Int32
    y_court: pl.Series  # Int32
    possession_id: pl.Series  # Int64
    possession_points: pl.Series  # Int64
    possession_outcome: pl.Series  # String

    target_dtypes: t.ClassVar[dict[str, pl.DataType | type[pl.DataType]]] = {
        "game_date": pl.Datetime("ms"),
        "gameId": pl.Int64(),
        "orderNumber": pl.Int64(),
        "actionType": pl.Categorical(),
        "subType": pl.Categorical(),
        "description": pl.String(),
        "scoreHome": pl.Int64(),
        "scoreAway": pl.Int64(),
        "possession": pl.Int64(),
        "period": pl.Int64(),
        "game_seconds_elapsed": pl.Float64(),
        "seconds_remaining": pl.Float64(),
        "seconds_elapsed": pl.Float64(),
        "IsPlayoff": pl.Boolean(),
        "personId": pl.Int64(),
        "season": pl.Int64(),
        "season_type": pl.String(),
        "points_scored": pl.Int64(),
        "score_margin": pl.Int64(),
        "is_clutch": pl.Boolean(),
    }

    @classmethod
    def load_from_parquet(cls, path: str | Path | None = None) -> "CDNNBADatasetPL":
        path = path or NBAConstants.NBA_DATA_DIR / "cdnnba_enriched.parquet"
        df = pl.read_parquet(path)
        df = cls._inject_tv_timeouts(df)
        ret = cls(df)
        ret.validate_data()
        return ret

    # ------------------------------------------------------------------ #
    #  Inferred TV / official timeouts                                     #
    # ------------------------------------------------------------------ #
    #
    # cdnnba only logs coach-called timeouts ("full", "challenge").
    # NBA rules mandate ~2 TV timeouts per quarter at first dead ball
    # under 6:59 and 2:59 remaining.  We detect these by finding moments
    # where real-world time (timeActual) advances >> game clock, with no
    # logged timeout.  Injected rows have actionType="timeout",
    # subType="official_inferred".

    _TV_TIMEOUT_MIN_EXCESS_SEC = 90.0

    @staticmethod
    def _infer_tv_timeouts(df: pl.DataFrame) -> pl.DataFrame:
        """Detect TV/official timeouts from real-time vs game-clock gaps.

        Returns a DataFrame of new rows (one per inferred timeout) with
        the same schema as *df*, ready to be concatenated.
        """
        spine = df.select(
            "gameId",
            "game_date",
            "period",
            "game_seconds_elapsed",
            "seconds_remaining",
            "seconds_elapsed",
            "timeActual",
            "actionType",
            "orderNumber",
            "scoreHome",
            "scoreAway",
            "possession",
            "IsPlayoff",
            "periodType",
            "season",
            "season_type",
            "personId",
            "points_scored",
            "score_margin",
            "is_clutch",
            "possession_id",
        ).sort("gameId", "orderNumber")

        # Assign moment_id: contiguous events at the same game clock
        spine = spine.with_columns(
            (
                (pl.col("game_seconds_elapsed") != pl.col("game_seconds_elapsed").shift(1))
                | (pl.col("gameId") != pl.col("gameId").shift(1))
            )
            .cum_sum()
            .alias("moment_id")
        )

        # Aggregate each moment
        moments = (
            spine.group_by("moment_id")
            .agg(
                pl.col("gameId").first(),
                pl.col("game_date").first(),
                pl.col("period").first(),
                pl.col("game_seconds_elapsed").first().alias("gse"),
                pl.col("seconds_remaining").first(),
                pl.col("seconds_elapsed").first(),
                pl.col("timeActual").min().alias("time_start"),
                pl.col("timeActual").max().alias("time_end"),
                pl.col("orderNumber").first().alias("order_before"),
                pl.col("possession_id").last().alias("possession_id"),
                pl.col("scoreHome").first(),
                pl.col("scoreAway").first(),
                pl.col("possession").first(),
                pl.col("IsPlayoff").first(),
                pl.col("periodType").first(),
                pl.col("season").first(),
                pl.col("season_type").first(),
                pl.col("score_margin").first(),
                pl.col("is_clutch").first(),
                (pl.col("actionType") == "timeout").any().alias("has_timeout"),
                (pl.col("actionType") == "period").any().alias("has_period"),
            )
            .sort("moment_id")
        )

        # Compute real-time excess between consecutive moments
        game_boundary = moments["gameId"] != moments["gameId"].shift(1)
        gap_real = (
            pl.when(~game_boundary)
            .then((moments["time_start"] - moments["time_end"].shift(1)).dt.total_seconds())
            .otherwise(None)
        )
        gap_game = pl.when(~game_boundary).then(moments["gse"] - moments["gse"].shift(1)).otherwise(None)
        moments = moments.with_columns((gap_real - gap_game).alias("excess"))

        # Filter: large excess, no logged timeout, not a period boundary
        candidates = moments.filter(
            (pl.col("excess") >= CDNNBADatasetPL._TV_TIMEOUT_MIN_EXCESS_SEC)
            & (~pl.col("has_timeout"))
            & (~pl.col("has_period"))
            & (pl.col("excess").is_not_null())
        )

        if candidates.height == 0:
            return pl.DataFrame(schema=df.schema)

        # Build new rows matching df's schema.
        # orderNumber: place just before the moment (order_before - 1)
        new_rows = pl.DataFrame(
            {
                "gameId": candidates["gameId"],
                "game_date": candidates["game_date"],
                "period": candidates["period"],
                "game_seconds_elapsed": candidates["gse"],
                "seconds_remaining": candidates["seconds_remaining"],
                "seconds_elapsed": candidates["seconds_elapsed"],
                "orderNumber": candidates["order_before"] - 1,
                "actionType": pl.Series(["timeout"] * candidates.height, dtype=pl.String),
                "subType": pl.Series(["official_inferred"] * candidates.height, dtype=pl.String),
                "description": pl.Series(["Inferred TV/Official Timeout"] * candidates.height, dtype=pl.String),
                "scoreHome": candidates["scoreHome"],
                "scoreAway": candidates["scoreAway"],
                "possession": candidates["possession"],
                "IsPlayoff": candidates["IsPlayoff"],
                "periodType": candidates["periodType"],
                "season": candidates["season"],
                "season_type": candidates["season_type"],
                "personId": pl.Series([0] * candidates.height, dtype=pl.Int64),
                "points_scored": pl.Series([0] * candidates.height, dtype=pl.Int64),
                "score_margin": candidates["score_margin"],
                "is_clutch": candidates["is_clutch"],
                "possession_id": candidates["possession_id"],
            }
        )

        # Fill remaining columns with nulls to match schema
        for col_name, col_dtype in df.schema.items():
            if col_name not in new_rows.columns:
                new_rows = new_rows.with_columns(pl.lit(None).cast(col_dtype).alias(col_name))

        # Reorder to match original schema
        return new_rows.select(df.columns)

    @staticmethod
    def _inject_tv_timeouts(df: pl.DataFrame) -> pl.DataFrame:
        """Infer TV timeouts and insert them into the DataFrame, maintaining sort order."""
        new_rows = CDNNBADatasetPL._infer_tv_timeouts(df)
        if new_rows.height == 0:
            return df

        # Cast categorical columns in new_rows to match df
        for col_name in df.columns:
            if df.schema[col_name] == pl.Categorical and new_rows.schema[col_name] != pl.Categorical:
                new_rows = new_rows.with_columns(pl.col(col_name).cast(pl.Categorical))

        combined = pl.concat([df, new_rows], how="diagonal_relaxed")
        combined = combined.sort("game_date", "gameId", "orderNumber")

        n_new = new_rows.height
        n_games = new_rows["gameId"].n_unique()
        print(
            f"  Injected {n_new:,} inferred TV timeouts across {n_games:,} games "
            f"({n_new / max(n_games, 1):.1f}/game)"
        )
        return combined

    _VALID_OUTCOMES = {
        "made_2pt",
        "made_3pt",
        "made_2pt_and1",
        "made_3pt_and1",
        "miss",
        "miss_def_reb",
        "turnover_live",
        "turnover_dead",
        "end_of_period",
        "violation",
        "other",
    }

    def validate_data(self):
        print("Validating cdnnba data (Polars)...")
        errors: list[str] = []
        warnings: list[str] = []

        # --- required columns ---
        base_attrs = set(vars(Enriched_DF_PL).keys())
        required = [k for k in self.__class__.__annotations__.keys() if k not in base_attrs]
        missing = [c for c in required if c not in self.columns]
        if missing:
            errors.append(f"Missing columns: {missing}")

        # --- sorting ---
        if not self["game_date"].is_sorted():
            errors.append("game_date is not sorted.")

        gid = self["gameId"]
        # rle length == n_unique means each gameId appears in one contiguous block
        n_runs = gid.ne(gid.shift(1)).fill_null(True).sum()
        if n_runs != gid.n_unique():
            errors.append("gameId is not contiguously grouped (games interleaved).")

        # --- no-null columns ---
        no_null_cols = [
            "gameId",
            "orderNumber",
            "period",
            "scoreHome",
            "scoreAway",
            "game_seconds_elapsed",
            "seconds_remaining",
            "seconds_elapsed",
            "game_date",
            "actionType",
            "personId",
            "possession",
            "IsPlayoff",
            "season",
            "season_type",
            "points_scored",
            "score_margin",
            "is_clutch",
        ]
        for col in no_null_cols:
            if col in self.columns:
                n_null = self[col].null_count()
                if n_null > 0:
                    errors.append(f"Null values in non-nullable column '{col}' ({n_null:,})")

        # --- monotonicity within games ---
        for col in ["orderNumber", "scoreHome", "scoreAway"]:
            if col not in self.columns:
                continue
            check = (
                self.group_by("gameId", maintain_order=True)
                .agg((pl.col(col).diff().drop_nulls() < 0).any().alias(col))
                .filter(pl.col(col))
            )
            if check.height > 0:
                bad = check["gameId"].head(5).to_list()
                errors.append(f"{col} not non-decreasing within gameId, violations: {bad}")

        # game_seconds_elapsed — warn only (known data quirk)
        if "game_seconds_elapsed" in self.columns:
            check = (
                self.group_by("gameId", maintain_order=True)
                .agg((pl.col("game_seconds_elapsed").diff().drop_nulls() < 0).any().alias("game_seconds_elapsed"))
                .filter(pl.col("game_seconds_elapsed"))
            )
            if check.height > 0:
                warnings.append(
                    f"game_seconds_elapsed has minor reversals in {check.height} games "
                    "(instant replay / memo events — not a pipeline bug)."
                )

        # --- value ranges ---
        if "period" in self.columns:
            pmin, pmax = int(self["period"].min()), int(self["period"].max())  # type: ignore[arg-type]
            if pmin < 1 or pmax > 10:
                errors.append(f"period out of range [1, 10]: min={pmin}, max={pmax}")

        if "seconds_remaining" in self.columns:
            if (self["seconds_remaining"] < 0).any():
                errors.append("seconds_remaining has negative values.")

        if "game_seconds_elapsed" in self.columns:
            if (self["game_seconds_elapsed"] < 0).any():
                errors.append("game_seconds_elapsed has negative values.")

        if "scoreHome" in self.columns and "scoreAway" in self.columns:
            if (self["scoreHome"] < 0).any() or (self["scoreAway"] < 0).any():
                errors.append("Negative scores found.")

        # --- enriched column checks ---
        if "season" in self.columns:
            valid_seasons = set(range(2020, 2030))
            actual = set(self["season"].unique().to_list())
            bad_seasons = actual - valid_seasons
            if bad_seasons:
                errors.append(f"Invalid season values: {bad_seasons}")

        if "season_type" in self.columns:
            actual_st = set(self["season_type"].unique().to_list())
            if not actual_st.issubset({"rg", "po"}):
                errors.append(f"Invalid season_type values: {actual_st - {'rg', 'po'}}")

        if "IsPlayoff" in self.columns and "season_type" in self.columns:
            mismatch = (
                (self["IsPlayoff"] & (self["season_type"] != "po"))
                | (~self["IsPlayoff"] & (self["season_type"] != "rg"))
            ).sum()
            if mismatch > 0:
                errors.append(f"IsPlayoff/season_type mismatch: {mismatch:,} rows")

        if "shot_value" in self.columns:
            actual_sv = set(self["shot_value"].drop_nulls().unique().to_list())
            if not actual_sv.issubset({1, 2, 3}):
                errors.append(f"Invalid shot_value values: {actual_sv - {1, 2, 3}}")

        if "points_scored" in self.columns:
            actual_ps = set(self["points_scored"].unique().to_list())
            if not actual_ps.issubset({0, 1, 2, 3}):
                errors.append(f"Invalid points_scored values: {actual_ps - {0, 1, 2, 3}}")

        if "score_margin" in self.columns:
            expected_margin = self["scoreHome"] - self["scoreAway"]
            if not (self["score_margin"] == expected_margin).all():
                errors.append("score_margin != scoreHome - scoreAway")

        if "is_clutch" in self.columns:
            expected_clutch = (self["score_margin"].abs() <= 5) & (self["game_seconds_elapsed"] >= 2400.0)
            if not (self["is_clutch"] == expected_clutch).all():
                errors.append("is_clutch does not match definition (|margin| <= 5 AND elapsed >= 2400).")

        if "possession_outcome" in self.columns:
            outcomes = set(self["possession_outcome"].drop_nulls().unique().to_list())
            fixed_outcomes = {o for o in outcomes if not o.startswith("ft_")}
            unexpected = fixed_outcomes - self._VALID_OUTCOMES
            if unexpected:
                warnings.append(f"Unexpected possession_outcome values: {unexpected}")

        if "possession_id" in self.columns and "possession_points" in self.columns:
            has_pid = self["possession_id"].is_not_null()
            has_ppts = self["possession_points"].is_not_null()
            mismatch_n = (has_pid != has_ppts).sum()
            if mismatch_n > 0:
                warnings.append(f"possession_id/possession_points nullity mismatch: {mismatch_n:,} rows")

        # --- report ---
        for w in warnings:
            print(f"  Warning: {w}")
        if errors:
            msg = "\n  ".join(errors)
            raise AssertionError(f"Validation failed:\n  {msg}")
        print(f"  Passed ({self.height:,} rows, {len(self.columns)} cols, {len(warnings)} warnings).")
