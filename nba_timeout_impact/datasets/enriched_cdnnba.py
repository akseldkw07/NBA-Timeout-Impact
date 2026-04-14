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
    # NBA rules mandate TV timeouts at specific game-clock marks in each
    # applicable period.  We provide two inference methods:
    #
    # 1. Heuristic (original): find large real-time gaps in timeActual
    #    with no logged timeout. Requires wall-clock timestamps.
    # 2. Rulebook: walk through period events and inject a mandatory
    #    timeout at the first dead ball past each rulebook threshold,
    #    skipping if a coach TO already occurred in that window.
    #    Validated against 2013-2016 nbastatsv3 ground truth with
    #    Rule v7: thresholds [540, 360, 180] in Q2/Q4 → F1 ~0.88-0.92.
    #
    # Injected rows have actionType="timeout", subType="official_inferred".

    _TV_TIMEOUT_MIN_EXCESS_SEC = 90.0

    # Rulebook params tuned for the cdnnba era (post-2017 NBA rules).
    # Applied to ALL four quarters with 2 mandatory marks: 7:00 and 3:00 remaining.
    _RULEBOOK_THRESHOLDS_POST2017 = [420, 180]
    _RULEBOOK_PERIODS_POST2017 = [1, 2, 3, 4]
    _CDNNBA_COACH_TO_SUBTYPES = ["full", "challenge"]
    _CDNNBA_DEAD_BALL_ACTION_TYPES = {
        "foul",
        "freethrow",
        "substitution",
        "turnover",
        "violation",
        "jumpball",
        "stoppage",
        "timeout",
        "ejection",
    }

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
        """Infer TV timeouts and insert them into the DataFrame, maintaining sort order.

        Uses the rulebook-based method by default. Set
        ``CDNNBADatasetPL._USE_RULEBOOK_INJECTION = False`` to fall back
        to the real-time excess heuristic.
        """
        if CDNNBADatasetPL._USE_RULEBOOK_INJECTION:
            new_rows = CDNNBADatasetPL._infer_tv_timeouts_rulebook(df)
        else:
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
        method = "rulebook" if CDNNBADatasetPL._USE_RULEBOOK_INJECTION else "heuristic"
        print(
            f"  Injected {n_new:,} inferred TV timeouts ({method}) across {n_games:,} games "
            f"({n_new / max(n_games, 1):.1f}/game)"
        )
        return combined

    _USE_RULEBOOK_INJECTION = True  # set False for real-time excess heuristic

    @staticmethod
    def _infer_tv_timeouts_rulebook(df: pl.DataFrame) -> pl.DataFrame:
        """Rulebook-based TV timeout detection for cdnnba.

        Runs the generic ``infer_tv_timeouts_rulebook`` with post-2017
        thresholds and cdnnba-specific action/subType conventions, then
        builds full injection rows by joining context columns from the
        triggering event.
        """
        hits = CDNNBADatasetPL.infer_tv_timeouts_rulebook(
            df,
            thresholds=CDNNBADatasetPL._RULEBOOK_THRESHOLDS_POST2017,
            periods=CDNNBADatasetPL._RULEBOOK_PERIODS_POST2017,
            coach_to_subtypes=CDNNBADatasetPL._CDNNBA_COACH_TO_SUBTYPES,
            dead_ball_action_types=CDNNBADatasetPL._CDNNBA_DEAD_BALL_ACTION_TYPES,
            order_col="orderNumber",
        )
        if hits.height == 0:
            return pl.DataFrame(schema=df.schema)

        # Look up the full row context at the triggering event via join.
        # `order_before` is the orderNumber of the dead-ball event that
        # triggered the mandatory. We place the injected row just before
        # it by subtracting 1 from orderNumber (same convention as the
        # heuristic method).
        context_cols = [
            "game_date",
            "game_seconds_elapsed",
            "seconds_elapsed",
            "scoreHome",
            "scoreAway",
            "possession",
            "IsPlayoff",
            "periodType",
            "season",
            "season_type",
            "score_margin",
            "is_clutch",
            "possession_id",
        ]
        context = df.select(
            "gameId",
            pl.col("orderNumber").alias("order_before"),
            *[c for c in context_cols if c in df.columns],
        )
        joined = hits.join(context, on=["gameId", "order_before"], how="left")

        new_rows = pl.DataFrame(
            {
                "gameId": joined["gameId"],
                "game_date": joined["game_date"],
                "period": joined["period"],
                "game_seconds_elapsed": joined["game_seconds_elapsed"],
                "seconds_remaining": joined["seconds_remaining"],
                "seconds_elapsed": joined["seconds_elapsed"],
                "orderNumber": joined["order_before"] - 1,
                "actionType": pl.Series(["timeout"] * joined.height, dtype=pl.String),
                "subType": pl.Series(["official_inferred"] * joined.height, dtype=pl.String),
                "description": pl.Series(["Inferred TV/Official Timeout"] * joined.height, dtype=pl.String),
                "scoreHome": joined["scoreHome"],
                "scoreAway": joined["scoreAway"],
                "possession": joined["possession"],
                "IsPlayoff": joined["IsPlayoff"],
                "periodType": joined["periodType"],
                "season": joined["season"],
                "season_type": joined["season_type"],
                "personId": pl.Series([0] * joined.height, dtype=pl.Int64),
                "points_scored": pl.Series([0] * joined.height, dtype=pl.Int64),
                "score_margin": joined["score_margin"],
                "is_clutch": joined["is_clutch"],
                "possession_id": joined["possession_id"],
            }
        )

        for col_name, col_dtype in df.schema.items():
            if col_name not in new_rows.columns:
                new_rows = new_rows.with_columns(pl.lit(None).cast(col_dtype).alias(col_name))

        return new_rows.select(df.columns)

    @staticmethod
    def infer_tv_timeouts_rulebook(
        df: pl.DataFrame,
        thresholds: list[int],
        periods: list[int],
        coach_to_subtypes: list[str],
        dead_ball_action_types: set[str],
        action_type_col: str = "actionType",
        sub_type_col: str = "subType",
        gameId_col: str = "gameId",
        period_col: str = "period",
        sr_col: str = "seconds_remaining",
        order_col: str = "orderNumber",
    ) -> pl.DataFrame:
        """Rulebook-based inference of mandatory TV timeouts.

        For each (game, period in ``periods``), walk thresholds in order.
        For each threshold T:
          - If any coach TO fired in the window (T, prev_T], mark slot absorbed
            and advance to next threshold.
          - Else fire a mandatory timeout at the first dead ball with
            seconds_remaining <= T (excluding coach TOs themselves).
          - Only the FIRST unabsorbed slot produces a mandatory.

        Parameters
        ----------
        df : pl.DataFrame
            Play-by-play data. Must contain columns named by
            ``action_type_col``, ``sub_type_col``, ``gameId_col``,
            ``period_col``, ``sr_col``, ``order_col``.
        thresholds : list[int]
            Ordered list of seconds-remaining thresholds to check, e.g.
            [540, 360, 180] for the pre-2017 9:00/6:00/3:00 marks.
        periods : list[int]
            Periods in which the rule applies, e.g. [2, 4] for pre-2017.
        coach_to_subtypes : list[str]
            subType values that indicate a coach timeout (e.g. ["Regular",
            "Short", "Coach Challenge"] for nbastatsv3 or ["full", "challenge"]
            for cdnnba).
        dead_ball_action_types : set[str]
            actionType values considered dead-ball events.

        Returns
        -------
        pl.DataFrame with columns: gameId, period, seconds_remaining,
        order_before (the orderNumber of the triggering event).
        One row per inferred mandatory timeout.
        """
        # Vectorized implementation. For each (gameId, period):
        #  - per threshold k with absorption window (t_k, prev_t_k]:
        #      absorbed_k = any coach TO in that window
        #      fire_order_k, fire_sr_k = first (smallest order_col) non-coach
        #          dead-ball event with sr <= t_k
        #  - the mandatory fires at the FIRST unabsorbed slot that has a fire
        #    candidate. "Unabsorbed" = the slot wasn't skipped due to a coach
        #    TO in its absorption window.
        #
        # Equivalent to the prior Python loop but computed with one group_by.

        is_coach_timeout_expr = pl.col(action_type_col).is_in(["timeout", "Timeout"]) & pl.col(sub_type_col).cast(
            pl.String
        ).is_in(coach_to_subtypes)
        is_dead_ball_expr = pl.col(action_type_col).is_in(list(dead_ball_action_types))
        is_non_coach_dead_ball_expr = is_dead_ball_expr & ~is_coach_timeout_expr

        working = (
            df.filter(pl.col(period_col).is_in(periods))
            .select(gameId_col, period_col, sr_col, order_col, action_type_col, sub_type_col)
            .sort(gameId_col, period_col, order_col)
        )

        if working.height == 0:
            schema = {
                gameId_col: pl.Int64,
                period_col: pl.Int64,
                sr_col: pl.Float64,
                "order_before": pl.Int64,
            }
            return pl.DataFrame(schema=schema)

        # Build aggregation expressions: 3 per threshold.
        agg_exprs: list[pl.Expr] = []
        prev_ts = [720] + thresholds[:-1]
        for prev_t, t in zip(prev_ts, thresholds):
            # Absorbed = any coach TO with sr in (t, prev_t]
            absorbed_expr = (
                (is_coach_timeout_expr & (pl.col(sr_col) <= prev_t) & (pl.col(sr_col) > t)).any().alias(f"absorbed_{t}")
            )
            # First non-coach dead ball with sr <= t (smallest order)
            fire_mask = is_non_coach_dead_ball_expr & (pl.col(sr_col) <= t)
            fire_order_expr = pl.col(order_col).filter(fire_mask).first().alias(f"fire_order_{t}")
            fire_sr_expr = pl.col(sr_col).filter(fire_mask).first().alias(f"fire_sr_{t}")
            agg_exprs.extend([absorbed_expr, fire_order_expr, fire_sr_expr])

        grouped = working.group_by([gameId_col, period_col], maintain_order=True).agg(*agg_exprs)

        # Determine which slot actually fires.
        # Slot k fires iff for all j < k the slot was either absorbed OR had no
        # fire candidate, AND slot k is not absorbed and has a fire candidate.
        # Build a chain of when/then (no .otherwise() until the end so we can
        # keep appending branches — otherwise() returns a plain Expr).
        fire_order_chain = None  # polars Then object
        fire_sr_chain = None
        prior_invalid_expr = pl.lit(True)  # all prior slots were invalid (no fire)
        for t in thresholds:
            absorbed = pl.col(f"absorbed_{t}")
            fo = pl.col(f"fire_order_{t}")
            fsr = pl.col(f"fire_sr_{t}")
            slot_fires = prior_invalid_expr & ~absorbed & fo.is_not_null()
            if fire_order_chain is None:
                fire_order_chain = pl.when(slot_fires).then(fo)
                fire_sr_chain = pl.when(slot_fires).then(fsr)
            else:
                fire_order_chain = fire_order_chain.when(slot_fires).then(fo)
                fire_sr_chain = fire_sr_chain.when(slot_fires).then(fsr)  # type: ignore[assignment]
            # Slot was invalid iff absorbed OR no fire candidate
            prior_invalid_expr = prior_invalid_expr & (absorbed | fo.is_null())

        assert fire_order_chain is not None and fire_sr_chain is not None
        fire_order_expr = fire_order_chain.otherwise(None).alias("order_before")
        fire_sr_expr = fire_sr_chain.otherwise(None).alias(sr_col)

        result = (
            grouped.with_columns(fire_order_expr, fire_sr_expr)
            .filter(pl.col("order_before").is_not_null())
            .select(gameId_col, period_col, sr_col, "order_before")
        )
        return result

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
