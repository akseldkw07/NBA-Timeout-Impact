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
    timeout_role: pl.Series  # String — slot_K_mandatory / discretionary / challenge / ""
    timeout_cause: (
        pl.Series
    )  # String — tv_mandatory / coach_preempt / coach_absorb / coach_discretionary / challenge / ""
    timeout_duration_s: pl.Series  # Float64 — wall-clock seconds to game resumption; null on non-TO rows
    cumTimeoutsPeriod: pl.Series  # Int64 — cumulative count of timeouts in (gameId, period) at this row
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
        """Load the enriched cdnnba parquet and inject the timeout
        classification columns (``timeout_role``, ``timeout_cause``,
        ``timeout_duration_s``).

        Injection preserves row order — the classifier doesn't sort — so
        the result is row-aligned with the raw parquet and the
        ``CDNNBAMemoPL`` memo series.
        """
        path = path or NBAConstants.NBA_DATA_DIR / "cdnnba_enriched.parquet"
        df = pl.read_parquet(path)
        df = cls._inject_timeout_columns(df)
        ret = cls(df)
        ret.validate_data()
        return ret

    @staticmethod
    def _inject_timeout_columns(df: pl.DataFrame) -> pl.DataFrame:
        """Add ``timeout_role``, ``timeout_cause``, ``timeout_duration_s``,
        and ``cumTimeoutsPeriod`` to a raw cdnnba frame.

        Uses ``TVTimeoutValidation.compute_timeout_duration_s`` for the
        wall-clock duration (sanity-clamped to ``[0, 600]`` seconds; nulls
        outside that range). ``cumTimeoutsPeriod`` is the cumulative count
        of timeout rows in each ``(gameId, period)`` (inclusive at each row)
        and is what drives the rulebook-faithful cause classification.

        Preserves row order so the result is row-aligned with the input.
        Raises if the classifier ever returns a different height than the
        input — alignment is a hard contract.
        """
        # Local import to avoid a circular dependency at module load time.
        from nba_timeout_impact.data_pipes.tv_timeout_injection import TVTimeoutValidation

        duration = TVTimeoutValidation.compute_timeout_duration_s(df)
        df = df.with_columns(duration.alias("timeout_duration_s"))
        classified = TVTimeoutValidation.classify_timeouts(df, source="cdnnba")
        if classified.height != df.height:
            raise RuntimeError(
                f"timeout classifier dropped rows ({classified.height} vs {df.height}) "
                "— alignment broken; refusing to merge"
            )
        return df.with_columns(
            classified["timeout_role"].alias("timeout_role"),
            classified["timeout_cause"].alias("timeout_cause"),
            classified["cumTimeoutsPeriod"].alias("cumTimeoutsPeriod"),
        )

    # Post-2017 mandatory timeouts are charged to a team's count and logged
    # identically to coach-called TOs in the cdnnba feed — no row-level
    # injection is needed. At load time we DO enrich each row with three
    # label columns (see ``_inject_timeout_columns`` for the source):
    #   - ``timeout_role``       : slot_K_mandatory / discretionary / challenge / ""
    #   - ``timeout_cause``      : tv_mandatory / coach_preempt / coach_absorb /
    #                             coach_discretionary / challenge / ""
    #   - ``timeout_duration_s`` : wall-clock duration in seconds (null off-TO)
    # See ``TVTimeoutValidation.classify_timeouts`` for the cause taxonomy.
    # CAUTION: rows with ``timeout_cause == "tv_mandatory"`` carry
    # ``teamTricode`` / ``teamId`` per the NBA's structural charge-to-home
    # (slot 1) or charge-to-road (slot 2) convention. For *true auto-fires*
    # (no coach actually called the TO) the team label is a bookkeeping
    # artifact, NOT the coach's decision. Downstream analyses that depend
    # on which coach made a decision should restrict to ``coach_*`` causes.

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
