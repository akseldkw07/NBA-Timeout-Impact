"""
Polars-based Enriched DataFrame and MemoDataFrame for cleaned cdn.nba.com play-by-play data.

Usage:
    from nba_timeout_impact.datasets.memo_cdnnba_pl import CDNNBAMemoPL

    memo = CDNNBAMemoPL.load_all()
"""

import typing as t

import polars as pl
from kret_polars.memo_df_pl import InputTypedDictPL, MemoDataFramePL, memo_fn, memo_series
from scipy import stats as sp_stats

from .enriched_boxscores import BoxscoresDatasetPL
from .enriched_cdnnba import CDNNBADatasetPL
from .enriched_player_advanced_stats import PlayerAdvancedStatsDatasetPL
from .enriched_player_season_stats import PlayerSeasonStatsDatasetPL
from .enriched_rotations import RotationsDatasetPL
from .enriched_stints import StintsDatasetPL


class CDNNBADatasetInputPL(InputTypedDictPL):
    data: CDNNBADatasetPL
    boxscores: BoxscoresDatasetPL
    player_advanced_stats: PlayerAdvancedStatsDatasetPL
    player_season_stats: PlayerSeasonStatsDatasetPL
    rotations: RotationsDatasetPL
    stints: StintsDatasetPL


class CDNNBAMemoPL(MemoDataFramePL[CDNNBADatasetInputPL]):

    # -- dataset accessors --

    @property
    def cdnnba(self) -> CDNNBADatasetPL:
        return self.inputs["data"]

    @property
    def boxscores(self) -> BoxscoresDatasetPL:
        return self.inputs["boxscores"]

    @property
    def player_advanced_stats(self) -> PlayerAdvancedStatsDatasetPL:
        return self.inputs["player_advanced_stats"]

    @property
    def player_season_stats(self) -> PlayerSeasonStatsDatasetPL:
        return self.inputs["player_season_stats"]

    @property
    def rotations(self) -> RotationsDatasetPL:
        return self.inputs["rotations"]

    @property
    def stints(self) -> StintsDatasetPL:
        return self.inputs["stints"]

    # -- load_all --

    @classmethod
    def load_all(cls) -> "CDNNBAMemoPL":
        """Load all datasets from parquet and return a fully-initialized CDNNBAMemoPL."""
        return cls(
            {
                "data": CDNNBADatasetPL.load_from_parquet(),
                "boxscores": BoxscoresDatasetPL.load_from_parquet(),
                "player_advanced_stats": PlayerAdvancedStatsDatasetPL.load_from_parquet(),
                "player_season_stats": PlayerSeasonStatsDatasetPL.load_from_parquet(),
                "rotations": RotationsDatasetPL.load_from_parquet(),
                "stints": StintsDatasetPL.load_from_parquet(),
            }
        )

    # ------------------------------------------------------------------ #
    #  Core memo series                                                    #
    # ------------------------------------------------------------------ #

    @memo_series
    def f_clock_reversal(self) -> pl.Series:
        """Boolean mask: True for rows where game_seconds_elapsed decreases
        relative to the previous row within the same game.

        These are caused by:
        - ``instantreplay`` requests logged with a slightly stale clock
        - ``memo`` events (stat corrections / annotations) carrying the
          clock of the original play they reference

        Both are non-play metadata — safe to exclude from time-based analysis.
        """
        df = self.cdnnba
        game_boundary = df["gameId"] != df["gameId"].shift(1)
        gse_diff = df["game_seconds_elapsed"].diff()
        return ((gse_diff < 0) & ~game_boundary).fill_null(False)

    # ------------------------------------------------------------------ #
    #  Ported from NBAMemoDF (pandas) — timeouts, lead, streaks, ptrs     #
    # ------------------------------------------------------------------ #

    @memo_fn
    def ptr_n_mins(self, n: float) -> pl.Series:
        """For every event, return the row index of the closest event
        ``n`` minutes ahead (n > 0) or behind (n < 0) within the same game.

        Returns null where no match exists.
        """
        window = n * 60.0
        df = self.cdnnba
        lookup = (
            pl.DataFrame({"gameId": df["gameId"], "gse": df["game_seconds_elapsed"]}).with_row_index("ptr").sort("gse")
        )
        queries = lookup.with_columns((pl.col("gse") + window).alias("target_time")).sort("target_time")
        merged = queries.join_asof(
            lookup.rename({"gse": "gse_r", "ptr": "ptr_r"}),
            left_on="target_time",
            right_on="gse_r",
            by="gameId",
            strategy="nearest",
        )
        return merged.sort("ptr")["ptr_r"]

    # -- timeouts --

    @memo_series
    def f_timeout(self) -> pl.Series:
        return self.cdnnba["actionType"] == "timeout"

    @memo_series
    def f_timeout_endogenous(self) -> pl.Series:
        """Coach-called timeouts (full or challenge)."""
        df = self.cdnnba
        return (df["actionType"] == "timeout") & df["subType"].is_in(["full", "challenge"])

    @memo_series
    def f_timeout_exogenous(self) -> pl.Series:
        """Inferred TV/official timeouts (injected during load)."""
        df = self.cdnnba
        return (df["actionType"] == "timeout") & (df["subType"] == "official_inferred")

    @memo_series
    def f_stoppage(self) -> pl.Series:
        return self.cdnnba["actionType"] == "stoppage"

    @memo_fn
    def f_stoppage_subtype(self, subtype: str) -> pl.Series:
        """Filter for a specific stoppage subType (e.g. 'out-of-bounds', 'injury', 'blood rule')."""
        df = self.cdnnba
        return (df["actionType"] == "stoppage") & (df["subType"] == subtype)

    @memo_series
    def f_coach_challenge(self):
        """Coaching Challenge"""
        df = self.cdnnba
        return (df["actionType"] == "timeout") & (df["subType"] == "challenge")

    # -- lead & lead change --

    @memo_series
    def lead(self) -> pl.Series:
        return self.cdnnba["scoreHome"] - self.cdnnba["scoreAway"]

    @memo_fn
    def bin_sr(self, width: int = 60) -> pl.Series:
        """Bin ``seconds_remaining`` into integer buckets of ``width`` seconds.

        The label is the bin's floor in seconds (e.g. width=60 gives 0, 60, 120, ...).
        Used to coarse-grain the distribution of events relative to the
        period clock — useful for histograms of timeout density at the
        rulebook trigger marks (e.g. 6:59 ≈ bin 420).
        """
        return ((self.cdnnba["seconds_remaining"] // width) * width).cast(pl.Int32).alias(f"sr_bin_{width}s")

    @memo_series
    def wall_clock_delta_seconds(self) -> pl.Series:
        """Real-world seconds between this event and the previous event in the same game.

        Computed from `timeActual` (UTC datetime). The first event of each game
        is null. Rulebook-injected TV timeouts have `timeActual=null` by
        construction, so their own delta and the next event's delta are both
        null.
        """
        df = self.cdnnba
        game_boundary = (df["gameId"] != df["gameId"].shift(1)).fill_null(True)
        delta_ms = (df["timeActual"] - df["timeActual"].shift(1)).dt.total_milliseconds()
        return (delta_ms / 1000.0).set(game_boundary, None).alias("wall_clock_delta_seconds")

    @memo_fn
    def lead_change_n_mins(self, n: float) -> pl.Series:
        """Positive = home lead increased after n minutes; negative = decreased."""
        future_lead = self.lead.gather(self.ptr_n_mins(n).cast(pl.UInt32).fill_null(0))
        valid = self.ptr_n_mins(n).is_not_null()
        return (future_lead - self.lead).set(~valid, None)

    @memo_fn
    def score_diff_n_mins(self, n: float) -> pl.Series:
        """Alias for lead_change_n_mins."""
        return self.lead_change_n_mins(n)

    # -- streaks --

    @memo_series
    def streak(self) -> pl.Series:
        """Running scoring streak. Positive = home run, negative = away run.

        Resets at game boundaries and when the scoring team changes.
        """
        df = self.cdnnba
        game_boundary = df["gameId"] != df["gameId"].shift(1)

        home_pts = df["scoreHome"].diff().clip(0, None).fill_null(0).set(game_boundary, 0)
        away_pts = df["scoreAway"].diff().clip(0, None).fill_null(0).set(game_boundary, 0)
        net = home_pts - away_pts

        # Forward-fill the sign of the last scoring event (home=1, away=-1)
        scorer = net.sign().set(net == 0, None).forward_fill().fill_null(0)

        # New segment when: game starts OR scoring team changes
        new_segment = game_boundary | ((net != 0) & (scorer != scorer.shift(1).fill_null(0)))
        segment = new_segment.cum_sum()

        # Cumulative sum of net within each segment
        return pl.DataFrame({"net": net, "segment": segment}).with_columns(
            pl.col("net").cum_sum().over("segment").alias("streak")
        )["streak"]

    @memo_fn
    def f_streak_n(self, n: int, direction: t.Literal["home", "away", "either"] = "either") -> pl.Series:
        if direction == "home":
            return self.streak >= n
        elif direction == "away":
            return self.streak <= -n
        elif direction == "either":
            return self.streak.abs() >= n
        else:
            raise ValueError(f"Invalid direction: {direction}")

    # ------------------------------------------------------------------ #
    #  Possession-level table and post-timeout analysis                    #
    # ------------------------------------------------------------------ #

    @property
    def possessions(self) -> pl.DataFrame:
        """One row per possession with aggregated stats.

        Columns: gameId, possession_id, period, possession_points,
        possession_outcome, poss_start_gse, seconds_remaining,
        score_margin, streak, season, season_type, IsPlayoff,
        possession (teamId of possessing team).

        Cached in ``_df_dict`` on first access.
        """
        key = "_possessions"
        if key not in self._df_dict:
            print("Computing possessions table...")
            df = pl.DataFrame._from_pydf(self.cdnnba._df)
            streak_s = self.streak

            poss = (
                df.with_columns(streak_s.alias("_streak"))
                .filter(pl.col("possession_id").is_not_null())
                .group_by("gameId", "possession_id", maintain_order=True)
                .agg(
                    pl.col("period").first(),
                    pl.col("possession_points").last(),
                    pl.col("possession_outcome").last(),
                    pl.col("game_seconds_elapsed").first().alias("poss_start_gse"),
                    pl.col("seconds_remaining").first(),
                    pl.col("score_margin").first(),
                    pl.col("_streak").first().alias("streak"),
                    pl.col("season").first(),
                    pl.col("season_type").first(),
                    pl.col("IsPlayoff").first(),
                    pl.col("possession").first(),
                )
            )
            print(f"  {poss.height:,} possessions")
            self._df_dict[key] = poss
        return self._df_dict[key]

    @property
    def post_timeout_possessions(self) -> pl.DataFrame:
        """For each timeout row, join the next N possessions (up to 5) in the same game.

        Returns a table with one row per (timeout, offset) pair:
            all timeout-row context columns + offset (1-5) + possession_points + possession_outcome

        Cached in ``_df_dict`` on first access.
        """
        key = "_post_timeout_possessions"
        if key not in self._df_dict:
            print("Computing post_timeout_possessions...")
            df = pl.DataFrame._from_pydf(self.cdnnba._df)

            # Timeout rows with their context
            # teamId = team that called the timeout (null for official_inferred)
            timeouts = df.filter(pl.col("actionType") == "timeout").select(
                "gameId",
                "possession_id",
                "period",
                "game_seconds_elapsed",
                "seconds_remaining",
                "score_margin",
                "season",
                "season_type",
                "IsPlayoff",
                pl.col("subType").alias("timeout_subtype"),
                pl.col("teamId").alias("timeout_team"),
            )

            poss = self.possessions
            max_offset = 5
            frames = []
            for offset in range(1, max_offset + 1):
                joined = (
                    timeouts.with_columns((pl.col("possession_id") + offset).alias("_target_poss_id"))
                    .join(
                        poss.select(
                            "gameId",
                            "possession_id",
                            "possession_points",
                            "possession_outcome",
                            pl.col("possession").alias("poss_team"),
                        ),
                        left_on=["gameId", "_target_poss_id"],
                        right_on=["gameId", "possession_id"],
                        how="left",
                    )
                    .with_columns(
                        pl.lit(offset).alias("offset").cast(pl.Int32),
                        # Whether this possession belongs to the team that called the timeout
                        (pl.col("timeout_team") == pl.col("poss_team")).alias("is_calling_team_poss"),
                    )
                    .drop("_target_poss_id")
                )
                frames.append(joined)

            result = pl.concat(frames)
            # Drop rows where the possession doesn't exist (end of game)
            result = result.filter(pl.col("possession_points").is_not_null())
            print(f"  {result.height:,} (timeout, offset) pairs")
            self._df_dict[key] = result
        return self._df_dict[key]

    # ------------------------------------------------------------------ #
    #  Counterfactual timeout analysis                                     #
    # ------------------------------------------------------------------ #

    # Window size for trailing/forward possession windows.
    # Even number so each team gets exactly half the possessions.
    _COUNTERFACTUAL_WINDOW = 6

    @property
    def timeout_counterfactual(self) -> pl.DataFrame:
        """Bucket-matched comparison of post-timeout vs non-timeout outcomes.

        For every possession, computes the trailing and forward N-possession
        net (N = _COUNTERFACTUAL_WINDOW, default 6) from that team's perspective.
        Uses an even window so each team gets exactly N/2 possessions.

        Timeouts are matched to non-timeout possessions with the same
        trailing net (rounded to nearest integer).

        Returns a DataFrame with one row per trailing_net bucket:
            trail_bucket, to_fwd_mean, to_fwd_std, to_n,
            ctrl_fwd_mean, ctrl_fwd_std, ctrl_n,
            causal_effect, p_value

        Cached in ``_df_dict`` on first access.
        """
        key = "_timeout_counterfactual"
        if key not in self._df_dict:
            print("Computing timeout_counterfactual...")
            self._df_dict[key] = self._compute_timeout_counterfactual()
        return self._df_dict[key]

    def _compute_timeout_counterfactual(self) -> pl.DataFrame:

        W = self._COUNTERFACTUAL_WINDOW
        poss = self.possessions
        df = pl.DataFrame._from_pydf(self.cdnnba._df)

        # Build lag/lead columns for every possession
        p = poss.select("gameId", "possession_id", "possession_points", "possession").sort("gameId", "possession_id")
        for i in range(1, W + 1):
            p = p.with_columns(
                pl.col("possession_points").shift(i).over("gameId").alias(f"pts_lag{i}"),
                pl.col("possession").shift(i).over("gameId").alias(f"tm_lag{i}"),
                pl.col("possession_points").shift(-i).over("gameId").alias(f"pts_lead{i}"),
                pl.col("possession").shift(-i).over("gameId").alias(f"tm_lead{i}"),
            )

        # Trailing/forward net from possessor's perspective
        trail_for = pl.lit(0)
        trail_ag = pl.lit(0)
        fwd_for = pl.lit(0)
        fwd_ag = pl.lit(0)
        for i in range(1, W + 1):
            same = pl.col(f"tm_lag{i}") == pl.col("possession")
            trail_for = trail_for + pl.when(same).then(pl.col(f"pts_lag{i}")).otherwise(0)
            trail_ag = trail_ag + pl.when(~same).then(pl.col(f"pts_lag{i}")).otherwise(0)
            same_f = pl.col(f"tm_lead{i}") == pl.col("possession")
            fwd_for = fwd_for + pl.when(same_f).then(pl.col(f"pts_lead{i}")).otherwise(0)
            fwd_ag = fwd_ag + pl.when(~same_f).then(pl.col(f"pts_lead{i}")).otherwise(0)

        p = p.with_columns(
            (trail_for - trail_ag).alias("trailing_net"),
            (fwd_for - fwd_ag).alias("forward_net"),
        ).drop_nulls(subset=[f"pts_lag{W}", f"pts_lead{W}"])

        # Split into timeout and non-timeout possessions
        timeout_keys = (
            df.filter((pl.col("actionType") == "timeout") & (pl.col("subType") == "full"))
            .select("gameId", "possession_id", pl.col("teamId").alias("timeout_team"))
            .unique()
        )

        p_to = p.join(timeout_keys, on=["gameId", "possession_id"], how="inner").with_columns(
            # Flip net if timeout_team != possession at this row
            pl.when(pl.col("timeout_team") == pl.col("possession"))
            .then(pl.col("trailing_net"))
            .otherwise(-pl.col("trailing_net"))
            .alias("trailing_net"),
            pl.when(pl.col("timeout_team") == pl.col("possession"))
            .then(pl.col("forward_net"))
            .otherwise(-pl.col("forward_net"))
            .alias("forward_net"),
        )
        p_ctrl = p.join(timeout_keys.select("gameId", "possession_id"), on=["gameId", "possession_id"], how="anti")

        print(f"  Window: {W} possessions ({W // 2} per team)")
        print(f"  Timeout possessions: {p_to.height:,}, control: {p_ctrl.height:,}")

        # Bucket and compare
        to_bucketed = p_to.with_columns(pl.col("trailing_net").round(0).cast(pl.Int32).alias("trail_bucket"))
        ctrl_bucketed = p_ctrl.with_columns(pl.col("trailing_net").round(0).cast(pl.Int32).alias("trail_bucket"))

        to_agg = to_bucketed.group_by("trail_bucket").agg(
            pl.col("forward_net").mean().alias("to_fwd_mean"),
            pl.col("forward_net").std().alias("to_fwd_std"),
            pl.col("forward_net").len().alias("to_n"),
        )
        ctrl_agg = ctrl_bucketed.group_by("trail_bucket").agg(
            pl.col("forward_net").mean().alias("ctrl_fwd_mean"),
            pl.col("forward_net").std().alias("ctrl_fwd_std"),
            pl.col("forward_net").len().alias("ctrl_n"),
        )

        merged = to_agg.join(ctrl_agg, on="trail_bucket", how="inner").sort("trail_bucket")
        merged = merged.filter((pl.col("to_n") >= 50) & (pl.col("ctrl_n") >= 50))
        merged = merged.with_columns(
            (pl.col("to_fwd_mean") - pl.col("ctrl_fwd_mean")).alias("causal_effect"),
        )

        # Compute p-values per bucket via Welch's t-test
        p_values = []
        for row in merged.iter_rows(named=True):
            b = row["trail_bucket"]
            to_vals = to_bucketed.filter(pl.col("trail_bucket") == b)["forward_net"].to_numpy()
            ctrl_vals = ctrl_bucketed.filter(pl.col("trail_bucket") == b)["forward_net"].to_numpy()
            _, pval = sp_stats.ttest_ind(to_vals, ctrl_vals, equal_var=False)
            p_values.append(pval)

        merged = merged.with_columns(pl.Series("p_value", p_values))

        # Compute overall weighted causal estimate
        weighted = (merged["causal_effect"] * merged["to_n"]).sum() / merged["to_n"].sum()  # type: ignore
        total_to = merged["to_n"].sum()
        print(f"  Causal effect: {weighted:+.4f} pts over {W} possessions ({W // 2} per team), n={total_to:,}")
        return merged

    # ------------------------------------------------------------------ #
    #  Home team lookup                                                    #
    # ------------------------------------------------------------------ #

    @property
    def home_team_per_game(self) -> pl.DataFrame:
        """One row per gameId with the home team's teamId.

        Derived from scoring events where scoreHome increases.
        Cached in ``_df_dict``.
        """
        key = "_home_team_per_game"
        if key not in self._df_dict:
            df = pl.DataFrame._from_pydf(self.cdnnba._df)
            scoring = df.filter((pl.col("points_scored") > 0) & (pl.col("teamId") > 0))
            scoring = scoring.with_columns(pl.col("scoreHome").diff().over("gameId").alias("_hd"))
            self._df_dict[key] = (
                scoring.filter(pl.col("_hd") > 0).select("gameId", pl.col("teamId").alias("home_teamId")).unique()
            )
        return self._df_dict[key]

    # ------------------------------------------------------------------ #
    #  Stoppage impact on runs                                             #
    # ------------------------------------------------------------------ #

    def stoppage_run_impact(self, run_size: int = 5, minutes: float = 3.0) -> pl.DataFrame:
        """Compare recovery after endogenous timeout, exogenous stoppage, or no stoppage
        when a team is suffering a scoring run.

        For every spine event during a run of |streak| >= *run_size*, classifies
        it as endogenous (coach timeout by suffering team), exogenous (TV timeout
        or stoppage), or control (no stoppage). Then measures lead change over the
        next *minutes* from the suffering team's perspective.

        Returns a DataFrame with columns:
            group ("endogenous", "exogenous", "control"),
            recovery (lead change from suffering team's perspective),
            running_team_pts (points scored by running team in next N minutes),
            streak_at_event, gameId, game_seconds_elapsed,
            suffering_location, suffering_margin

        NOT cached — recomputed each call since parameters vary.
        """
        df = pl.DataFrame._from_pydf(self.cdnnba._df)
        streak_s = self.streak
        lead_s = self.lead
        lead_fwd = self.lead_change_n_mins(minutes)
        ptr_fwd = self.ptr_n_mins(minutes)
        home_teams = self.home_team_per_game

        # Compute running team's raw points in next N minutes
        # ptr_fwd gives us the row index N minutes ahead
        score_home_now = df["scoreHome"]
        score_away_now = df["scoreAway"]
        valid_ptr = ptr_fwd.is_not_null()
        safe_ptr = ptr_fwd.cast(pl.UInt32).fill_null(0)
        score_home_fwd = df["scoreHome"].gather(safe_ptr).set(~valid_ptr, None)
        score_away_fwd = df["scoreAway"].gather(safe_ptr).set(~valid_ptr, None)
        home_pts_scored = score_home_fwd - score_home_now
        away_pts_scored = score_away_fwd - score_away_now

        analysis = pl.DataFrame(
            {
                "gameId": df["gameId"],
                "game_seconds_elapsed": df["game_seconds_elapsed"],
                "actionType": df["actionType"],
                "subType": df["subType"],
                "teamId": df["teamId"],
                "streak": streak_s,
                "lead": lead_s,
                "lead_change": lead_fwd,
                "home_pts_scored": home_pts_scored,
                "away_pts_scored": away_pts_scored,
            }
        ).join(home_teams, on="gameId", how="left")

        # Filter to events during a run
        analysis = analysis.filter(pl.col("streak").abs() >= run_size)
        # Drop rows where we can't measure forward impact
        analysis = analysis.filter(pl.col("lead_change").is_not_null())

        # From suffering team's perspective:
        # streak > 0 → home running, away suffering → flip sign
        # streak < 0 → away running, home suffering → keep sign
        # Running team's points:
        # streak > 0 → home running → running_team_pts = home_pts_scored
        # streak < 0 → away running → running_team_pts = away_pts_scored
        analysis = analysis.with_columns(
            (-pl.col("streak").sign() * pl.col("lead_change")).alias("recovery"),
            (-pl.col("streak").sign() * pl.col("lead")).alias("suffering_margin"),
            pl.when(pl.col("streak") > 0)
            .then(pl.col("home_pts_scored"))
            .otherwise(pl.col("away_pts_scored"))
            .alias("running_team_pts"),
        )

        # Classify events
        is_timeout = pl.col("actionType") == "timeout"
        is_endogenous_sub = pl.col("subType").is_in(["full", "challenge"])
        is_exogenous_sub = pl.col("subType") == "official_inferred"
        is_stoppage = pl.col("actionType") == "stoppage"

        # For endogenous: timeout called by the SUFFERING team
        # streak > 0 (home running) → suffering = away → timeout teamId != home
        # streak < 0 (away running) → suffering = home → timeout teamId == home
        suffering_called = ((pl.col("streak") > 0) & (pl.col("teamId") != pl.col("home_teamId"))) | (
            (pl.col("streak") < 0) & (pl.col("teamId") == pl.col("home_teamId"))
        )

        analysis = analysis.with_columns(
            pl.when(is_timeout & is_endogenous_sub & suffering_called)
            .then(pl.lit("endogenous"))
            .when(is_timeout & is_exogenous_sub)
            .then(pl.lit("exogenous"))
            .when(is_stoppage)
            .then(pl.lit("exogenous"))
            .otherwise(pl.lit("control"))
            .alias("group")
        )

        # De-duplicate: for control, we have many events per run — sample one per
        # (gameId, run segment) to avoid over-counting.
        # For timeouts/stoppages, keep all (they're sparse).
        # A "run segment" is a contiguous block of the same streak sign within a game.
        analysis = analysis.with_columns(
            pl.col("streak").sign().alias("_streak_sign"),
        ).with_columns(
            (
                (pl.col("_streak_sign") != pl.col("_streak_sign").shift(1))
                | (pl.col("gameId") != pl.col("gameId").shift(1))
            )
            .cum_sum()
            .alias("_run_segment")
        )

        # For control: take the FIRST event in each run segment (the moment the run reaches threshold)
        control = (
            analysis.filter(pl.col("group") == "control")
            .group_by("gameId", "_run_segment", maintain_order=True)
            .first()
        )
        non_control = analysis.filter(pl.col("group") != "control")

        # Tag whether the suffering team is home or away
        # streak > 0 → home running → suffering = away
        # streak < 0 → away running → suffering = home
        non_control = non_control.with_columns(
            pl.when(pl.col("streak") > 0).then(pl.lit("away")).otherwise(pl.lit("home")).alias("suffering_location")
        )
        control = control.with_columns(
            pl.when(pl.col("streak") > 0).then(pl.lit("away")).otherwise(pl.lit("home")).alias("suffering_location")
        )

        cols = [
            "group",
            "recovery",
            "running_team_pts",
            pl.col("streak").alias("streak_at_event"),
            "gameId",
            "game_seconds_elapsed",
            "suffering_location",
            "suffering_margin",
        ]
        result = pl.concat(
            [
                non_control.select(cols),
                control.select(cols),
            ]
        )

        for g in ["endogenous", "exogenous", "control"]:
            grp = result.filter(pl.col("group") == g)
            if grp.height > 0:
                print(
                    f"  {g}: n={grp.height:,}, recovery mean={grp['recovery'].mean():.3f}, "
                    f"std={grp['recovery'].std():.3f}"
                )
            else:
                print(f"  {g}: no data")

        return result

    # ------------------------------------------------------------------ #
    #  Pointer memo series — row indices into supplemental datasets        #
    #                                                                      #
    #  Each ptr_* returns a pl.Series(UInt32) aligned to the spine.        #
    #  Usage:  memo.boxscores[memo.ptr_boxscores()]  to materialize.       #
    # ------------------------------------------------------------------ #

    @memo_series
    def ptr_boxscores(self) -> pl.Series:
        """Row index into boxscores for each spine row, joined on (gameId, personId).
        Null where personId == 0 or no boxscore match.
        """
        spine = self.cdnnba.select("gameId", "personId")
        box = pl.DataFrame._from_pydf(self.boxscores._df).select("gameId", "personId").with_row_index("_ptr")
        return spine.join(box, on=["gameId", "personId"], how="left", coalesce=True)["_ptr"]

    @memo_series
    def ptr_player_advanced_stats(self) -> pl.Series:
        """Row index into player_advanced_stats for each spine row,
        joined on (personId=PLAYER_ID, season=season_int).
        """
        spine = self.cdnnba.select("personId", "season")
        pas = (
            pl.DataFrame._from_pydf(self.player_advanced_stats._df)
            .rename({"PLAYER_ID": "personId", "season_int": "season"})
            .select("personId", "season")
            .with_row_index("_ptr")
        )
        return spine.join(pas, on=["personId", "season"], how="left", coalesce=True)["_ptr"]

    @memo_series
    def ptr_player_season_stats(self) -> pl.Series:
        """Row index into player_season_stats (TOT-deduped) for each spine row,
        joined on (personId=PLAYER_ID, season=season_int, season_type).

        For traded players the TOT aggregate row is preferred so the pointer
        is 1:1 per (player, season, season_type).
        """
        spine = self.cdnnba.select("personId", "season", "season_type")
        # Add row index to the ORIGINAL df first, so ptrs reference original row positions.
        pss = pl.DataFrame._from_pydf(self.player_season_stats._df).with_row_index("_ptr")
        pss_tot = pss.filter(pl.col("TEAM_ABBREVIATION") == "TOT")
        tot_keys = pss_tot.select("PLAYER_ID", "season_int", "season_type")
        pss_single = pss.join(tot_keys, on=["PLAYER_ID", "season_int", "season_type"], how="anti")
        pss_deduped = (
            pl.concat([pss_tot, pss_single])
            .rename({"PLAYER_ID": "personId", "season_int": "season"})
            .select("personId", "season", "season_type", "_ptr")
        )
        return spine.join(pss_deduped, on=["personId", "season", "season_type"], how="left", coalesce=True)["_ptr"]

    @memo_series
    def ptr_stints(self) -> pl.Series:
        """Row index into stints for each spine row. Uses join_asof on
        (gameId, personId) with in_game_seconds <= game_seconds_elapsed < out_game_seconds.
        """
        spine = self.cdnnba.select("gameId", "personId", "game_seconds_elapsed").with_row_index("_spine_idx")
        st = (
            pl.DataFrame._from_pydf(self.stints._df)
            .with_row_index("_ptr")
            .select("gameId", "personId", "in_game_seconds", "out_game_seconds", "_ptr")
            .sort("gameId", "personId", "in_game_seconds")
        )
        joined = spine.sort("gameId", "personId", "game_seconds_elapsed").join_asof(
            st,
            left_on="game_seconds_elapsed",
            right_on="in_game_seconds",
            by=["gameId", "personId"],
            strategy="backward",
        )
        # Null out pointer where game_seconds_elapsed is past the stint's out time
        joined = joined.with_columns(
            pl.when(pl.col("game_seconds_elapsed") >= pl.col("out_game_seconds"))
            .then(None)
            .otherwise(pl.col("_ptr"))
            .alias("_ptr")
        )
        return joined.sort("_spine_idx")["_ptr"]

    @memo_series
    def ptr_rotations(self) -> pl.Series:
        """Row index into rotations for each spine row. Uses join_asof on
        (gameId, personId) with IN_TIME_REAL/10 <= game_seconds_elapsed < OUT_TIME_REAL/10.
        """
        spine = self.cdnnba.select("gameId", "personId", "game_seconds_elapsed").with_row_index("_spine_idx")
        rot = (
            pl.DataFrame._from_pydf(self.rotations._df)
            .with_row_index("_ptr")
            .rename({"PERSON_ID": "personId"})
            .with_columns(
                (pl.col("IN_TIME_REAL") / 10.0).alias("in_seconds"),
                (pl.col("OUT_TIME_REAL") / 10.0).alias("out_seconds"),
            )
            .select("gameId", "personId", "in_seconds", "out_seconds", "_ptr")
            .sort("gameId", "personId", "in_seconds")
        )
        joined = spine.sort("gameId", "personId", "game_seconds_elapsed").join_asof(
            rot,
            left_on="game_seconds_elapsed",
            right_on="in_seconds",
            by=["gameId", "personId"],
            strategy="backward",
        )
        # Null out pointer where game_seconds_elapsed is past the rotation's out time
        joined = joined.with_columns(
            pl.when(pl.col("game_seconds_elapsed") >= pl.col("out_seconds"))
            .then(None)
            .otherwise(pl.col("_ptr"))
            .alias("_ptr")
        )
        return joined.sort("_spine_idx")["_ptr"]
