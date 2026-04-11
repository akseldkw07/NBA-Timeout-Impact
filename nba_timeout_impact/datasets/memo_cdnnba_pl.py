"""
Polars-based Enriched DataFrame and MemoDataFrame for cleaned cdn.nba.com play-by-play data.

Usage:
    from nba_timeout_impact.datasets.memo_cdnnba_pl import CDNNBAMemoPL

    memo = CDNNBAMemoPL.load_all()
"""

import typing as t

import polars as pl
from kret_polars.memo_df_pl import InputTypedDictPL, MemoDataFramePL, memo_fn, memo_series

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

    # -- lead & lead change --

    @memo_series
    def lead(self) -> pl.Series:
        return self.cdnnba["scoreHome"] - self.cdnnba["scoreAway"]

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
