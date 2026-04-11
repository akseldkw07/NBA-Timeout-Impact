"""
Polars-based Enriched DataFrame and MemoDataFrame for cleaned cdn.nba.com play-by-play data.

Usage:
    from nba_timeout_impact.datasets.memo_cdnnba_pl import CDNNBAMemoPL

    memo = CDNNBAMemoPL.load_all()
"""

import typing as t
from pathlib import Path

import polars as pl
from kret_polars.enriched_df_pl import Enriched_DF_PL
from kret_polars.memo_df_pl import InputTypedDictPL, MemoDataFramePL, memo_fn, memo_series

from nba_timeout_impact.constants import NBAConstants
from .enriched_cdnnba import CDNNBADatasetPL
from .enriched_boxscores import BoxscoresDatasetPL
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
        return cls({
            "data": CDNNBADatasetPL.load_from_parquet(),
            "boxscores": BoxscoresDatasetPL.load_from_parquet(),
            "player_advanced_stats": PlayerAdvancedStatsDatasetPL.load_from_parquet(),
            "player_season_stats": PlayerSeasonStatsDatasetPL.load_from_parquet(),
            "rotations": RotationsDatasetPL.load_from_parquet(),
            "stints": StintsDatasetPL.load_from_parquet(),
        })

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
        box = (
            pl.DataFrame._from_pydf(self.boxscores._df)
            .select("gameId", "personId")
            .with_row_index("_ptr")
        )
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
        spine = (
            self.cdnnba
            .select("gameId", "personId", "game_seconds_elapsed")
            .with_row_index("_spine_idx")
        )
        st = (
            pl.DataFrame._from_pydf(self.stints._df)
            .with_row_index("_ptr")
            .select("gameId", "personId", "in_game_seconds", "out_game_seconds", "_ptr")
            .sort("gameId", "personId", "in_game_seconds")
        )
        joined = (
            spine.sort("gameId", "personId", "game_seconds_elapsed")
            .join_asof(
                st,
                left_on="game_seconds_elapsed",
                right_on="in_game_seconds",
                by=["gameId", "personId"],
                strategy="backward",
            )
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
        spine = (
            self.cdnnba
            .select("gameId", "personId", "game_seconds_elapsed")
            .with_row_index("_spine_idx")
        )
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
        joined = (
            spine.sort("gameId", "personId", "game_seconds_elapsed")
            .join_asof(
                rot,
                left_on="game_seconds_elapsed",
                right_on="in_seconds",
                by=["gameId", "personId"],
                strategy="backward",
            )
        )
        # Null out pointer where game_seconds_elapsed is past the rotation's out time
        joined = joined.with_columns(
            pl.when(pl.col("game_seconds_elapsed") >= pl.col("out_seconds"))
            .then(None)
            .otherwise(pl.col("_ptr"))
            .alias("_ptr")
        )
        return joined.sort("_spine_idx")["_ptr"]

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
