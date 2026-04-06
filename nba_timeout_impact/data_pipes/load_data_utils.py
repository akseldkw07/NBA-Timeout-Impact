"""
Utility helpers for the NBA Timeout Impact project.

Data lives in the sibling repo:
    /Users/Akseldkw/coding/Columbia/nba_data/datasets/

Available dataset types
-----------------------
nbastats    - stats.nba.com play-by-play (1996-2024)
nbastatsv3  - stats.nba.com v3 play-by-play (1996-2025)
datanba     - data.nba.com play-by-play with court coordinates (2016-2024)
pbpstats    - pbpstats.com play-by-play with possession context (2000-2024)
shotdetail  - shot-level detail (1996-2025)
cdnnba      - CDN NBA data (2020-2025)
matchups    - matchup data (2017-2025)

Season type suffix: none = regular season, _po_ = playoffs
File naming: <type>_<year>.tar.xz  or  <type>_po_<year>.tar.xz

EVENTMSGTYPE reference (nbastats)
----------------------------------
1  Made field goal
2  Missed field goal
3  Free throw
4  Rebound
5  Turnover
6  Foul
7  Violation
8  Substitution
9  Timeout  <- key event for this project
10 Jump ball
11 Ejection
12 Start of period
13 End of period
18 Instant replay
"""

import tarfile
import time
import typing as t
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import (
    LeagueGameLog,
    gamerotation,
    leaguedashplayerbiostats,
    leaguedashptstats,
    leaguehustlestatsplayer,
    playercareerstats,
    playerestimatedmetrics,
)

from nba_timeout_impact.constants import NBAConstants

DATA_SRC_TYPE = t.Literal["nbastats", "nbastatsv3", "datanba", "pbpstats", "shotdetail", "cdnnba", "matchups"]
SeasonsArg = int | t.Sequence[int] | None


def _resolve_parent_dir() -> Path:
    """Resolve the parent directory containing nba_data, nba-on-court, etc.

    Walks up from this file's location looking for a directory that contains
    'nba_data/datasets/'.  Works on any machine as long as the repo layout is:

        <parent>/
            NBA-Timeout-Impact/   (this repo)
            nba_data/datasets/
            nba-on-court/
    """
    anchor = Path(__file__).resolve().parent
    for _ in range(6):
        anchor = anchor.parent
        if (anchor / "nba_data" / "datasets").is_dir():
            return anchor
    raise RuntimeError(
        "Cannot find parent directory containing nba_data/datasets/. "
        "Expected it as a sibling of the NBA-Timeout-Impact repo."
    )


class NBADataLoader:
    """
    Loads NBA play-by-play data from the local nba_data archive.

    All methods are classmethods — no instantiation needed:

        from nba_timeout_impact.load_data_utils import NBADataLoader as NBA

        df    = NBA.load_seasons("nbastats", seasons=2022)
        dates = NBA.load_game_dates(seasons=2022)
        avail = NBA.available_seasons("nbastats")
    """

    _PARENT: t.ClassVar[Path] = _resolve_parent_dir()
    DATA_DIR: t.ClassVar[Path] = _PARENT / "nba_data" / "datasets"
    ON_COURT_DIR: t.ClassVar[Path] = _PARENT / "nba-on-court"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @classmethod
    def _tar_path(cls, data_type: str, season: int, playoffs: bool = False) -> Path:
        suffix = f"_po_{season}" if playoffs else f"_{season}"
        return cls.DATA_DIR / f"{data_type}{suffix}.tar.xz"

    @classmethod
    def _resolve_seasons(
        cls,
        data_type: DATA_SRC_TYPE,
        seasons: SeasonsArg,
        playoffs: bool,
    ) -> list[int]:
        """Normalise the ``seasons`` argument to a concrete list of ints."""
        if seasons is None:
            result = cls.available_seasons(data_type, playoffs)
            if not result:
                raise FileNotFoundError(
                    f"No local files found for data_type='{data_type}', playoffs={playoffs}. "
                    f"Expected files in: {cls.DATA_DIR}"
                )
            return result
        if isinstance(seasons, int):
            if seasons < 0:
                all_s = cls.available_seasons(data_type, playoffs)
                return sorted(all_s)[seasons:]  # last N seasons
            return [seasons]
        return list(seasons)

    @staticmethod
    def _read_tar_csv(path: Path, **read_csv_kwargs) -> pd.DataFrame:
        csv_name = path.name.replace(".tar.xz", ".csv")
        with tarfile.open(path, "r:xz") as tar:
            f = tar.extractfile(csv_name)
            assert f is not None, f"CSV {csv_name} not found in {path}"
            return pd.read_csv(f, **read_csv_kwargs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def available_seasons(cls, data_type: DATA_SRC_TYPE = "nbastats", playoffs: bool = False) -> list[int]:
        """Return sorted list of locally available seasons for a given data type."""
        prefix = f"{data_type}_po_" if playoffs else f"{data_type}_"
        pattern = f"{data_type}_po_*.tar.xz" if playoffs else f"{data_type}_[0-9]*.tar.xz"
        seasons = []
        for f in sorted(cls.DATA_DIR.glob(pattern)):
            stem = f.name.replace(".tar.xz", "")
            try:
                seasons.append(int(stem.replace(prefix, "")))
            except ValueError:
                pass
        return seasons

    @classmethod
    def load_season(
        cls,
        data_type: DATA_SRC_TYPE = "nbastats",
        season: int = 2022,
        playoffs: bool = False,
    ) -> pd.DataFrame:
        """Load a single season from the local archive.

        Parameters
        ----------
        data_type : str
            One of 'nbastats', 'nbastatsv3', 'datanba', 'pbpstats',
            'shotdetail', 'cdnnba', 'matchups'.
        season : int
            The year the season started (e.g. 2022 = 2022-23 season).
        playoffs : bool
            If True, load playoff data instead of regular season.
        """
        path = cls._tar_path(data_type, season, playoffs)
        if not path.exists():
            available = sorted(cls.DATA_DIR.glob(f"{data_type}_*.tar.xz"))
            raise FileNotFoundError(f"Dataset not found: {path}\nAvailable: {available}")
        return cls._read_tar_csv(path)

    @classmethod
    def load_seasons(
        cls,
        data_type: DATA_SRC_TYPE = "nbastats",
        seasons: SeasonsArg = None,
        playoffs: bool = False,
        skip_missing: bool = False,
    ) -> pd.DataFrame:
        """Load and concatenate multiple seasons into one DataFrame.

        Parameters
        ----------
        data_type : str
            Dataset type (e.g. 'nbastats', 'cdnnba', 'datanba', ...).
        seasons : int, sequence of int, or None
            Seasons to load. ``None`` loads all locally available seasons.
            A negative int ``-N`` loads the last N available seasons.
        playoffs : bool
            If True, load playoff data instead of regular season.
        skip_missing : bool
            Silently skip seasons whose files don't exist.
        """
        season_list = cls._resolve_seasons(data_type, seasons, playoffs)
        print(f"Loading data_type='{data_type}', seasons={season_list}, playoffs={playoffs}...")

        frames = []
        for s in season_list:
            if skip_missing and not cls._tar_path(data_type, s, playoffs).exists():
                continue
            frames.append(cls.load_season(data_type, s, playoffs))

        if not frames:
            raise ValueError(f"No data loaded for data_type='{data_type}', seasons={season_list}, playoffs={playoffs}.")
        return pd.concat(frames, ignore_index=True)

    @classmethod
    def load_game_dates(
        cls,
        seasons: SeasonsArg = None,
        playoffs: bool | None = False,
        skip_missing: bool = False,
    ) -> pd.DataFrame:
        """Return a GAME_ID -> game_date lookup table.

        Sources used in priority order:
        - pbpstats  (2000-2024): has a GAMEDATE column directly.
        - datanba   (2016-2024): derives date from the wallclk UTC timestamp.

        Parameters
        ----------
        seasons : int, sequence of int, or None
            Seasons to include. ``None`` loads all pbpstats-available seasons.
            A negative int ``-N`` loads the last N available seasons.
        playoffs : bool or None
            False = regular season only, True = playoffs only,
            None  = both combined.
        skip_missing : bool
            Silently skip seasons with no local file rather than raising.

        Returns
        -------
        pd.DataFrame with columns: GAME_ID (int), game_date (datetime.date)
        """
        base_playoffs = False if playoffs is None else bool(playoffs)
        season_list = cls._resolve_seasons("pbpstats", seasons, base_playoffs)
        season_types: list[bool] = [False, True] if playoffs is None else [bool(playoffs)]

        frames: list[pd.DataFrame] = []

        for stype in season_types:
            for s in season_list:
                # --- pbpstats: GAMEDATE column directly ---
                pbp_path = cls._tar_path("pbpstats", s, stype)
                if pbp_path.exists():
                    chunk = cls._read_tar_csv(pbp_path, usecols=["GAMEID", "GAMEDATE"])
                    chunk = (
                        chunk.drop_duplicates("GAMEID")
                        .rename(columns={"GAMEID": "GAME_ID", "GAMEDATE": "game_date"})
                        .assign(game_date=lambda d: pd.to_datetime(d["game_date"]).dt.date)
                    )
                    frames.append(chunk)
                    continue

                # --- datanba fallback: parse wallclk ---
                dn_path = cls._tar_path("datanba", s, stype)
                if dn_path.exists():
                    chunk = cls._read_tar_csv(dn_path, usecols=["GAME_ID", "wallclk"])
                    chunk = (
                        chunk.drop_duplicates("GAME_ID")
                        .assign(game_date=lambda d: pd.to_datetime(d["wallclk"], format="ISO8601", utc=True).dt.date)
                        .drop(columns="wallclk")
                    )
                    frames.append(chunk)
                    continue

                if not skip_missing:
                    raise FileNotFoundError(
                        f"No pbpstats or datanba file found for season={s}, playoffs={stype}. "
                        f"Pass skip_missing=True to ignore missing seasons."
                    )

        if not frames:
            raise ValueError(f"No game date data loaded for seasons={season_list}, playoffs={playoffs}.")

        return (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates("GAME_ID")
            .sort_values("game_date")
            .reset_index(drop=True)
        )

    @classmethod
    def load_game_dates_from_api(
        cls,
        seasons: SeasonsArg = None,
        playoffs: bool = False,
        sleep: float = 0.6,
    ) -> pd.DataFrame:
        """Fetch GAME_ID -> game_date via nba_api.LeagueGameLog.

        Intended for seasons not covered by local files (i.e. 1996-1999),
        but works for any season range. Requires a network connection to
        stats.nba.com — run this on your Mac, not in the Cowork VM.

        Results can be cached and merged into load_game_dates() output:

            early = NBA.load_game_dates_from_api(seasons=[1996, 1997, 1998, 1999])
            early.to_parquet(NBA.DATA_DIR / "game_dates_1996_1999.parquet", index=False)

            # Later, merge with the local lookup:
            dates = pd.concat([
                NBA.load_game_dates(),
                pd.read_parquet(NBA.DATA_DIR / "game_dates_1996_1999.parquet"),
            ]).drop_duplicates("GAME_ID").sort_values("game_date").reset_index(drop=True)

        Parameters
        ----------
        seasons : int, sequence of int, or None
            Seasons to fetch. ``None`` fetches all seasons available in the
            local nbastats files (since that is the play-by-play source).
            A negative int ``-N`` fetches the last N available seasons.
        playoffs : bool
            False = Regular Season, True = Playoffs.
        sleep : float
            Seconds to wait between API calls to respect rate limits.
            Default 0.6 s keeps well within stats.nba.com limits.

        Returns
        -------
        pd.DataFrame with columns: GAME_ID (int), game_date (datetime.date)
        """

        SEASON_TYPE = {False: "Regular Season", True: "Playoffs"}

        season_list = cls._resolve_seasons("nbastats", seasons, playoffs)

        frames: list[pd.DataFrame] = []
        for s in season_list:
            # NBA season string format: "1996-97", "2022-23", etc.
            season_str = f"{s}-{str(s + 1)[-2:]}"
            print(f"  Fetching {season_str} ({SEASON_TYPE[playoffs]})...", end=" ", flush=True)
            try:
                log = LeagueGameLog(
                    season=season_str,
                    season_type_all_star=SEASON_TYPE[playoffs],
                )
                df = log.get_data_frames()[0][["GAME_ID", "GAME_DATE"]]
                chunk = (
                    df.drop_duplicates("GAME_ID")
                    .rename(columns={"GAME_DATE": "game_date"})
                    .assign(
                        GAME_ID=lambda d: d["GAME_ID"].astype(int),
                        game_date=lambda d: pd.to_datetime(d["game_date"]).dt.date,
                    )
                )
                frames.append(chunk)
                print(f"{len(chunk)} games")
            except Exception as e:
                print(f"FAILED ({e})")
            time.sleep(sleep)

        if not frames:
            raise ValueError(f"No data fetched for seasons={season_list}, playoffs={playoffs}.")

        return (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates("GAME_ID")
            .sort_values("game_date")
            .reset_index(drop=True)
        )

    @staticmethod
    def get_timeouts(df: pd.DataFrame) -> pd.DataFrame:
        """Filter a nbastats DataFrame to timeout events only (EVENTMSGTYPE == 9)."""
        return df[df["EVENTMSGTYPE"] == 9].copy()

    # ------------------------------------------------------------------
    # API helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _season_int_to_str(season: int) -> str:
        """Convert integer season (e.g. 2024) to NBA API string ('2024-25')."""
        return f"{season}-{str(season + 1)[-2:]}"

    @staticmethod
    def _api_call_with_retry(fn, retries: int = 1, wait: float = 30.0, **kwargs) -> pd.DataFrame:
        """Call an nba_api endpoint with retry logic. Returns empty DataFrame on failure."""
        for attempt in range(1 + retries):
            try:
                result = fn(**kwargs)
                dfs = result.get_data_frames()
                return dfs[0] if dfs else pd.DataFrame()
            except (KeyError, ValueError):
                return pd.DataFrame()
            except Exception as e:
                if attempt < retries:
                    print(f"    Retry after error: {e}")
                    time.sleep(wait)
                else:
                    print(f"    FAILED: {e}")
                    return pd.DataFrame()
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Fetch auxiliary data from NBA API
    # ------------------------------------------------------------------

    @classmethod
    def fetch_player_season_stats(
        cls,
        seasons: SeasonsArg = None,
        out_path: Path | str | None = None,
        throttle: float = 1.5,
    ) -> pd.DataFrame:
        """Fetch per-player season totals via PlayerCareerStats.

        Calls the API once per unique player across all seasons of the
        cdnnba PBP data. Saves to parquet.

        Parameters
        ----------
        seasons : int, sequence of int, or None
            Seasons to include. None = all cdnnba seasons.
        out_path : Path or None
            Where to save. Defaults to NBA_DATA_DIR / "player_season_stats.parquet".
        throttle : float
            Seconds between API calls.
        """
        out_path = Path(out_path) if out_path else NBAConstants.NBA_DATA_DIR / "player_season_stats.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        season_list = cls._resolve_seasons("cdnnba", seasons, playoffs=False)
        season_strs = {cls._season_int_to_str(s) for s in season_list}

        # Get unique player IDs from the stacked PBP
        pbp_path = NBAConstants.NBA_DATA_DIR / "cdnnba_stacked.parquet"
        print(f"Reading player IDs from {pbp_path}...")
        pbp = pd.read_parquet(pbp_path, columns=["personId"])
        player_ids = sorted(pbp["personId"].dropna().unique())
        player_ids = [pid for pid in player_ids if pid > 0]
        print(f"  {len(player_ids)} unique players")

        # Load existing output + checkpoint if they exist
        checkpoint_path = out_path.parent / ".checkpoint_player_season_stats.parquet"
        done_ids: set[int] = set()
        frames: list[pd.DataFrame] = []
        if out_path.exists():
            existing = pd.read_parquet(out_path)
            frames.append(existing)
            done_ids = set(existing["PLAYER_ID"].unique())
            print(f"  Existing output: {len(done_ids)} players already saved")
        if checkpoint_path.exists():
            ckpt = pd.read_parquet(checkpoint_path)
            frames.append(ckpt)
            done_ids |= set(ckpt["PLAYER_ID"].unique())
            print(f"  Resuming from checkpoint: {len(done_ids)} total players done")

        remaining = [pid for pid in player_ids if pid not in done_ids]
        print(f"  Fetching {len(remaining)} remaining players...")

        for i, pid in enumerate(remaining):
            try:
                result = playercareerstats.PlayerCareerStats(player_id=str(int(pid)))
                dfs = result.get_data_frames()

                rs = dfs[0] if len(dfs) > 0 else pd.DataFrame()  # SeasonTotalsRegularSeason
                po = dfs[3] if len(dfs) > 3 else pd.DataFrame()  # SeasonTotalsPostSeason

                if not rs.empty:
                    rs = rs.assign(season_type="rg")
                if not po.empty:
                    po = po.assign(season_type="po")

                combined = pd.concat([rs, po], ignore_index=True)
                if not combined.empty and "SEASON_ID" in combined.columns:
                    combined["season_int"] = combined["SEASON_ID"].str[:4].astype(int)
                    combined = combined[combined["SEASON_ID"].isin(season_strs)]
                    if not combined.empty:
                        frames.append(combined)
            except (KeyError, ValueError):
                pass
            except Exception as e:
                print(f"    Player {pid} failed: {e}")

            if (i + 1) % 100 == 0:
                print(f"    {i + 1}/{len(remaining)} done")
                if frames:
                    pd.concat(frames, ignore_index=True).to_parquet(checkpoint_path, index=False)

            time.sleep(throttle)

        if not frames:
            raise ValueError("No player stats fetched.")

        result_df = pd.concat(frames, ignore_index=True).drop_duplicates(
            subset=["PLAYER_ID", "SEASON_ID", "season_type"]
        )
        result_df.to_parquet(out_path, index=False, compression="zstd")
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        print(f"Saved player_season_stats: {len(result_df):,} rows -> {out_path}")
        return result_df

    @classmethod
    def fetch_player_advanced_stats(
        cls,
        seasons: SeasonsArg = None,
        out_path: Path | str | None = None,
        throttle: float = 1.5,
    ) -> pd.DataFrame:
        """Fetch league-wide advanced/tracking stats per season.

        Calls 8 endpoints per season:
        - PlayerEstimatedMetrics
        - LeagueDashPlayerBioStats
        - LeagueHustleStatsPlayer
        - LeagueDashPtStats (Drives, CatchShoot, PullUpShot, Passing, Defense, SpeedDistance)

        Merges all on (PLAYER_ID, season_int). Saves to parquet.
        """
        out_path = Path(out_path) if out_path else NBAConstants.NBA_DATA_DIR / "player_advanced_stats.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        season_list = cls._resolve_seasons("cdnnba", seasons, playoffs=False)
        TRACKING_MEASURES = ["Drives", "CatchShoot", "PullUpShot", "Passing", "Defense", "SpeedDistance"]

        all_groups: dict[str, list[pd.DataFrame]] = {
            "bio": [],
            "em": [],
            "hustle": [],
        }
        for mt in TRACKING_MEASURES:
            all_groups[mt.lower()] = []

        for s in season_list:
            season_str = cls._season_int_to_str(s)
            print(f"  Fetching advanced stats for {season_str}...")

            # PlayerEstimatedMetrics
            df = cls._api_call_with_retry(
                playerestimatedmetrics.PlayerEstimatedMetrics,
                season=season_str,
                season_type="Regular Season",
            )
            if not df.empty:
                df = df.drop(
                    columns=[c for c in df.columns if c.endswith("_RANK")] + ["W", "L", "W_PCT", "GP", "MIN"],
                    errors="ignore",
                )
                df["season_int"] = s
                all_groups["em"].append(df)
            time.sleep(throttle)

            # LeagueDashPlayerBioStats
            df = cls._api_call_with_retry(
                leaguedashplayerbiostats.LeagueDashPlayerBioStats,
                season=season_str,
                season_type_all_star="Regular Season",
                per_mode_simple="PerGame",
            )
            if not df.empty:
                df["season_int"] = s
                all_groups["bio"].append(df)
            time.sleep(throttle)

            # LeagueHustleStatsPlayer
            df = cls._api_call_with_retry(
                leaguehustlestatsplayer.LeagueHustleStatsPlayer,
                season=season_str,
                season_type_all_star="Regular Season",
                per_mode_time="PerGame",
            )
            if not df.empty:
                df = df.drop(columns=["TEAM_ID", "TEAM_ABBREVIATION", "AGE", "G", "MIN"], errors="ignore")
                df["season_int"] = s
                all_groups["hustle"].append(df)
            time.sleep(throttle)

            # LeagueDashPtStats (6 tracking types)
            for mt in TRACKING_MEASURES:
                df = cls._api_call_with_retry(
                    leaguedashptstats.LeagueDashPtStats,
                    season=season_str,
                    season_type_all_star="Regular Season",
                    per_mode_simple="PerGame",
                    player_or_team="Player",
                    pt_measure_type=mt,
                )
                if not df.empty:
                    df = df.drop(columns=["TEAM_ID", "TEAM_ABBREVIATION", "GP", "W", "L", "MIN"], errors="ignore")
                    df["season_int"] = s
                    all_groups[mt.lower()].append(df)
                time.sleep(throttle)

        # Concat each group across seasons
        merged_groups: dict[str, pd.DataFrame] = {}
        for label, dfs in all_groups.items():
            if dfs:
                merged_groups[label] = pd.concat(dfs, ignore_index=True)

        if "bio" not in merged_groups:
            raise ValueError("No bio stats fetched — cannot build advanced stats.")

        # Start with bio as base, outer-merge everything else
        result = merged_groups.pop("bio")
        merge_keys = ["PLAYER_ID", "season_int"]

        for label, df in merged_groups.items():
            # Drop columns that already exist in result (except merge keys and player name)
            existing = set(result.columns) - set(merge_keys)
            drop_cols = [c for c in df.columns if c in existing and c != "PLAYER_NAME"]
            if "PLAYER_NAME" in df.columns and "PLAYER_NAME" in result.columns:
                df = df.rename(columns={"PLAYER_NAME": f"PLAYER_NAME_{label}"})
            df = df.drop(columns=drop_cols, errors="ignore")
            result = result.merge(df, on=merge_keys, how="outer", suffixes=("", f"_{label}"))

        result.to_parquet(out_path, index=False, compression="zstd")
        print(f"Saved player_advanced_stats: {len(result):,} rows x {len(result.columns)} cols -> {out_path}")
        return result

    @classmethod
    def fetch_rotations(
        cls,
        out_path: Path | str | None = None,
        throttle: float = 1.5,
    ) -> pd.DataFrame:
        """Fetch player rotation data (in/out times) for all games in the stacked PBP.

        Calls GameRotation endpoint per game. Saves to parquet.
        """
        out_path = Path(out_path) if out_path else NBAConstants.NBA_DATA_DIR / "rotations.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Get unique game IDs
        pbp_path = NBAConstants.NBA_DATA_DIR / "cdnnba_stacked.parquet"
        print(f"Reading game IDs from {pbp_path}...")
        game_ids = sorted(pd.read_parquet(pbp_path, columns=["gameId"])["gameId"].unique())
        print(f"  {len(game_ids)} unique games")

        # Load checkpoint
        checkpoint_path = out_path.parent / ".checkpoint_rotations.parquet"
        done_ids: set[int] = set()
        frames: list[pd.DataFrame] = []
        if checkpoint_path.exists():
            ckpt = pd.read_parquet(checkpoint_path)
            frames.append(ckpt)
            done_ids = set(ckpt["gameId"].unique())
            print(f"  Resuming from checkpoint: {len(done_ids)} games done")
        if out_path.exists():
            existing = pd.read_parquet(out_path)
            frames.insert(0, existing)
            done_ids |= set(existing["gameId"].unique())
            print(f"  {len(done_ids)} games already saved")

        remaining = [gid for gid in game_ids if gid not in done_ids]
        print(f"  Fetching rotations for {len(remaining)} remaining games...")

        for i, gid in enumerate(remaining):
            gid_str = f"{gid:010d}"
            try:
                result = gamerotation.GameRotation(game_id=gid_str)
                dfs = result.get_data_frames()
                batch = []
                for idx, label in enumerate(["away", "home"]):
                    if idx < len(dfs) and not dfs[idx].empty:
                        chunk = dfs[idx].copy()
                        chunk["location"] = label
                        chunk["gameId"] = gid
                        batch.append(chunk)
                if batch:
                    frames.append(pd.concat(batch, ignore_index=True))
            except Exception as e:
                print(f"    Game {gid} failed: {e}")

            if (i + 1) % 50 == 0:
                print(f"    {i + 1}/{len(remaining)} done")
                if frames:
                    pd.concat(frames, ignore_index=True).to_parquet(checkpoint_path, index=False)

            time.sleep(throttle)

        if not frames:
            raise ValueError("No rotation data fetched.")

        result_df = pd.concat(frames, ignore_index=True)
        result_df.to_parquet(out_path, index=False, compression="zstd")
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        print(f"Saved rotations: {len(result_df):,} rows -> {out_path}")
        return result_df
