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
9  Timeout  ← key event for this project
10 Jump ball
11 Ejection
12 Start of period
13 End of period
18 Instant replay
"""

from pathlib import Path
import tarfile
import pandas as pd
import typing as t


# Resolve paths whether running on Mac or in the Cowork VM
def _resolve_columbia_dir() -> Path:
    candidates = [
        Path("/Users/Akseldkw/coding/Columbia"),  # Mac
        Path("/sessions/busy-determined-hypatia/mnt/Columbia"),  # Cowork VM
    ]
    for p in candidates:
        if p.exists():
            return p
    raise RuntimeError(
        "Cannot find Columbia directory. " "Expected at /Users/Akseldkw/coding/Columbia or the Cowork VM mount."
    )


_COLUMBIA = _resolve_columbia_dir()
NBA_DATA_DIR = _COLUMBIA / "nba_data" / "datasets"
NBA_ON_COURT_DIR = _COLUMBIA / "nba-on-court"
DATA_SRC_TYPE = t.Literal["nbastats", "nbastatsv3", "datanba", "pbpstats", "shotdetail", "cdnnba", "matchups"]


def _tar_path(data_type: str, season: int, playoffs: bool = False) -> Path:
    if playoffs:
        name = f"{data_type}_po_{season}.tar.xz"
    else:
        name = f"{data_type}_{season}.tar.xz"
    return NBA_DATA_DIR / name


def load_season(
    data_type: DATA_SRC_TYPE = "nbastats",
    season: int = 2022,
    playoffs: bool = False,
) -> pd.DataFrame:
    """
    Load a single season of NBA play-by-play data from the local archive.

    Parameters
    ----------
    data_type : str
        One of 'nbastats', 'nbastatsv3', 'datanba', 'pbpstats',
        'shotdetail', 'cdnnba', 'matchups'.
    season : int
        The year the season *started* (e.g. 2022 = 2022-23 season).
    playoffs : bool
        If True, load playoff data instead of regular season.

    Returns
    -------
    pd.DataFrame
    """
    path = _tar_path(data_type, season, playoffs)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n" f"Available files: {sorted(NBA_DATA_DIR.glob(f'{data_type}_*.tar.xz'))}"
        )
    csv_name = path.stem.replace(".tar", "") + ".csv"  # strip .tar from stem
    # stem of "nbastats_2022.tar.xz" is "nbastats_2022.tar", so:
    csv_name = path.name.replace(".tar.xz", ".csv")
    with tarfile.open(path, "r:xz") as tar:
        f = tar.extractfile(csv_name)
        assert f is not None, f"CSV file {csv_name} not found in archive {path}"
        return pd.read_csv(f)


def load_seasons(
    data_type: DATA_SRC_TYPE = "nbastats",
    seasons: int | t.Sequence[int] | None = None,
    playoffs: bool = False,
    skip_missing: bool = False,
) -> pd.DataFrame:
    """Load and concatenate multiple seasons into one DataFrame.

    Parameters
    ----------
    data_type : str
        Dataset type (e.g. 'nbastats', 'cdnnba', 'datanba', ...).
    seasons : int, sequence of int, or None
        Seasons to load. If None (default), automatically loads all
        locally available seasons for the given data_type and playoffs flag.
    playoffs : bool
        If True, load playoff data instead of regular season.
    skip_missing : bool
        If True, silently skip seasons whose files don't exist rather than
        raising FileNotFoundError. Useful when passing a broad range.

    Returns
    -------
    pd.DataFrame
    """
    if seasons is None:
        seasons = available_seasons(data_type, playoffs)
        if not seasons:
            raise FileNotFoundError(
                f"No local files found for data_type='{data_type}', playoffs={playoffs}.\n"
                f"Expected files in: {NBA_DATA_DIR}"
            )
    elif isinstance(seasons, int):
        seasons = [seasons]

    frames = []
    for s in seasons:
        if skip_missing and not _tar_path(data_type, s, playoffs).exists():
            continue
        frames.append(load_season(data_type, s, playoffs))

    if not frames:
        raise ValueError(f"No data loaded for data_type='{data_type}', seasons={list(seasons)}, playoffs={playoffs}.")
    return pd.concat(frames, ignore_index=True)


def get_timeouts(df: pd.DataFrame) -> pd.DataFrame:
    """Filter play-by-play DataFrame to timeout events only (EVENTMSGTYPE == 9)."""
    return df[df["EVENTMSGTYPE"] == 9].copy()


def available_seasons(data_type: DATA_SRC_TYPE = "nbastats", playoffs: bool = False) -> list[int]:
    """Return sorted list of locally available seasons for a given data type."""
    if playoffs:
        pattern = f"{data_type}_po_*.tar.xz"
        prefix = f"{data_type}_po_"
    else:
        pattern = f"{data_type}_[0-9]*.tar.xz"
        prefix = f"{data_type}_"
    files = sorted(NBA_DATA_DIR.glob(pattern))
    seasons = []
    for f in files:
        stem = f.name.replace(".tar.xz", "")
        try:
            year = int(stem.replace(prefix, ""))
            seasons.append(year)
        except ValueError:
            pass
    return seasons
