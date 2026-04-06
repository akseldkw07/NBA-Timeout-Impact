"""
Pipeline for loading, stacking, and cleaning cdn.nba.com play-by-play data.

Two-step workflow:
    1. load_and_stack_all  – load all cdnnba seasons (regular + playoff),
                             add an IsPlayoff column, save to parquet.
    2. clean_stacked       – load the stacked parquet, run the cleaning
                             pipeline, save the cleaned result.

Usage:
    from nba_timeout_impact.data_pipes.cdnnba_pipeline import CDNNBAPipelineHelper

    CDNNBAPipelineHelper.load_and_stack_all()
    CDNNBAPipelineHelper.clean_stacked()
"""

from pathlib import Path

import numpy as np
import pandas as pd
from kret_np_pd.UTILS_np_pd import NP_PD_Utils
from kret_sklearn.custom_transformers import PandasColumnOrderBase
from kret_sklearn.pd_pipeline import PipelinePD
from sklearn.preprocessing import FunctionTransformer

from nba_timeout_impact.constants import NBAConstants
from nba_timeout_impact.data_pipes.load_data_utils import NBADataLoader

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OUT_DIR = NBAConstants.NBA_DATA_DIR
_STACKED_PATH = _OUT_DIR / "cdnnba_stacked.parquet"
_CLEAN_PATH = _OUT_DIR / "cdnnba_clean.parquet"
_ENRICHED_PATH = _OUT_DIR / "cdnnba_enriched.parquet"
_STINTS_PATH = _OUT_DIR / "stints.parquet"

# Halftime and OT boundaries in tenths-of-seconds (rotation API units)
# 14400 = halftime, 28800 = end regulation, 31800/34800/... = OT boundaries
_BREAK_POINTS = [14400, 28800, 31800, 34800, 37800, 40800]

# Columns that only appear in some seasons — drop for consistency
_DROP_COLS = {"value", "jerseyNumber", "shortFormattedClock"}

# Float columns that are actually whole numbers — cast to (nullable) int
_FLOAT_TO_INT_COLS = [
    "teamId",
    "pointsTotal",
    "jumpBallWonPersonId",
    "jumpBallLostPersonId",
    "jumpBallRecoverdPersonId",
    "officialId",
    "turnoverTotal",
    "foulPersonalTotal",
    "foulTechnicalTotal",
    "foulDrawnPersonId",
    "assistPersonId",
    "assistTotal",
    "shotActionNumber",
    "reboundTotal",
    "reboundDefensiveTotal",
    "reboundOffensiveTotal",
    "blockPersonId",
    "stealPersonId",
    "xLegacy",
    "yLegacy",
]

# String columns that are ISO 8601 datetimes
_DATETIME_COLS = ["timeActual", "edited"]

# Columns to convert to category
_CAT_COLS = [
    "actionType",
    "subType",
    "periodType",
    "playerName",
    "playerNameI",
    "teamTricode",
    "shotResult",
    "side",
    "area",
    "areaDetail",
]

# Desired column ordering after cleaning
_START_COLS = [
    "game_date",
    "gameId",
    "orderNumber",
    "actionType",
    "subType",
    "description",
    "scoreHome",
    "scoreAway",
    "pointsTotal",
    "possession",
    "period",
    "game_seconds_elapsed",
    "seconds_remaining",
    "seconds_elapsed",
    "IsPlayoff",
]

_END_COLS = [
    "playerNameI",
    "actionNumber",
    "clock",
    "timeActual",
    "teamId",
    "personId",
    "playerName",
    "xLegacy",
    "yLegacy",
    "x",
    "y",
    "shotDistance",
    "videoAvailable",
]


# ---------------------------------------------------------------------------
# Transformers
# ---------------------------------------------------------------------------


class StripStringColumns(PandasColumnOrderBase):
    """Strip leading/trailing whitespace from every object/string column."""

    def _fit(self, X: pd.DataFrame, y=None) -> "StripStringColumns":
        # Only strip columns whose non-null values are actually strings
        self._str_cols = [
            col
            for col in X.select_dtypes(include=["object", "string"]).columns
            if X[col].dropna().head(100).apply(type).eq(str).all()  # type: ignore
        ]
        return self

    def _transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        for col in self._str_cols:
            X[col] = X[col].str.strip()
        self.new_columns = list(X.columns)
        return X


class CDNNBAClockParser(PandasColumnOrderBase):
    """
    Parse the ISO 8601 ``clock`` column (e.g. 'PT04M56.00S') into:

    seconds_remaining  – time left in the period
    seconds_elapsed    – time played so far in the period
    """

    REG_PERIOD_SECS = 720
    OT_PERIOD_SECS = 300

    def _fit(self, X: pd.DataFrame, y=None) -> "CDNNBAClockParser":
        assert "clock" in X.columns
        assert "period" in X.columns
        return self

    @staticmethod
    def _parse_clock(series: pd.Series) -> pd.Series:
        parsed = series.str.extract(r"PT(\d+)M([\d.]+)S").astype(float)
        return parsed[0] * 60 + parsed[1]

    def _transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        period_len = np.where(X["period"] <= 4, self.REG_PERIOD_SECS, self.OT_PERIOD_SECS)
        X["seconds_remaining"] = self._parse_clock(X["clock"])
        X["seconds_elapsed"] = period_len - X["seconds_remaining"]
        self.new_columns = list(X.columns)
        return X


class GameTimeElapsed(PandasColumnOrderBase):
    """
    Add ``game_seconds_elapsed`` — absolute seconds from tip-off.

    Regulation periods 1-4 : each 720 s  (12 min)
    OT periods 5, 6, 7 …  : each 300 s  (5 min)
    """

    REG_PERIOD_SECS = 720
    OT_PERIOD_SECS = 300
    REG_PERIODS = 4

    def _fit(self, X: pd.DataFrame, y=None) -> "GameTimeElapsed":
        assert "period" in X.columns
        assert "seconds_elapsed" in X.columns
        return self

    def _transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        reg_offset = np.minimum(X["period"] - 1, self.REG_PERIODS) * self.REG_PERIOD_SECS
        ot_offset = np.maximum(X["period"] - self.REG_PERIODS - 1, 0) * self.OT_PERIOD_SECS
        X["game_seconds_elapsed"] = reg_offset + ot_offset + X["seconds_elapsed"]
        self.new_columns = list(X.columns)
        return X


class DeriveGameDate(PandasColumnOrderBase):
    """
    Derive ``game_date`` from the ``timeActual`` wall-clock timestamp.
    No external date file needed — cdnnba embeds real timestamps.
    """

    def _fit(self, X: pd.DataFrame, y=None) -> "DeriveGameDate":
        assert "timeActual" in X.columns
        return self

    def _transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        ts = pd.to_datetime(X["timeActual"], format="ISO8601", utc=True)
        # NBA games that tip off late evening (ET) can cross midnight UTC.
        # Use the date of the first event per game to tag every row.
        first_date = ts.groupby(X["gameId"]).transform("first").dt.date
        X["game_date"] = pd.to_datetime(pd.Series(first_date, index=X.index))
        self.new_columns = list(X.columns)
        return X


class SortByDate(PandasColumnOrderBase):
    """Sort by game_date, gameId, orderNumber."""

    sort_cols = ["game_date", "gameId", "orderNumber"]

    def _fit(self, X: pd.DataFrame, y=None) -> "SortByDate":
        for col in self.sort_cols:
            assert col in X.columns, f"Missing column: {col}"
        return self

    def _transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.sort_values(self.sort_cols).reset_index(drop=True)
        self.new_columns = list(X.columns)
        return X


class CategoricalConversion(PandasColumnOrderBase):
    cols: list[str]

    def __init__(self, cols: list[str]) -> None:
        super().__init__()
        self.cols = cols

    def _fit(self, X: pd.DataFrame, y=None) -> "CategoricalConversion":
        return self

    def _transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        for col in self.cols:
            if col in X.columns:
                X[col] = pd.Categorical(X[col])
        self.new_columns = list(X.columns)
        return X


class EnforceMonotonicity(PandasColumnOrderBase):
    """Enforce non-decreasing scores within each game via cummax."""

    cols: list[str]

    def __init__(self, cols: list[str]) -> None:
        super().__init__()
        self.cols = cols

    def _fit(self, X: pd.DataFrame, y=None) -> "EnforceMonotonicity":
        return self

    def _transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        for col in self.cols:
            X[col] = X.groupby("gameId")[col].cummax()
        self.new_columns = list(X.columns)
        return X


class CastFloatToInt(PandasColumnOrderBase):
    """Cast float64 columns that only contain whole numbers to nullable Int64.

    Columns with NaN are cast to pd.Int64Dtype() (nullable integer).
    Columns without NaN are cast to plain int64.
    """

    cols: list[str]

    def __init__(self, cols: list[str]) -> None:
        super().__init__()
        self.cols = cols

    def _fit(self, X: pd.DataFrame, y=None) -> "CastFloatToInt":
        return self

    def _transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        for col in self.cols:
            if col not in X.columns:
                continue
            if X[col].isna().any():
                X[col] = X[col].astype(pd.Int64Dtype())
            else:
                X[col] = X[col].astype(np.int64)
        self.new_columns = list(X.columns)
        return X


class ParseDatetimeColumns(PandasColumnOrderBase):
    """Parse string columns to datetime."""

    cols: list[str]

    def __init__(self, cols: list[str]) -> None:
        super().__init__()
        self.cols = cols

    def _fit(self, X: pd.DataFrame, y=None) -> "ParseDatetimeColumns":
        return self

    def _transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        for col in self.cols:
            if col not in X.columns:
                continue
            X[col] = pd.to_datetime(X[col], format="ISO8601", utc=True)
        self.new_columns = list(X.columns)
        return X


# ---------------------------------------------------------------------------
# Pipeline helper
# ---------------------------------------------------------------------------


class CDNNBAPipelineHelper:

    @classmethod
    def load_and_stack_all(cls, out_path: str | Path = _STACKED_PATH) -> pd.DataFrame:
        """Load all cdnnba seasons (regular + playoff), stack, save to parquet."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        frames: list[pd.DataFrame] = []
        for playoffs in (False, True):
            seasons = NBADataLoader.available_seasons("cdnnba", playoffs=playoffs)
            for s in seasons:
                df = NBADataLoader.load_season("cdnnba", s, playoffs=playoffs)
                df["IsPlayoff"] = playoffs
                frames.append(df)
                print(f"  Loaded {'PO' if playoffs else 'RS'} {s}: {len(df):,} rows, {len(df.columns)} cols")

        stacked = pd.concat(frames, ignore_index=True)

        # Drop columns that only appear in some seasons
        drop = _DROP_COLS & set(stacked.columns)
        if drop:
            stacked = stacked.drop(columns=list(drop))
            print(f"  Dropped inconsistent columns: {drop}")

        stacked.to_parquet(out_path, index=False, compression="zstd")
        print(f"\nSaved stacked: {len(stacked):,} rows x {len(stacked.columns)} cols -> {out_path}")
        return stacked

    @classmethod
    def clean_stacked(cls, in_path: str | Path = _STACKED_PATH, out_path: str | Path = _CLEAN_PATH) -> pd.DataFrame:
        """Load stacked parquet, run cleaning pipeline, save cleaned result."""
        in_path, out_path = Path(in_path), Path(out_path)

        score_cols = ["scoreHome", "scoreAway"]
        # Only include end cols that will actually be present
        end_cols = [c for c in _END_COLS if c != "videoAvailable"]

        sort_cols_fn = FunctionTransformer(
            func=NP_PD_Utils.move_columns, validate=False, kw_args={"start": _START_COLS, "end": end_cols}
        )

        pipeline = PipelinePD(
            steps=[
                ("strip_strings", StripStringColumns()),
                ("parse_clock", CDNNBAClockParser()),
                ("game_time", GameTimeElapsed()),
                ("derive_date", DeriveGameDate()),
                ("sort", SortByDate()),
                ("cats", CategoricalConversion(_CAT_COLS)),
                ("enforce_monotonicity", EnforceMonotonicity(score_cols)),
                ("cast_int", CastFloatToInt(_FLOAT_TO_INT_COLS)),
                ("parse_datetimes", ParseDatetimeColumns(_DATETIME_COLS)),
                ("sort_cols", sort_cols_fn),
            ]
        )

        print(f"Loading stacked data from {in_path}...")
        df = pd.read_parquet(in_path)
        print(f"  {len(df):,} rows x {len(df.columns)} cols")

        print("Running cleaning pipeline...")
        clean = pipeline.fit_transform_df(df)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        clean.to_parquet(out_path, index=False, compression="zstd")
        print(f"\nSaved clean: {len(clean):,} rows x {len(clean.columns)} cols -> {out_path}")
        return clean

    # ------------------------------------------------------------------
    # Step 3: Enrich cleaned PBP
    # ------------------------------------------------------------------

    @classmethod
    def enrich(cls, in_path: str | Path = _CLEAN_PATH, out_path: str | Path = _ENRICHED_PATH) -> pd.DataFrame:
        """Add derived columns to cleaned PBP data.

        Adds:
        - season, season_type       — derived from gameId
        - shot_value, points_scored — 3/2/1/0 per event
        - score_margin, is_clutch   — convenience columns
        - prev_action_type          — lag of actionType within game
        - x_court, y_court          — half-court pixel coordinates
        - possession_id             — unique per possession per game
        - possession_outcome        — how each possession ended
        - possession_points         — total points per possession
        """
        in_path, out_path = Path(in_path), Path(out_path)
        print(f"Loading cleaned data from {in_path}...")
        df = pd.read_parquet(in_path)
        print(f"  {len(df):,} rows x {len(df.columns)} cols")

        print("Enriching...")

        # --- season / season_type from gameId ---
        # gameId as 8-digit int: TYYNNNNNN
        # T: 2=RS, 4=PO; YY: season last 2 digits; NNNNN: game number
        # e.g. 22000001 → T=2 (RS), YY=20 (2020), game 00001
        #      42400101 → T=4 (PO), YY=24 (2024), game 00101
        game_type_digit = df["gameId"] // 10000000  # first digit
        df["season_type"] = np.where(game_type_digit == 2, "rg", "po")
        df["season"] = 2000 + (df["gameId"] % 10000000) // 100000

        # --- shot_value, points_scored ---
        at = df["actionType"]
        sr = df["shotResult"]
        df["shot_value"] = np.select([at == "3pt", at == "2pt", at == "freethrow"], [3, 2, 1], default=pd.NA)  # type: ignore
        df["shot_value"] = df["shot_value"].astype(pd.Int64Dtype())

        df["points_scored"] = np.select(
            [(at == "3pt") & (sr == "Made"), (at == "2pt") & (sr == "Made"), (at == "freethrow") & (sr == "Made")],
            [3, 2, 1],
            default=0,
        ).astype(int)

        # --- score_margin, is_clutch ---
        df["score_margin"] = df["scoreHome"] - df["scoreAway"]
        df["is_clutch"] = (df["score_margin"].abs() <= 5) & (df["game_seconds_elapsed"] >= 2400.0)

        # --- prev_action_type ---
        game_boundary = df["gameId"].ne(df["gameId"].shift(fill_value=-1))
        df["prev_action_type"] = df["actionType"].shift(1)
        df.loc[game_boundary, "prev_action_type"] = pd.NA

        # --- x_court, y_court (half-court pixel coords) ---
        side = df["side"]
        x_raw = df["x"]
        y_raw = df["y"]

        df["x_court"] = pd.array(
            np.select(  # type: ignore
                [side == "right", side == "left"],
                [(5 * (y_raw - 50)).round(0), (-5 * (y_raw - 50)).round(0)],
                default=pd.NA,  # type: ignore
            ),
            dtype=pd.Int32Dtype(),
        )
        df["y_court"] = pd.array(
            np.select(  # type: ignore
                [side == "right", side == "left"],
                [(-9.4 * x_raw + 887.5).round(0), (9.4 * x_raw - 52.5).round(0)],
                default=pd.NA,  # type: ignore
            ),
            dtype=pd.Int32Dtype(),
        )

        # --- possession_id ---
        print("  Computing possession IDs...")
        prev_poss = df.groupby("gameId")["possession"].shift(1)
        poss_change = ((df["possession"] > 0) & (prev_poss.isna() | (df["possession"] != prev_poss))).astype(int)
        poss_id_raw = poss_change.groupby(df["gameId"]).cumsum()
        df["possession_id"] = poss_id_raw.where(df["possession"] > 0, other=pd.NA).astype(pd.Int64Dtype())  # type: ignore

        # --- possession_points ---
        print("  Computing possession points...")
        poss_pts = (
            df[df["possession_id"].notna()]
            .groupby(["gameId", "possession_id"])["points_scored"]
            .sum()
            .rename("possession_points")
        )
        df = df.merge(poss_pts, on=["gameId", "possession_id"], how="left")
        df["possession_points"] = df["possession_points"].astype(pd.Int64Dtype())

        # --- possession_outcome ---
        print("  Computing possession outcomes...")
        df["possession_outcome"] = cls._compute_possession_outcomes(df)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False, compression="zstd")
        print(f"\nSaved enriched: {len(df):,} rows x {len(df.columns)} cols -> {out_path}")
        return df

    @staticmethod
    def _compute_possession_outcomes(df: pd.DataFrame) -> pd.Series:
        """Classify how each possession ended. Fully vectorized — no groupby.apply().

        Returns a Series aligned with df's index. Rows with no possession_id get NA.
        """
        poss_key = ["gameId", "possession_id"]
        mask = df["possession_id"].notna()
        sub = df.loc[mask, poss_key + ["actionType", "subType", "shotResult", "points_scored"]].copy()

        at = sub["actionType"]
        sr = sub["shotResult"]

        # --- Per-row boolean flags ---
        sub["_is_steal"] = at == "steal"
        sub["_is_made_2"] = (at == "2pt") & (sr == "Made")
        sub["_is_made_3"] = (at == "3pt") & (sr == "Made")
        sub["_is_missed_fg"] = at.isin(["2pt", "3pt"]) & (sr == "Missed")
        sub["_is_block"] = at == "block"
        sub["_is_ft"] = at == "freethrow"
        sub["_is_ft_made"] = (at == "freethrow") & (sr == "Made")

        # --- Per-possession aggregations (vectorized groupby) ---
        g = sub.groupby(poss_key, sort=False)
        poss = pd.DataFrame(index=g.ngroup().drop_duplicates().index)

        # last event per possession
        last = g.tail(1).set_index(poss_key)
        agg = g.agg(
            has_steal=("_is_steal", "any"),
            has_made_2=("_is_made_2", "any"),
            has_made_3=("_is_made_3", "any"),
            has_missed_fg=("_is_missed_fg", "any"),
            has_block=("_is_block", "any"),
            ft_made=("_is_ft_made", "sum"),
            ft_total=("_is_ft", "sum"),
        )
        agg["last_action"] = last["actionType"]
        agg["last_sub"] = last["subType"]

        # --- Classify (vectorized, priority order: last assignment wins) ---
        outcome = pd.Series("other", index=agg.index)
        no_made = ~agg["has_made_2"] & ~agg["has_made_3"]

        # Missed FG that ends possession (no rebound in same possession, or blocked)
        outcome[no_made & agg["has_missed_fg"] & (agg["ft_total"] == 0)] = "miss"

        outcome[agg["last_action"] == "violation"] = "violation"
        outcome[agg["last_action"] == "period"] = "end_of_period"
        outcome[(agg["last_action"] == "rebound") & (agg["last_sub"] == "defensive")] = "miss_def_reb"
        outcome[(agg["last_action"] == "turnover") & ~agg["has_steal"]] = "turnover_dead"
        outcome[(agg["last_action"] == "turnover") & agg["has_steal"]] = "turnover_live"

        # Free throws only (no made FG)
        ft_only = ~agg["has_made_2"] & ~agg["has_made_3"] & (agg["ft_total"] > 0)
        if ft_only.any():
            ft_str = (
                "ft_"
                + agg.loc[ft_only, "ft_made"].astype(int).astype(str)
                + "_of_"
                + agg.loc[ft_only, "ft_total"].astype(int).astype(str)
            )
            outcome[ft_only] = ft_str

        # Made FG (no and-1)
        outcome[agg["has_made_2"] & (agg["ft_made"] == 0)] = "made_2pt"
        outcome[agg["has_made_3"] & (agg["ft_made"] == 0)] = "made_3pt"

        # And-1
        outcome[agg["has_made_2"] & (agg["ft_made"] > 0)] = "made_2pt_and1"
        outcome[agg["has_made_3"] & (agg["ft_made"] > 0)] = "made_3pt_and1"

        # --- Map back to original df index ---
        poss_map = outcome.reset_index()
        poss_map.columns = ["gameId", "possession_id", "possession_outcome"]  # type: ignore[assignment]
        merged = df[["gameId", "possession_id"]].merge(poss_map, on=["gameId", "possession_id"], how="left")
        merged.index = df.index
        return merged["possession_outcome"]

    # ------------------------------------------------------------------
    # Step 4: Build stints from rotation data
    # ------------------------------------------------------------------

    @classmethod
    def build_stints(
        cls, rotations_path: str | Path | None = None, out_path: str | Path = _STINTS_PATH
    ) -> pd.DataFrame:
        """Convert raw rotation data into player stints, split at halftime/OT boundaries.

        Each stint is a continuous stretch when a player was on the court.
        Stints that cross halftime or OT boundaries are split into sub-stints.
        """
        rotations_path = Path(rotations_path) if rotations_path else NBAConstants.NBA_DATA_DIR / "rotations.parquet"
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Loading rotations from {rotations_path}...")
        rot = pd.read_parquet(rotations_path)
        print(f"  {len(rot):,} rows")

        rows: list[dict] = []
        for _, row in rot.iterrows():
            in_t = row["IN_TIME_REAL"]
            out_t = row["OUT_TIME_REAL"]

            splits = [bp for bp in _BREAK_POINTS if in_t < bp < out_t]

            if not splits:
                rows.append(dict(row))
            else:
                boundaries = [in_t] + splits + [out_t]
                for j in range(len(boundaries) - 1):
                    new_row = dict(row)
                    new_row["IN_TIME_REAL"] = boundaries[j]
                    new_row["OUT_TIME_REAL"] = boundaries[j + 1]
                    new_row["PLAYER_PTS"] = None
                    new_row["PT_DIFF"] = None
                    rows.append(new_row)

        stints = pd.DataFrame(rows)

        # Derived columns
        stints["in_game_seconds"] = stints["IN_TIME_REAL"] / 10.0
        stints["out_game_seconds"] = stints["OUT_TIME_REAL"] / 10.0
        stints["stint_duration_minutes"] = (stints["OUT_TIME_REAL"] - stints["IN_TIME_REAL"]) / 600.0

        # Rename to match project conventions
        stints = stints.rename(
            columns={
                "PERSON_ID": "personId",
                "TEAM_ID": "teamId",
                "PLAYER_FIRST": "playerFirst",
                "PLAYER_LAST": "playerLast",
                "PLAYER_PTS": "player_pts",
                "PT_DIFF": "pt_diff",
            }
        )

        # Add stint_id per (gameId, personId)
        stints = stints.sort_values(["gameId", "personId", "in_game_seconds"]).reset_index(drop=True)
        stints["stint_id"] = stints.groupby(["gameId", "personId"]).cumcount() + 1

        # Select final columns
        keep = [
            "gameId",
            "personId",
            "playerFirst",
            "playerLast",
            "teamId",
            "location",
            "stint_id",
            "in_game_seconds",
            "out_game_seconds",
            "stint_duration_minutes",
            "player_pts",
            "pt_diff",
        ]
        stints = stints[[c for c in keep if c in stints.columns]]

        stints.to_parquet(out_path, index=False, compression="zstd")
        print(f"Saved stints: {len(stints):,} rows x {len(stints.columns)} cols -> {out_path}")
        return stints
