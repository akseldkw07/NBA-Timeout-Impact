"""Scratch space for iterating on TV-timeout classification rules.

Each experiment is a self-contained ``classify_vN(v3_pl, ...)`` function
that returns a polars DataFrame with a ``timeout_role`` column (the same
contract as ``tv_timeout_injection.TVTimeoutValidation.classify_timeouts``).

The notebook drives them through ``run_experiment(name, classify_fn)``.

Once a method wins, port it into ``tv_timeout_injection.py``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import polars as pl

PRE_2017_TRIGGERS = [540, 360, 180]  # 8:59 / 5:59 / 2:59 (Q2/Q4 only, pre-2017)
PRE_2017_PERIODS = [2, 4]
POST_2017_TRIGGERS = [420, 180]  # 6:59 / 2:59 (Q1-Q4, post-2017)
POST_2017_PERIODS = [1, 2, 3, 4]

CHALLENGE_SUBS = {"Coach Challenge"}


# --------------------------------------------------------------------------- #
#  Common preamble / scoring                                                  #
# --------------------------------------------------------------------------- #


def _prep_pd(v3_pl: pl.DataFrame, seasons: tuple[int, int]) -> pd.DataFrame:
    """Filter to seasons, normalize strings, sort by event order, return pandas."""
    df = v3_pl.to_pandas()
    lo, hi = seasons
    df = df[df["season"].between(lo, hi)].copy()
    df["actionType"] = df["actionType"].astype(str).str.strip()
    df["subType"] = df["subType"].astype(str).str.strip()
    df = df.sort_values(["gameId", "actionNumber"]).reset_index(drop=True)
    return df


def prep_v3_full(memo, seasons: tuple[int, int]) -> pl.DataFrame:
    """Like ``TVTimeoutValidation._prep_v3`` but keeps richer columns
    (``personId``, ``teamId``, ``description``) for context-aware experiments.
    """
    v3 = memo.data
    cols = [
        "gameId",
        "actionNumber",
        "period",
        "actionType",
        "subType",
        "seconds_remaining",
        "season",
        "season_type",
        "personId",
        "teamId",
        "description",
    ]
    cols = [c for c in cols if c in v3.columns]
    sub = v3[(v3["season"] >= seasons[0]) & (v3["season"] <= seasons[1])][cols].copy()
    for c in ("actionType", "subType"):
        sub[c] = sub[c].astype("string").str.strip()
    return pl.from_pandas(sub)


def _init_roles(df: pd.DataFrame, challenge_subs: set[str]) -> tuple[pd.Series, pd.Series]:
    """Initialize ``timeout_role`` column. Returns ``(is_coach_to, is_pre_2017)``."""
    df["timeout_role"] = ""
    is_timeout = df["actionType"] == "Timeout"
    is_challenge = is_timeout & df["subType"].isin(challenge_subs)
    is_coach_to = is_timeout & ~is_challenge
    df.loc[is_challenge, "timeout_role"] = "challenge"
    df.loc[is_coach_to, "timeout_role"] = "discretionary"
    pre_mask = (df["season"] < 2017).fillna(False).astype(bool)
    return is_coach_to, pre_mask


def score(classified: pl.DataFrame, label: str) -> dict:
    """Row-by-row TP/FP/FN against v3 Official subType. Returns a dict."""
    tos = (
        classified.filter(pl.col("actionType").cast(pl.String).str.strip_chars() == "Timeout")
        .with_columns(
            pl.col("subType").cast(pl.String).str.strip_chars().alias("_gt"),
            pl.col("timeout_role").str.contains("_mandatory").alias("_pred_mand"),
        )
        .with_columns(pl.col("_gt").is_in(["Official", "Official TV"]).alias("_is_gt"))
    )
    tp = tos.filter(pl.col("_is_gt") & pl.col("_pred_mand")).height
    fp = tos.filter(~pl.col("_is_gt") & pl.col("_pred_mand")).height
    fn = tos.filter(pl.col("_is_gt") & ~pl.col("_pred_mand")).height
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return dict(label=label, tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=f1)


def confusion(classified: pl.DataFrame) -> pd.DataFrame:
    """Cross-tab predicted role vs v3 subType (timeouts only)."""
    tos = (
        classified.filter(pl.col("actionType").cast(pl.String).str.strip_chars() == "Timeout")
        .select(
            pl.col("subType").cast(pl.String).str.strip_chars().alias("gt_subType"),
            pl.col("timeout_role").alias("predicted_role"),
        )
        .to_pandas()
    )
    return pd.crosstab(tos["gt_subType"], tos["predicted_role"])


# --------------------------------------------------------------------------- #
#  Shared slot-claim primitive (used by v1/v2/v3)                             #
# --------------------------------------------------------------------------- #


def _apply_slot_claim(
    df: pd.DataFrame,
    *,
    eligible: pd.Series,
    triggers: list[int],
    periods_ok: list[int],
    mand_tol_below: int,
    mand_tol_above: int,
    absorb_upper_offset: int | None,  # None = previous trigger; int = trigger + offset
    cascading: bool,
) -> None:
    """Sequential slot claim. For each slot K (in trigger order), claim the
    first eligible coach TO per (gameId, period) whose sr is in
    ``(trigger_K - mand_tol_below, upper_K]`` where
    ``upper_K = trigger_K + absorb_upper_offset`` (or the previous trigger
    if ``absorb_upper_offset is None``). Tag mandatory if sr ≤
    trigger_K + mand_tol_above, else absorbed.
    """
    sr = df["seconds_remaining"]
    in_period = df["period"].isin(periods_ok)
    claimed = pd.Series(False, index=df.index)
    blocked = pd.Series(False, index=df.index)
    group_keys = [df["gameId"], df["period"]]

    for K, trigger in enumerate(triggers, start=1):
        if absorb_upper_offset is None:
            upper = triggers[K - 2] if K >= 2 else 720
        else:
            upper = trigger + absorb_upper_offset
        lower = trigger - mand_tol_below
        mand_upper = trigger + mand_tol_above

        slot_eligible = eligible & in_period & (sr > lower) & (sr <= upper) & ~claimed & ~blocked
        cum = slot_eligible.astype(int).groupby(group_keys).cumsum()
        is_first = (cum == 1) & slot_eligible

        is_mand = is_first & (sr <= mand_upper)
        is_absorb = is_first & (sr > mand_upper)

        df.loc[is_absorb, "timeout_role"] = f"slot_{K}_absorbed"
        df.loc[is_mand, "timeout_role"] = f"slot_{K}_mandatory"

        claimed = claimed | is_first

        if cascading:
            fired_per_group = is_mand.astype(int).groupby(group_keys).transform("max").astype(bool)
            blocked = blocked | fired_per_group


# --------------------------------------------------------------------------- #
#  Experiments                                                                #
# --------------------------------------------------------------------------- #


def classify_v1_baseline(v3_pl: pl.DataFrame, seasons: tuple[int, int] = (2013, 2016)) -> pl.DataFrame:
    """v1: reproduce the production baseline. Cascading, mand_tol_below=90,
    no above-tolerance, absorb upper = previous trigger. F1 ≈ 0.668."""
    df = _prep_pd(v3_pl, seasons)
    is_coach_to, pre_mask = _init_roles(df, CHALLENGE_SUBS)
    _apply_slot_claim(
        df,
        eligible=pre_mask & is_coach_to,
        triggers=PRE_2017_TRIGGERS,
        periods_ok=PRE_2017_PERIODS,
        mand_tol_below=90,
        mand_tol_above=0,
        absorb_upper_offset=None,
        cascading=True,
    )
    _apply_slot_claim(
        df,
        eligible=(~pre_mask) & is_coach_to,
        triggers=POST_2017_TRIGGERS,
        periods_ok=POST_2017_PERIODS,
        mand_tol_below=90,
        mand_tol_above=0,
        absorb_upper_offset=None,
        cascading=False,
    )
    return pl.from_pandas(df)


def classify_v2_tight_absorb(v3_pl: pl.DataFrame, seasons: tuple[int, int] = (2013, 2016)) -> pl.DataFrame:
    """v2: absorb window capped at trigger + 30s.

    Hypothesis from the data: 0 Officials sit at sr > trigger anywhere. So
    a Regular TO at sr=600 is just a pre-trigger coach TO, NOT absorbing
    slot 1. The baseline (with upper=720) wrongly claims slot 1 with that
    TO and blocks slot 2/3 from being claimed by the actual Official.

    Fix: cap absorb at trigger + 30s. Pre-trigger coach TOs above that
    stay discretionary, leaving slot K open to claim the real Official.
    """
    df = _prep_pd(v3_pl, seasons)
    is_coach_to, pre_mask = _init_roles(df, CHALLENGE_SUBS)
    _apply_slot_claim(
        df,
        eligible=pre_mask & is_coach_to,
        triggers=PRE_2017_TRIGGERS,
        periods_ok=PRE_2017_PERIODS,
        mand_tol_below=90,
        mand_tol_above=0,
        absorb_upper_offset=30,
        cascading=True,
    )
    _apply_slot_claim(
        df,
        eligible=(~pre_mask) & is_coach_to,
        triggers=POST_2017_TRIGGERS,
        periods_ok=POST_2017_PERIODS,
        mand_tol_below=90,
        mand_tol_above=0,
        absorb_upper_offset=30,
        cascading=False,
    )
    return pl.from_pandas(df)


def classify_v3_no_absorb(v3_pl: pl.DataFrame, seasons: tuple[int, int] = (2013, 2016)) -> pl.DataFrame:
    """v3: no absorb at all — slot K only claimed by TOs at sr ≤ trigger_K.

    Even harder version of v2: drop absorb entirely. Empirically there are
    0 Officials at sr above any trigger, so an "absorbed" label adds no
    signal AND can pollute downstream cascading state.
    """
    df = _prep_pd(v3_pl, seasons)
    is_coach_to, pre_mask = _init_roles(df, CHALLENGE_SUBS)
    _apply_slot_claim(
        df,
        eligible=pre_mask & is_coach_to,
        triggers=PRE_2017_TRIGGERS,
        periods_ok=PRE_2017_PERIODS,
        mand_tol_below=90,
        mand_tol_above=0,
        absorb_upper_offset=0,
        cascading=True,
    )
    _apply_slot_claim(
        df,
        eligible=(~pre_mask) & is_coach_to,
        triggers=POST_2017_TRIGGERS,
        periods_ok=POST_2017_PERIODS,
        mand_tol_below=90,
        mand_tol_above=0,
        absorb_upper_offset=0,
        cascading=False,
    )
    return pl.from_pandas(df)


def classify_v4_one_per_period(v3_pl: pl.DataFrame, seasons: tuple[int, int] = (2013, 2016)) -> pl.DataFrame:
    """v4: at most ONE mandatory per (gameId, period), winner-take-all.

    Data: 99.9% of pre-2017 Q2/Q4 game-periods have 0 or 1 Official. So
    we should never predict ≥ 2 mandatories per period.

    Algorithm:
    - For each coach TO at sr ∈ [trigger_K - 90, trigger_K] for any K,
      compute (slot, distance_below_trigger).
    - Per (gameId, period), pick the candidate with the smallest distance.
      Tag that one ``slot_K_mandatory``. Everyone else: discretionary.
    """
    df = _prep_pd(v3_pl, seasons)
    is_coach_to, pre_mask = _init_roles(df, CHALLENGE_SUBS)
    TOL_BELOW = 90

    def _annotate(eligible: pd.Series, triggers: list[int], periods_ok: list[int]):
        sr = df["seconds_remaining"]
        cand = eligible & df["period"].isin(periods_ok)
        best_slot = pd.Series(0, index=df.index, dtype="int64")
        best_dist = pd.Series(np.inf, index=df.index, dtype="float64")
        for K, trigger in enumerate(triggers, start=1):
            dist = (trigger - sr).astype("float64")
            in_window = cand & (dist >= 0) & (dist <= TOL_BELOW)
            improves = in_window & (dist < best_dist)
            best_slot = best_slot.mask(improves, K)
            best_dist = best_dist.mask(improves, dist)
        scored = df.assign(_best_slot=best_slot, _best_dist=best_dist).loc[best_slot > 0]
        if scored.empty:
            return
        idx = scored.groupby(["gameId", "period"])["_best_dist"].idxmin().to_numpy()
        winner_slots = best_slot.loc[idx]
        for K in range(1, len(triggers) + 1):
            mask = winner_slots == K
            df.loc[winner_slots.index[mask], "timeout_role"] = f"slot_{K}_mandatory"

    _annotate(pre_mask & is_coach_to, PRE_2017_TRIGGERS, PRE_2017_PERIODS)
    _annotate((~pre_mask) & is_coach_to, POST_2017_TRIGGERS, POST_2017_PERIODS)
    df.drop(columns=[c for c in ("_best_slot", "_best_dist") if c in df.columns], inplace=True)
    return pl.from_pandas(df)


def classify_v5_q1q3_added(v3_pl: pl.DataFrame, seasons: tuple[int, int] = (2013, 2016)) -> pl.DataFrame:
    """v5: rulebook-correct pre-2017 — Q1/Q3 also get mandatories.

    Research finding (NBA rulebook 1998–2016, Rule 5 §II): Q1 and Q3 each
    had 2 mandatory triggers at 5:59 and 2:59. Q2/Q4 had 3 at 8:59/5:59/2:59.
    OT had 1 at 2:59. My previous models gave Q1/Q3 zero slots — even
    though v3 had only 1 Official across them in 2013-2016 (essentially
    all absorbed by coach TOs), the slots ARE there.

    Pre-2017 cascading, mand_tol_below=90, absorb=previous trigger.
    """
    df = _prep_pd(v3_pl, seasons)
    is_coach_to, pre_mask = _init_roles(df, CHALLENGE_SUBS)

    # Q2/Q4 — 3 slots
    _apply_slot_claim(
        df,
        eligible=pre_mask & is_coach_to,
        triggers=[540, 360, 180],
        periods_ok=[2, 4],
        mand_tol_below=90,
        mand_tol_above=0,
        absorb_upper_offset=None,
        cascading=True,
    )
    # Q1/Q3 — 2 slots (5:59, 2:59)
    _apply_slot_claim(
        df,
        eligible=pre_mask & is_coach_to,
        triggers=[360, 180],
        periods_ok=[1, 3],
        mand_tol_below=90,
        mand_tol_above=0,
        absorb_upper_offset=None,
        cascading=True,
    )
    # OT — 1 slot (2:59); OT periods are 5 minutes (sr ≤ 300)
    _apply_slot_claim(
        df,
        eligible=pre_mask & is_coach_to,
        triggers=[180],
        periods_ok=[5, 6, 7, 8, 9, 10],
        mand_tol_below=90,
        mand_tol_above=0,
        absorb_upper_offset=None,
        cascading=True,
    )
    # Post-2017
    _apply_slot_claim(
        df,
        eligible=(~pre_mask) & is_coach_to,
        triggers=POST_2017_TRIGGERS,
        periods_ok=POST_2017_PERIODS,
        mand_tol_below=90,
        mand_tol_above=0,
        absorb_upper_offset=None,
        cascading=False,
    )
    return pl.from_pandas(df)


def classify_v6_stateful_rulebook(v3_pl: pl.DataFrame, seasons: tuple[int, int] = (2013, 2016)) -> pl.DataFrame:
    """v6: stateful 'first TO after threshold X:59' per rulebook.

    Rulebook rule (research finding): mandatory K fires iff *no team TO*
    has been logged since trigger K-1. The next dead ball after sr crosses
    below trigger_K resolves the mandatory — and that dead ball is itself
    typically logged as a Timeout row (the Official TV).

    Algorithm: per (gameId, period), walk all timeouts in order. Maintain
    ``next_slot`` = index of the next unsatisfied trigger.
    - If TO at sr > trigger[next_slot] (still pre-trigger): the TO ABSORBS
      slot ``next_slot`` (coach used their TO before the trigger expired).
      Advance ``next_slot``.
    - If TO at sr ≤ trigger[next_slot]: the trigger has just expired;
      this TO is the mandatory firing. Tag ``slot_K_mandatory``. Advance.
    - Subsequent TOs after all slots resolved stay discretionary.

    Includes Q1/Q3 triggers from the rulebook.
    """
    df = _prep_pd(v3_pl, seasons)
    is_coach_to, pre_mask = _init_roles(df, CHALLENGE_SUBS)

    def _trig_for(season: int, period: int) -> list[int]:
        if season < 2017:
            if period in (2, 4):
                return [540, 360, 180]
            if period in (1, 3):
                return [360, 180]
            if period >= 5:  # OT
                return [180]
            return []
        # post-2017
        if period in (1, 2, 3, 4):
            return [420, 180]
        if period >= 5:
            return [180]
        return []

    # Walk per (gameId, period) — small Python loop is fine; the dataframe
    # is sorted and we only touch TIMEOUT rows.
    is_to = df["actionType"] == "Timeout"
    tos = df[is_to].copy()
    for (gid, per), group in tos.groupby(["gameId", "period"], sort=False):
        season = int(group["season"].iloc[0])
        trig = _trig_for(season, per)  # type: ignore[assignment]
        if not trig:
            continue
        next_slot = 0
        for idx, row in group.iterrows():
            if next_slot >= len(trig):
                break
            if row["subType"] in CHALLENGE_SUBS:
                continue
            sr = row["seconds_remaining"]
            if pd.isna(sr):
                continue
            t = trig[next_slot]
            if sr > t:
                df.at[idx, "timeout_role"] = f"slot_{next_slot + 1}_absorbed"
            else:
                df.at[idx, "timeout_role"] = f"slot_{next_slot + 1}_mandatory"
            next_slot += 1
    return pl.from_pandas(df)


def classify_v7_personid_zero(v3_pl: pl.DataFrame, seasons: tuple[int, int] = (2013, 2016)) -> pl.DataFrame:
    """v7: ``personId == 0`` is the structural Official-TV signal.

    Inspecting v3 raw rows: every Official subType has ``personId=0`` (no
    person/team charged — the league auto-fired it). Every Regular/Short
    has ``personId`` = the team's ID. This is a structural property of how
    the play-by-play feed records auto-charged mandatories, NOT a copy of
    the subType label.

    This should match v3's Officials almost exactly. It's also the
    candidate generalization to post-2017: if cdnnba auto-fired mandatories
    also have personId=0 (or null), the same rule transfers.

    NOTE: this rule requires ``personId`` in the input frame.
    """
    df = _prep_pd(v3_pl, seasons)
    is_coach_to, pre_mask = _init_roles(df, CHALLENGE_SUBS)

    is_timeout = df["actionType"] == "Timeout"
    if "personId" not in df.columns:
        raise ValueError("classify_v7 requires personId in the input frame")
    is_league_fired = is_timeout & (df["personId"].fillna(-1) == 0)

    # Assign slot by sr region (use Q2/Q4 triggers for slot naming;
    # Q1/Q3 mandatories also exist in the rulebook but are rare in data).
    def _slot_for(sr, period):
        if period in (2, 4):
            if sr > 360:
                return 1
            if sr > 180:
                return 2
            return 3
        if period in (1, 3):
            if sr > 180:
                return 1
            return 2
        return 1  # OT or unknown

    sub = df[is_league_fired].copy()
    if not sub.empty:
        sub["_slot"] = [_slot_for(s, p) for s, p in zip(sub["seconds_remaining"], sub["period"])]
        for K in (1, 2, 3):
            idx = sub.index[sub["_slot"] == K]
            df.loc[idx, "timeout_role"] = f"slot_{K}_mandatory"

    return pl.from_pandas(df)


def classify_v8_personid_with_position(v3_pl: pl.DataFrame, seasons: tuple[int, int] = (2013, 2016)) -> pl.DataFrame:
    """v8: ``personId == 0`` + sr near a trigger.

    Stricter than v7 — only tag mandatory if the TO has personId=0 AND its
    sr is within a sensible window below a known trigger. Should filter
    out any stray personId=0 rows that happen to be far from a trigger
    (rare but worth checking).
    """
    df = _prep_pd(v3_pl, seasons)
    is_coach_to, pre_mask = _init_roles(df, CHALLENGE_SUBS)
    is_timeout = df["actionType"] == "Timeout"
    is_league_fired = is_timeout & (df["personId"].fillna(-1) == 0)

    TOL = 90
    sr = df["seconds_remaining"]

    def _try_slots(periods_ok, triggers):
        in_per = df["period"].isin(periods_ok)
        for K, t in enumerate(triggers, start=1):
            in_win = (
                is_league_fired & in_per & (sr <= t) & (sr >= t - TOL) & (df["timeout_role"] != f"slot_{K}_mandatory")
            )
            df.loc[in_win, "timeout_role"] = f"slot_{K}_mandatory"

    _try_slots([2, 4], [540, 360, 180])  # Q2/Q4
    _try_slots([1, 3], [360, 180])  # Q1/Q3
    _try_slots([5, 6, 7, 8, 9, 10], [180])  # OT
    return pl.from_pandas(df)


# --------------------------------------------------------------------------- #
#  Registry — keys appear in the notebook in iteration order                  #
# --------------------------------------------------------------------------- #

CLASSIFIERS: dict[str, Callable[[pl.DataFrame, tuple[int, int]], pl.DataFrame]] = {
    "v1_baseline": classify_v1_baseline,
    "v2_tight_absorb": classify_v2_tight_absorb,
    "v3_no_absorb": classify_v3_no_absorb,
    "v4_one_per_period": classify_v4_one_per_period,
    "v5_q1q3_added": classify_v5_q1q3_added,
    "v6_stateful_rulebook": classify_v6_stateful_rulebook,
    "v7_personid_zero": classify_v7_personid_zero,
    "v8_personid_plus_position": classify_v8_personid_with_position,
}
