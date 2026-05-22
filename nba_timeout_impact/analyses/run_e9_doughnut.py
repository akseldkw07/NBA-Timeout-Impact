"""Robustness test for E9 matched-twin: doughnut constraint on clock distance.

The standard E9 matching requires |Δ seconds_remaining| ≤ 120s within the same
period. Because the forward window is 180s, treated and control windows
overlap heavily (60-180s out of 180s). The pair difference therefore
reduces to a localized diff-in-diff at the window edges.

This script re-runs the matched-twin estimator with a **doughnut**:

    same gameId
    same streak sign
    |Δ streak| ≤ 2
    |Δ suffering_margin| ≤ 3
    180 ≤ |Δ game_seconds_elapsed| ≤ 720     (windows don't overlap)

If the headline +0.408/+0.405 effect survives, the standard E9 design is
robust. If it shrinks toward zero, the standard estimate is being inflated
by window overlap.

Run with: ``python -m nba_timeout_impact.analyses.run_e9_doughnut``
"""

from __future__ import annotations

import polars as pl
from scipy import stats as sp_stats

from nba_timeout_impact.analyses.run_experiments_v2 import build_rich_analysis
from nba_timeout_impact.datasets.memo_cdnnba_pl import CDNNBAMemoPL

W = 180  # forward recovery window in seconds
DOUGHNUT_MIN = 180  # minimum |Δ game_seconds_elapsed|: no overlap
DOUGHNUT_MAX = 720  # maximum |Δ game_seconds_elapsed|: stay within ~12 min


def _doughnut_matched_twins(analysis: pl.DataFrame) -> pl.DataFrame:
    treated = analysis.filter(pl.col("group") != "control").with_row_index("_tid")
    controls = analysis.filter(pl.col("group") == "control")

    joined = treated.join(
        controls.select(
            "gameId",
            pl.col("streak").alias("c_streak"),
            pl.col("game_seconds_elapsed").alias("c_gse"),
            pl.col("suffering_margin").alias("c_margin"),
            pl.col("recovery").alias("c_recovery"),
        ),
        on=["gameId"],
        how="inner",
    )
    dt = (pl.col("game_seconds_elapsed") - pl.col("c_gse")).abs()
    joined = joined.filter(
        (pl.col("streak").sign() == pl.col("c_streak").sign())
        & ((pl.col("streak") - pl.col("c_streak")).abs() <= 2)
        & (dt >= DOUGHNUT_MIN)
        & (dt <= DOUGHNUT_MAX)
        & ((pl.col("suffering_margin") - pl.col("c_margin")).abs() <= 3)
    )
    joined = joined.with_columns(
        (
            (pl.col("streak") - pl.col("c_streak")).abs() * 2
            + (dt / 60)
            + ((pl.col("suffering_margin") - pl.col("c_margin")).abs() * 3)
        ).alias("_dist")
    )
    return joined.sort("_tid", "_dist").group_by("_tid", maintain_order=True).first()


def _standard_matched_twins(analysis: pl.DataFrame) -> pl.DataFrame:
    """Replica of experiment_9's matching for side-by-side comparison."""
    treated = analysis.filter(pl.col("group") != "control").with_row_index("_tid")
    controls = analysis.filter(pl.col("group") == "control")
    joined = treated.join(
        controls.select(
            "gameId",
            "period",
            pl.col("streak").alias("c_streak"),
            pl.col("seconds_remaining").alias("c_sr"),
            pl.col("suffering_margin").alias("c_margin"),
            pl.col("recovery").alias("c_recovery"),
        ),
        on=["gameId", "period"],
        how="inner",
    )
    joined = joined.filter(
        (pl.col("streak").sign() == pl.col("c_streak").sign())
        & ((pl.col("streak") - pl.col("c_streak")).abs() <= 2)
        & ((pl.col("seconds_remaining") - pl.col("c_sr")).abs() <= 120)
        & ((pl.col("suffering_margin") - pl.col("c_margin")).abs() <= 3)
    )
    joined = joined.with_columns(
        (
            (pl.col("streak") - pl.col("c_streak")).abs() * 2
            + ((pl.col("seconds_remaining") - pl.col("c_sr")).abs() / 30)
            + ((pl.col("suffering_margin") - pl.col("c_margin")).abs() * 3)
        ).alias("_dist")
    )
    return joined.sort("_tid", "_dist").group_by("_tid", maintain_order=True).first()


def _different_quarter_matched_twins(analysis: pl.DataFrame) -> pl.DataFrame:
    """Same as standard, but the control must be in a DIFFERENT period.

    Single rule change from the standard matching: ``period`` flips from
    equality (``on=["gameId", "period"]``) to inequality (joined on gameId,
    filtered to period != c_period). Everything else --- within-period clock
    proximity (|Δsr| ≤ 120), streak sign + magnitude, suffering margin ---
    is preserved.
    """
    treated = analysis.filter(pl.col("group") != "control").with_row_index("_tid")
    controls = analysis.filter(pl.col("group") == "control")
    joined = treated.join(
        controls.select(
            "gameId",
            pl.col("period").alias("c_period"),
            pl.col("streak").alias("c_streak"),
            pl.col("seconds_remaining").alias("c_sr"),
            pl.col("suffering_margin").alias("c_margin"),
            pl.col("recovery").alias("c_recovery"),
        ),
        on=["gameId"],
        how="inner",
    )
    joined = joined.filter(
        (pl.col("period") != pl.col("c_period"))
        & (pl.col("streak").sign() == pl.col("c_streak").sign())
        & ((pl.col("streak") - pl.col("c_streak")).abs() <= 2)
        & ((pl.col("seconds_remaining") - pl.col("c_sr")).abs() <= 120)
        & ((pl.col("suffering_margin") - pl.col("c_margin")).abs() <= 3)
    )
    joined = joined.with_columns(
        (
            (pl.col("streak") - pl.col("c_streak")).abs() * 2
            + ((pl.col("seconds_remaining") - pl.col("c_sr")).abs() / 30)
            + ((pl.col("suffering_margin") - pl.col("c_margin")).abs() * 3)
        ).alias("_dist")
    )
    return joined.sort("_tid", "_dist").group_by("_tid", maintain_order=True).first()


def _report(label: str, best: pl.DataFrame) -> None:
    print(f"\n=== {label} ===")
    print(f"matched pairs: {best.height:,}")
    print(f"{'group':<26}  {'n':>7}  {'treated μ':>10}  {'ctrl μ':>8}  {'pair Δ':>8}  {'t':>7}  {'p':>9}  sig")
    rows = []
    for grp in ["endogenous", "exogenous"]:
        sub = best.filter(pl.col("group") == grp)
        diffs = sub["recovery"].to_numpy() - sub["c_recovery"].to_numpy()
        rows.append((grp, sub["recovery"].to_numpy(), sub["c_recovery"].to_numpy(), diffs))
    for sub in [
        "tv_mandatory",
        "stoppage",
        "coach_absorb",
        "coach_discretionary",
        "mistagged_discretionary",
        "coach_challenge",
    ]:
        s = best.filter(pl.col("subtype_fine") == sub)
        if s.height < 5:
            continue
        diffs = s["recovery"].to_numpy() - s["c_recovery"].to_numpy()
        rows.append((sub, s["recovery"].to_numpy(), s["c_recovery"].to_numpy(), diffs))

    for name, t_arr, c_arr, diffs in rows:
        if len(diffs) < 2:
            continue
        t, p = sp_stats.ttest_1samp(diffs, 0)
        star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        print(
            f"{name:<26}  {len(diffs):>7,}  {t_arr.mean():>+10.3f}  {c_arr.mean():>+8.3f}  "
            f"{diffs.mean():>+8.4f}  {t:>7.3f}  {float(p):>9.4f}  {star}"
        )


def main() -> None:
    print("Loading memo…")
    memo = CDNNBAMemoPL.load_all()

    print("Building rich analysis frame…")
    analysis = build_rich_analysis(memo, run_size=6, minutes=3.0)
    print(f"  analysis rows: {analysis.height:,}")

    print("\nComputing STANDARD matched twins (same period, |Δsr| ≤ 120s)…")
    std = _standard_matched_twins(analysis)
    _report("Standard E9 (same period, |Δsr| ≤ 120s)", std)

    print("\nComputing DOUGHNUT matched twins (same game, 180 ≤ |Δgse| ≤ 720s)…")
    don = _doughnut_matched_twins(analysis)
    _report(f"Doughnut E9 ({DOUGHNUT_MIN} ≤ |Δgse| ≤ {DOUGHNUT_MAX}s)", don)

    print("\nComputing DIFFERENT-QUARTER matched twins (same |Δsr| ≤ 120, period ≠ c_period)…")
    dq = _different_quarter_matched_twins(analysis)
    _report("Different-quarter E9 (period ≠ c_period, |Δsr| ≤ 120s)", dq)

    print("\nDone.")


if __name__ == "__main__":
    main()
