"""Generate the figure set embedded in Report/applied-causality/report.tex.

Run with: ``python -m nba_timeout_impact.analyses.generate_report_figures``

Outputs PNGs into ``Report/applied-causality/figures/``.

Figures produced:
1. ``fig_cause_vs_sr.png`` — stacked histogram of timeout cause vs seconds
   remaining (§3.1).
2. ``fig_betweengroup_vs_matched.png`` — endo/exo means under the two
   estimators side by side (§4.1 → §4.2 reversal).
3. ``fig_recovery_distribution.png`` — KDE of recovery for endo/exo/control.
4. ``fig_matched_twin_forest.png`` — forest plot of E9 matched-twin pair-diffs
   with 95% CIs across the six fine subtypes (§4.2).
5. ``fig_moderator_heatmap.png`` — heatmap of Welch deltas across moderators
   (§4.3).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import TwoSlopeNorm
from scipy import stats as sp_stats

from nba_timeout_impact.analyses.run_experiments_v2 import build_rich_analysis
from nba_timeout_impact.datasets.memo_cdnnba_pl import CDNNBAMemoPL
from nba_timeout_impact.plotting_utils.timeout_injection_plots import TimeoutInjectionPlots

OUT_DIR = Path(__file__).resolve().parents[2] / "Report" / "applied-causality" / "figures"


# ============================================================================
# Figure 1 — cause distribution vs seconds remaining
# ============================================================================


def fig_cause_vs_sr(memo: CDNNBAMemoPL) -> None:
    fig, _ = TimeoutInjectionPlots.plot_role_vs_sr(
        memo.cdnnba,
        combine_periods=True,
        color_by="cause",
        height_per=3.2,
        width_per=14,
    )
    fig.savefig(OUT_DIR / "fig_cause_vs_sr.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_cause_vs_sr.png")


# ============================================================================
# Matched-twin pair computation (shared by figs 2, 4)
# ============================================================================


def _matched_twins(analysis: pl.DataFrame) -> pl.DataFrame:
    """Return DataFrame of best matched twins for treated events.

    Same matching rules as ``experiment_9``.
    """
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


# ============================================================================
# Figure 2 — between-group vs matched-twin estimator contrast
# ============================================================================


def fig_betweengroup_vs_matched(analysis: pl.DataFrame, best: pl.DataFrame) -> None:
    rec = {
        g: analysis.filter(pl.col("group") == g)["recovery"].to_numpy() for g in ["endogenous", "exogenous", "control"]
    }
    ctrl_mean = rec["control"].mean()
    bg = {g: rec[g].mean() - ctrl_mean for g in ["endogenous", "exogenous"]}

    mt = {}
    for g in ["endogenous", "exogenous"]:
        sub = best.filter(pl.col("group") == g)
        diffs = sub["recovery"].to_numpy() - sub["c_recovery"].to_numpy()
        mt[g] = diffs.mean()

    fig, (ax_bg, ax_mt) = plt.subplots(1, 2, figsize=(8.5, 3.6), sharey=True)
    colors = {"endogenous": "#1f77b4", "exogenous": "#d62728"}
    for ax, data, title in [
        (ax_bg, bg, "Between-group (§4.1)\n$\\mu_{treated} - \\mu_{control}$"),
        (ax_mt, mt, "Matched-twin (§4.2)\nper-pair $\\Delta$, in-game"),
    ]:
        xs = np.arange(2)
        ys = [data["endogenous"], data["exogenous"]]
        bars = ax.bar(xs, ys, color=[colors["endogenous"], colors["exogenous"]], width=0.55)
        for bar, val in zip(bars, ys):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + (0.02 if val >= 0 else -0.04),
                f"{val:+.3f}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=10,
                fontweight="bold",
            )
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels(["endogenous", "exogenous"])
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.25, 0.55)
        ax.grid(axis="y", alpha=0.3)
    ax_bg.set_ylabel("recovery $\\Delta$ vs control (pts)")
    fig.suptitle("Same effect, two estimators: the matched-twin design reverses the headline", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_betweengroup_vs_matched.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_betweengroup_vs_matched.png")


# ============================================================================
# Figure 3 — recovery distribution
# ============================================================================


def fig_recovery_distribution(analysis: pl.DataFrame) -> None:
    rec = {
        g: analysis.filter(pl.col("group") == g)["recovery"].to_numpy() for g in ["endogenous", "exogenous", "control"]
    }
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    bins = np.linspace(-15, 15, 81)
    colors = {"endogenous": "#1f77b4", "exogenous": "#d62728", "control": "#7f7f7f"}
    for g in ["control", "exogenous", "endogenous"]:
        ax.hist(
            rec[g],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=colors[g],
            label=f"{g} (n={len(rec[g]):,}, $\\mu={rec[g].mean():+.3f}$)",
        )
        ax.axvline(rec[g].mean(), color=colors[g], linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("3-minute recovery (pts clawed back)")
    ax.set_ylabel("density")
    ax.set_title("Recovery distributions overlap heavily; group means differ by < 0.5 pts")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.85)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_recovery_distribution.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_recovery_distribution.png")


# ============================================================================
# Figure 4 — matched-twin forest plot with 95% CIs
# ============================================================================


SUBTYPE_ORDER = [
    "tv_mandatory",
    "stoppage",
    "coach_absorb",
    "coach_discretionary",
    "mistagged_discretionary",
    "coach_challenge",
]

SUBTYPE_COLORS = {
    "tv_mandatory": "#2ca02c",
    "stoppage": "#7f7f7f",
    "coach_absorb": "#bcbd22",
    "coach_discretionary": "#1f77b4",
    "mistagged_discretionary": "#ff7f0e",
    "coach_challenge": "#9467bd",
}


def fig_matched_twin_forest(best: pl.DataFrame) -> None:
    rows = []
    for sub in SUBTYPE_ORDER:
        s = best.filter(pl.col("subtype_fine") == sub)
        if s.height < 5:
            continue
        diffs = s["recovery"].to_numpy() - s["c_recovery"].to_numpy()
        n = len(diffs)
        mu = diffs.mean()
        se = diffs.std(ddof=1) / np.sqrt(n)
        ci = 1.96 * se
        t, p = sp_stats.ttest_1samp(diffs, 0)
        rows.append({"subtype": sub, "n": n, "mu": mu, "ci": ci, "p": float(p)})

    # Aggregate endo + exo bars for the headline summary
    for grp in ["endogenous", "exogenous"]:
        s = best.filter(pl.col("group") == grp)
        diffs = s["recovery"].to_numpy() - s["c_recovery"].to_numpy()
        n = len(diffs)
        mu = diffs.mean()
        ci = 1.96 * diffs.std(ddof=1) / np.sqrt(n)
        t, p = sp_stats.ttest_1samp(diffs, 0)
        rows.append({"subtype": grp, "n": n, "mu": mu, "ci": ci, "p": float(p)})

    order = ["endogenous", "exogenous"] + SUBTYPE_ORDER
    rows = sorted(rows, key=lambda r: order.index(r["subtype"]))

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ys = np.arange(len(rows))[::-1]
    for y, r in zip(ys, rows):
        color = SUBTYPE_COLORS.get(r["subtype"], "#333")
        if r["subtype"] in ("endogenous", "exogenous"):
            color = "#1f77b4" if r["subtype"] == "endogenous" else "#d62728"
        ax.errorbar(r["mu"], y, xerr=r["ci"], fmt="o", color=color, capsize=4, markersize=7, linewidth=2)
        stars = "***" if r["p"] < 0.001 else ("**" if r["p"] < 0.01 else ("*" if r["p"] < 0.05 else "n.s."))
        ax.text(
            r["mu"] + r["ci"] + 0.02,
            y,
            f"{r['mu']:+.3f} {stars}  (n={r['n']:,})",
            va="center",
            fontsize=9,
        )
    ax.axvline(0, color="black", linewidth=0.7)
    # Separator between aggregate and fine
    ax.axhline(ys[1] - 0.5, color="grey", linewidth=0.7, linestyle=":", alpha=0.7)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["subtype"] for r in rows])
    ax.set_xlabel("Matched-twin pair difference, 3-min recovery (pts)")
    ax.set_xlim(-0.05, 0.85)
    ax.set_title("Matched-twin causal estimates: positive across every event type", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_matched_twin_forest.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_matched_twin_forest.png")


# ============================================================================
# Figure 5 — moderator heatmap (Welch deltas across stratifications)
# ============================================================================


def _welch(treated: np.ndarray, control: np.ndarray) -> tuple[float, float]:
    if len(treated) < 2 or len(control) < 2:
        return float("nan"), float("nan")
    t, p = sp_stats.ttest_ind(treated, control, equal_var=False)
    return float(treated.mean() - control.mean()), float(p)


def _strata(analysis: pl.DataFrame, mask: pl.Expr) -> dict[str, tuple[float, float, int]]:
    sub = analysis.filter(mask)
    out = {}
    for g in ["endogenous", "exogenous"]:
        treated = sub.filter(pl.col("group") == g)["recovery"].to_numpy()
        ctrl = sub.filter(pl.col("group") == "control")["recovery"].to_numpy()
        d, p = _welch(treated, ctrl)
        out[g] = (d, p, len(treated))
    return out


def _attach_subs_30s(memo: CDNNBAMemoPL, analysis: pl.DataFrame) -> pl.DataFrame:
    df = pl.DataFrame._from_pydf(memo.cdnnba._df)
    subs = (
        df.filter(pl.col("actionType") == "substitution")
        .select("gameId", "game_seconds_elapsed")
        .rename({"game_seconds_elapsed": "sub_gse"})
    )
    ev = analysis.with_row_index("_eid")
    joined = ev.select("_eid", "gameId", "game_seconds_elapsed").join(subs, on="gameId", how="left")
    joined = joined.with_columns(((pl.col("sub_gse") - pl.col("game_seconds_elapsed")).abs() <= 30).alias("nearby"))
    counts = joined.group_by("_eid").agg(pl.col("nearby").sum().alias("subs_30s"))
    return ev.join(counts, on="_eid", how="left").with_columns(pl.col("subs_30s").fill_null(0)).drop("_eid")


def _attach_team_net(memo: CDNNBAMemoPL, analysis: pl.DataFrame) -> pl.DataFrame:
    pas = pl.DataFrame._from_pydf(memo.player_advanced_stats._df)
    team_net = (
        pas.with_columns((pl.col("NET_RATING") * pl.col("GP")).alias("net_gp"))
        .group_by("TEAM_ID", "season_int")
        .agg((pl.col("net_gp").sum() / pl.col("GP").sum()).alias("team_net_rating"))
        .rename({"TEAM_ID": "teamId", "season_int": "season"})
    )
    home_teams = memo.home_team_per_game.rename({"home_teamId": "home_tid"})
    df = pl.DataFrame._from_pydf(memo.cdnnba._df)
    away_teams = (
        df.filter(pl.col("teamId") > 0)
        .join(home_teams, on="gameId", how="left")
        .filter(pl.col("teamId") != pl.col("home_tid"))
        .select("gameId", "teamId", "season")
        .unique()
        .group_by("gameId")
        .agg(pl.col("teamId").first().alias("away_tid"), pl.col("season").first())
    )
    game_quality = (
        home_teams.join(away_teams, on="gameId", how="inner")
        .join(
            team_net.rename({"teamId": "home_tid", "team_net_rating": "home_net"}),
            on=["home_tid", "season"],
            how="left",
        )
        .join(
            team_net.rename({"teamId": "away_tid", "team_net_rating": "away_net"}),
            on=["away_tid", "season"],
            how="left",
        )
    )
    a2 = analysis.join(game_quality.select("gameId", "home_net", "away_net"), on="gameId", how="left")
    return a2.with_columns(
        pl.when(pl.col("suffering_location") == "home")
        .then(pl.col("home_net") - pl.col("away_net"))
        .otherwise(pl.col("away_net") - pl.col("home_net"))
        .alias("suffering_vs_opp_net"),
    )


def _attach_ppp(memo: CDNNBAMemoPL, analysis: pl.DataFrame) -> pl.DataFrame:
    poss = memo.possessions
    poss_with_ppp = (
        poss.sort("gameId", "possession_id")
        .with_columns(
            pl.col("possession_points").cum_sum().over("gameId", "possession").alias("cum_pts"),
            pl.cum_count("possession_id").over("gameId", "possession").alias("cum_poss"),
        )
        .with_columns((pl.col("cum_pts") / pl.col("cum_poss")).alias("team_cum_ppp"))
        .select(
            pl.col("gameId"),
            pl.col("possession").alias("poss_team_at_event"),
            pl.col("possession_id"),
            "team_cum_ppp",
        )
    )
    df = pl.DataFrame._from_pydf(memo.cdnnba._df).select(
        "gameId",
        "orderNumber",
        pl.col("possession").alias("poss_team_at_event"),
        "possession_id",
    )
    ev = analysis.join(df, on=["gameId", "orderNumber"], how="left")
    return ev.join(poss_with_ppp, on=["gameId", "poss_team_at_event", "possession_id"], how="left").rename(
        {"team_cum_ppp": "running_team_ppp"}
    )


def fig_moderator_heatmap(memo: CDNNBAMemoPL, analysis: pl.DataFrame) -> None:
    print("    attaching subs_30s…")
    a = _attach_subs_30s(memo, analysis)
    print("    attaching team_net…")
    a = _attach_team_net(memo, a)
    print("    attaching ppp…")
    a = _attach_ppp(memo, a)

    rows: list[tuple[str, dict[str, tuple[float, float, int]]]] = []

    # Substitutions
    rows.append(("0 subs", _strata(a, pl.col("subs_30s") == 0)))
    rows.append(("1--2 subs", _strata(a, (pl.col("subs_30s") >= 1) & (pl.col("subs_30s") <= 2))))
    rows.append(("3+ subs", _strata(a, pl.col("subs_30s") >= 3)))

    # Team quality (NET_RATING gap)
    rows.append(("Sufferer $\\ll$ opp ($\\Delta \\leq -5$)", _strata(a, pl.col("suffering_vs_opp_net") <= -5)))
    rows.append(("Evenly matched ($|\\Delta| < 1$)", _strata(a, pl.col("suffering_vs_opp_net").abs() < 1)))
    rows.append(("Sufferer $\\gg$ opp ($\\Delta \\geq 5$)", _strata(a, pl.col("suffering_vs_opp_net") >= 5)))

    # Game phase
    rows.append(("Q1 (0--720s)", _strata(a, pl.col("game_seconds_elapsed") < 720)))
    rows.append(
        (
            "Q2 (720--1440s)",
            _strata(a, (pl.col("game_seconds_elapsed") >= 720) & (pl.col("game_seconds_elapsed") < 1440)),
        )
    )
    rows.append(
        (
            "Q3 (1440--2160s)",
            _strata(a, (pl.col("game_seconds_elapsed") >= 1440) & (pl.col("game_seconds_elapsed") < 2160)),
        )
    )
    rows.append(("Q4 (2160--2880s)", _strata(a, pl.col("game_seconds_elapsed") >= 2160)))

    # PPP
    rows.append(("Running PPP $< 1.00$", _strata(a, pl.col("running_team_ppp") < 1.00)))
    rows.append(("Running PPP $\\geq 1.30$", _strata(a, pl.col("running_team_ppp") >= 1.30)))

    # Build matrix
    labels = [r[0] for r in rows]
    mat = np.array([[r[1]["endogenous"][0], r[1]["exogenous"][0]] for r in rows])
    pmat = np.array([[r[1]["endogenous"][1], r[1]["exogenous"][1]] for r in rows])
    nmat = np.array([[r[1]["endogenous"][2], r[1]["exogenous"][2]] for r in rows])

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)
    im = ax.imshow(mat, aspect="auto", cmap="RdBu", norm=norm)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            p = pmat[i, j]
            n = nmat[i, j]
            star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            text_color = "white" if abs(val) > 0.3 else "black"
            ax.text(j, i, f"{val:+.2f}{star}\nn={n:,}", ha="center", va="center", fontsize=8, color=text_color)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["endogenous $\\Delta$", "exogenous $\\Delta$"])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Welch $\\Delta = \\mu_{treated} - \\mu_{control}$ by moderator stratum", fontsize=11)
    fig.colorbar(im, ax=ax, label="$\\Delta$ recovery (pts)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_moderator_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_moderator_heatmap.png")


# ============================================================================
# Driver
# ============================================================================


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {OUT_DIR}")

    print("Loading memo...")
    memo = CDNNBAMemoPL.load_all()

    print("\n[Figure 1] cause vs seconds remaining")
    fig_cause_vs_sr(memo)

    print("\nBuilding rich analysis frame...")
    analysis = build_rich_analysis(memo, run_size=6, minutes=3.0)
    print(f"  analysis rows: {analysis.height:,}")

    print("\nComputing matched twins...")
    best = _matched_twins(analysis)
    print(f"  matched pairs: {best.height:,}")

    print("\n[Figure 2] between-group vs matched-twin")
    fig_betweengroup_vs_matched(analysis, best)

    print("\n[Figure 3] recovery distribution")
    fig_recovery_distribution(analysis)

    print("\n[Figure 4] matched-twin forest plot")
    fig_matched_twin_forest(best)

    print("\n[Figure 5] moderator heatmap")
    fig_moderator_heatmap(memo, analysis)

    print("\nDone.")


if __name__ == "__main__":
    main()
