"""Plots for the TV / mandatory timeout reclassification analysis."""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from kret_matplotlib.UTILS_Matplotlib import UTILS_Plotting as UKS_MPL

from nba_timeout_impact.data_pipes.tv_timeout_injection import ValidationResult

# Pre-2017 trigger marks (Q2 / Q4): 8:59, 5:59, 2:59 → sr boundaries 540 / 360 / 180.
PRE_2017_TRIGGERS = [
    (540, "slot 1 (8:59)", "tab:red"),
    (360, "slot 2 (5:59)", "tab:purple"),
    (180, "slot 3 (2:59)", "tab:orange"),
]

# Post-2017 trigger marks (Q1-Q4): 6:59, 2:59 → sr boundaries 420 / 180.
POST_2017_TRIGGERS = [
    (420, "slot 1 (6:59)", "tab:red"),
    (180, "slot 2 (2:59)", "tab:orange"),
]

# Colors used across the cdnnba diagnostic plots — kept consistent so the
# same role reads the same across panels.
ROLE_COLORS = {
    "slot_1_mandatory": "tab:green",
    "slot_2_mandatory": "tab:olive",
    "slot_3_mandatory": "tab:cyan",
    "discretionary": "tab:gray",
    "challenge": "tab:purple",
}


class TimeoutInjectionPlots:
    """Static plotting helpers for `TVTimeoutValidation` outputs."""

    @staticmethod
    def _filter_q2q4_timeouts(classified: pl.DataFrame, periods: tuple[int, ...]) -> pl.DataFrame:
        return classified.filter(
            (pl.col("actionType").cast(pl.String).str.strip_chars() == "Timeout")
            & pl.col("period").is_in(list(periods))
        ).with_columns(
            pl.col("subType").cast(pl.String).str.strip_chars().alias("_gt_sub"),
        )

    @staticmethod
    def _split_tp_fp_fn(tos: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        is_gt = pl.col("_gt_sub").is_in(["Official", "Official TV"])
        is_pred = pl.col("timeout_role").str.contains("_mandatory")
        tp = tos.filter(is_gt & is_pred)["seconds_remaining"].to_numpy()
        fp = tos.filter(~is_gt & is_pred)["seconds_remaining"].to_numpy()
        fn = tos.filter(is_gt & ~is_pred)["seconds_remaining"].to_numpy()
        return tp, fp, fn

    @staticmethod
    def plot_gt_vs_predicted_overlap(
        r_v3: ValidationResult,
        classified_v3: pl.DataFrame,
        widths: tuple[int, ...] = (60,),
        periods: tuple[int, ...] = (2, 4),
        width_per: float = 14,
        height_per: float = 4,
    ):
        """Two-color overlap histogram: v3 ground-truth mandatories (Official /
        Official TV) vs predicted slot_K_mandatory rows.

        ``r_v3``: greedy-match ``ValidationResult`` whose TP/FP/FN annotate the box.
        ``classified_v3``: output of ``TVTimeoutValidation.classify_timeouts``
        on the v3 era (must contain ``timeout_role`` + ``subType`` columns).
        ``widths``: one bin-width per panel; the figure stacks ``len(widths)``
        rows vertically with shared x-axis.
        """
        tos = TimeoutInjectionPlots._filter_q2q4_timeouts(classified_v3, periods)
        gt = tos.filter(pl.col("_gt_sub").is_in(["Official", "Official TV"]))["seconds_remaining"].to_numpy()
        pred = tos.filter(pl.col("timeout_role").str.contains("_mandatory"))["seconds_remaining"].to_numpy()

        fig, axes = UKS_MPL.subplots(1, len(widths), width_per=width_per, height_per=height_per, sharex=True)
        for i, (ax, width) in enumerate(zip(axes, widths)):
            bins = np.arange(0, 720 + width, width)
            ax.hist(gt, bins=bins, alpha=0.6, color="C0", label=f"v3 Official / Official TV (n={len(gt):,})")  # type: ignore[arg-type]
            ax.hist(pred, bins=bins, alpha=0.6, color="C1", label=f"predicted slot_K_mandatory (n={len(pred):,})")  # type: ignore[arg-type]
            for x, lbl, c in PRE_2017_TRIGGERS:
                ax.axvline(x, color=c, linestyle="--", linewidth=1, label=lbl)
            ax.set_ylabel(f"count (bin = {width}s)")
            ax.set_title(
                f"v3 mandatory timeouts in Q{'/'.join(map(str, periods))} "
                f"(seasons {r_v3.seasons[0]}-{r_v3.seasons[1]}), sr_bin = {width}s"
            )
            ax.legend(loc="upper right", fontsize=8)
            if i == len(axes) - 1:
                ax.set_xlabel("seconds remaining in period (bin floor)")

        match_title = "row-by-row match" if r_v3.tolerance_s == 0 else f"fuzzy clock match (tol={r_v3.tolerance_s}s)"
        _annotate_metrics_box(
            axes[0],
            title=match_title,
            tp=r_v3.tp,
            fp=r_v3.fp,
            fn=r_v3.fn,
            p=r_v3.precision,
            r=r_v3.recall,
            f1=r_v3.f1,
        )
        return fig, axes

    @staticmethod
    def plot_stacked_tp_fp_fn(
        r_v3: ValidationResult,
        classified_v3: pl.DataFrame,
        widths: tuple[int, ...] = (60,),
        periods: tuple[int, ...] = (2, 4),
        width_per: float = 14,
        height_per: float = 4,
        colors: tuple[str, str, str] = ("tab:green", "tab:red", "tab:gray"),
    ):
        """Stacked histogram: each sr-bin shows row-by-row TP / FP / FN in
        three non-overlapping colors.

        TP / FP / FN are computed **row-by-row** on ``classified_v3`` (no clock
        tolerance — the same row must be both predicted-mandatory and GT
        Official). ``r_v3`` is accepted only for API symmetry with the overlap
        plot; its greedy metrics aren't used in the figure body. ``widths``
        controls how many vertically stacked panels to render (one per bin
        width).
        """

        tos = TimeoutInjectionPlots._filter_q2q4_timeouts(classified_v3, periods)
        tp_sr, fp_sr, fn_sr = TimeoutInjectionPlots._split_tp_fp_fn(tos)
        tp_n, fp_n, fn_n = len(tp_sr), len(fp_sr), len(fn_sr)
        p = tp_n / max(tp_n + fp_n, 1)
        rec = tp_n / max(tp_n + fn_n, 1)
        f1 = 2 * p * rec / max(p + rec, 1e-9)
        c_tp, c_fp, c_fn = colors

        fig, axes = UKS_MPL.subplots(1, len(widths), width_per=width_per, height_per=height_per, sharex=True)
        axes = list(np.atleast_1d(axes).ravel())
        for i, (ax, width) in enumerate(zip(axes, widths)):
            bins = np.arange(0, 720 + width, width)
            ax.hist(
                [tp_sr, fp_sr, fn_sr],
                bins=bins,  # type: ignore[arg-type]
                stacked=True,
                color=[c_tp, c_fp, c_fn],
                label=[f"TP (n={tp_n:,})", f"FP (n={fp_n:,})", f"FN (n={fn_n:,})"],
                edgecolor="white",
                linewidth=0.3,
            )
            for x, lbl, c in PRE_2017_TRIGGERS:
                ax.axvline(x, color=c, linestyle="--", linewidth=1, label=lbl)
            ax.set_ylabel(f"count (bin = {width}s)")
            ax.set_title(
                f"v3 mandatory row-by-row outcomes in Q{'/'.join(map(str, periods))} "
                f"(seasons {r_v3.seasons[0]}-{r_v3.seasons[1]}), sr_bin = {width}s"
            )
            ax.legend(loc="upper right", fontsize=8)
            if i == len(axes) - 1:
                ax.set_xlabel("seconds remaining in period (bin floor)")

        _annotate_metrics_box(axes[0], title="row-by-row (0s tol)", tp=tp_n, fp=fp_n, fn=fn_n, p=p, r=rec, f1=f1)
        return fig, axes, dict(tp=tp_n, fp=fp_n, fn=fn_n, precision=p, recall=rec, f1=f1)

    # ------------------------------------------------------------------ #
    #  cdnnba diagnostic plots                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _classified_timeouts_pd(classified: pl.DataFrame, action: str = "timeout") -> pd.DataFrame:
        """Pull timeout rows from a classified frame as pandas. Adds a
        ``_wallclock_delta`` column (seconds to next event in the same game)
        if ``timeActual`` is present.
        """
        tos = classified.filter(pl.col("actionType").cast(pl.String).str.strip_chars() == action).to_pandas()
        if "timeActual" in tos.columns:
            tos = tos.sort_values(["gameId", "orderNumber"]).reset_index(drop=True)
            tos["_t"] = pd.to_datetime(tos["timeActual"], utc=True, errors="coerce")
            # Wall-clock delta needs the *next event in the game*, not the next
            # timeout — but we don't have non-timeout rows here. Caller supplies
            # the full classified frame to plot_role_vs_duration if they want
            # the true delta. For now use TO-to-next-TO as a fallback.
            tos["_wallclock_delta"] = tos.groupby("gameId")["_t"].diff(-1).dt.total_seconds().mul(-1)
        return tos

    @staticmethod
    def plot_role_vs_sr(
        classified_cdnnba: pl.DataFrame,
        *,
        action: str = "timeout",
        width: int = 15,
        periods: tuple[int, ...] = (1, 2, 3, 4),
        height_per: float = 3,
        width_per: float = 14,
    ):
        """Histogram of predicted ``timeout_role`` distribution vs
        ``seconds_remaining``. One row per period.

        Use this to spot whether the classifier's predicted mandatories
        cluster at the rulebook trigger marks (6:59 / 2:59) the way the
        empirical data should, and whether ``discretionary`` predictions
        sit where they should (clutch time, pre-trigger).
        """
        tos = TimeoutInjectionPlots._classified_timeouts_pd(classified_cdnnba, action=action)
        tos = tos[tos["period"].isin(list(periods))].copy()
        roles_present = [r for r in ROLE_COLORS if (tos["timeout_role"] == r).any()]

        fig, axes = UKS_MPL.subplots(1, len(periods), width_per=width_per, height_per=height_per, sharex=True)
        axes = list(np.atleast_1d(axes).ravel())
        bins = np.arange(0, 720 + width, width)
        for ax, per in zip(axes, periods):
            sub = tos[tos["period"] == per]
            stacks = [sub.loc[sub["timeout_role"] == r, "seconds_remaining"].to_numpy() for r in roles_present]
            colors = [ROLE_COLORS[r] for r in roles_present]
            labels = [f"{r} (n={len(s):,})" for r, s in zip(roles_present, stacks)]
            ax.hist(stacks, bins=bins, stacked=True, color=colors, label=labels, edgecolor="white", linewidth=0.3)
            for x, lbl, c in POST_2017_TRIGGERS:
                ax.axvline(x, color=c, linestyle="--", linewidth=1)
            ax.set_title(f"Period {per}  (n={len(sub):,})")
            ax.set_ylabel(f"count (bin = {width}s)")
            ax.legend(loc="upper left", fontsize=7)
        axes[-1].set_xlabel("seconds remaining in period (bin floor)")
        return fig, axes

    @staticmethod
    def plot_role_vs_duration(
        classified_cdnnba: pl.DataFrame,
        full_pbp_pl: pl.DataFrame | None = None,
        *,
        bin_width: int = 10,
        max_sec: int = 300,
        height_per: float = 4,
        width_per: float = 14,
        exclude_next_action_types: tuple[str, ...] = ("substitution", "stoppage", "instantreplay"),
    ):
        """Stacked histogram of wall-clock duration (seconds to next event)
        split by predicted ``timeout_role``.

        Validates the intuition that mandatory TOs run longer (≈150-200s
        for TV breaks) than regular coach TOs (≈60-120s).

        IMPORTANT: cdnnba logs substitutions (and instant-replay rows) at
        the START of the TV break — before actual game resumption. If you
        naively diff timeouts to the literal next pbp row, you get a
        spurious 0-10s spike of "fast mandatories" that are really just
        followed by a sub. By default we **exclude** ``substitution`` /
        ``stoppage`` / ``instantreplay`` from "next event" computation,
        so deltas reflect time to actual game resumption. Pass
        ``exclude_next_action_types=()`` to disable.

        ``full_pbp_pl``: full play-by-play (needed for timeActual + correct
        "next event" lookup). Required for this plot to be meaningful;
        without it, falls back to TO-to-TO deltas which are too coarse.
        """
        if full_pbp_pl is not None and "timeActual" in full_pbp_pl.columns:
            df = full_pbp_pl.to_pandas().sort_values(["gameId", "orderNumber"]).reset_index(drop=True)
            if exclude_next_action_types:
                excl_set = set(exclude_next_action_types)
                df = df[~df["actionType"].astype(str).str.strip().isin(excl_set)].copy()
            df["_t"] = pd.to_datetime(df["timeActual"], utc=True, errors="coerce")
            df["_delta"] = df.groupby("gameId")["_t"].diff(-1).dt.total_seconds().mul(-1)
            cls = classified_cdnnba.to_pandas()
            tos = cls.merge(df[["gameId", "orderNumber", "_delta"]], on=["gameId", "orderNumber"], how="left")
            tos = tos[tos["actionType"].astype(str).str.strip().str.lower() == "timeout"]
        else:
            tos = TimeoutInjectionPlots._classified_timeouts_pd(classified_cdnnba)
            tos["_delta"] = tos.get("_wallclock_delta")

        roles_present = [r for r in ROLE_COLORS if (tos["timeout_role"] == r).any()]
        bins = np.arange(0, max_sec + bin_width, bin_width)

        fig, ax = UKS_MPL.subplots(1, 1, width_per=width_per, height_per=height_per)
        stacks = []
        labels = []
        colors = []
        for r in roles_present:
            d = tos.loc[tos["timeout_role"] == r, "_delta"].dropna()
            d = d.clip(0, max_sec)
            stacks.append(d.to_numpy())
            n = len(d)
            med = float(np.median(d)) if n else 0.0
            labels.append(f"{r}  n={n:,}  median={med:.0f}s")
            colors.append(ROLE_COLORS[r])
        ax.hist(
            stacks,
            bins=bins,  # type: ignore
            stacked=True,
            color=colors,
            label=labels,
            edgecolor="white",
            linewidth=0.3,
        )
        ax.set_xlabel(f"wall-clock seconds to next event (clipped at {max_sec}, bin = {bin_width}s)")
        ax.set_ylabel(f"count")
        ax.set_title("Predicted timeout role vs wall-clock duration")
        ax.legend(loc="upper right", fontsize=8)
        return fig, ax

    @staticmethod
    def plot_role_counts_per_period(
        classified_cdnnba: pl.DataFrame,
        *,
        action: str = "timeout",
        height_per: float = 4,
        width_per: float = 12,
    ):
        """Per-period predicted role counts and per-(game, period) mandatory
        count distribution. Reveals whether the classifier matches the
        rulebook expectation of ≈ 2 mandatories per regular Q and 1 per OT.
        """
        tos = TimeoutInjectionPlots._classified_timeouts_pd(classified_cdnnba, action=action)

        fig, axes = UKS_MPL.subplots(1, 2, width_per=width_per, height_per=height_per)
        axes = list(np.atleast_1d(axes).ravel())

        # Panel 1: stacked bar of role counts by period
        roles_present = [r for r in ROLE_COLORS if (tos["timeout_role"] == r).any()]
        per_period = (
            tos.groupby(["period", "timeout_role"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=roles_present, fill_value=0)
        )
        bottoms = np.zeros(len(per_period))
        x = np.arange(len(per_period))
        for r in roles_present:
            vals = per_period[r].values
            axes[0].bar(x, vals, bottom=bottoms, label=r, color=ROLE_COLORS[r], edgecolor="white", linewidth=0.5)
            bottoms = bottoms + vals  # type: ignore
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f"P{p}" for p in per_period.index])
        axes[0].set_xlabel("period")
        axes[0].set_ylabel("predicted count")
        axes[0].set_title("Predicted role counts by period")
        axes[0].legend(loc="upper right", fontsize=8)

        # Panel 2: per-(game, period) mandatory-count distribution
        is_mand = tos["timeout_role"].str.contains("_mandatory", na=False)
        per_gp = tos.loc[is_mand].groupby(["gameId", "period"]).size()
        counts = per_gp.value_counts().sort_index()
        # Also include 0-count game-periods (no mandatory predicted)
        all_gp = tos.groupby(["gameId", "period"]).size().index
        zero_count = len(set(all_gp) - set(per_gp.index))
        if zero_count:
            counts = pd.Series({0: zero_count, **counts.to_dict()}).sort_index()
        axes[1].bar(counts.index.astype(int), counts.values, color="tab:green", edgecolor="white", linewidth=0.5)
        axes[1].set_xlabel("# predicted mandatories per (game, period)")
        axes[1].set_ylabel("count of (game, period) buckets")
        axes[1].set_title("Mandatory-per-period distribution")
        for k, v in zip(counts.index, counts.values):
            axes[1].text(int(k), int(v), f"{int(v):,}", ha="center", va="bottom", fontsize=9)
        return fig, axes

    @staticmethod
    def plot_role_vs_prev_action_type(
        classified_cdnnba: pl.DataFrame,
        full_pbp_pl: pl.DataFrame | None = None,
        *,
        normalize: Literal["within_role", "within_action", "count"] = "within_role",
        top_k: int = 12,
        height_per: float = 5,
        width_per: float = 14,
    ):
        """Density of predicted ``timeout_role`` across ``prev_action_type``.

        Useful for spotting non-TV mandatories: if a particular
        ``prev_action_type`` (e.g., ``instantreplay`` or ``ejection``)
        skews towards mandatory but shouldn't, that's a flag worth
        investigating.

        ``normalize``:
            - ``"within_role"`` (default): each role's bars sum to 1. Shows
              "given a mandatory TO, what fraction were preceded by X?"
            - ``"within_action"``: each prev_action bar sums to 1. Shows
              "given a TO that follows X, what fraction is mandatory?"
            - ``"count"``: raw counts.

        ``full_pbp_pl``: optional source of ``prev_action_type``. If the
        classified frame already has it (as cdnnba enriched does), it's
        used directly.
        """
        cls = classified_cdnnba.to_pandas()
        if "prev_action_type" not in cls.columns:
            if full_pbp_pl is None or "prev_action_type" not in full_pbp_pl.columns:
                raise ValueError("prev_action_type not in classified frame; pass full_pbp_pl with the column")
            cls = cls.merge(
                full_pbp_pl.select(["gameId", "orderNumber", "prev_action_type"]).to_pandas(),
                on=["gameId", "orderNumber"],
                how="left",
            )
        tos = cls[cls["actionType"].astype(str).str.strip().str.lower() == "timeout"].copy()
        tos["prev"] = tos["prev_action_type"].astype("string").fillna("(none)")

        # Top-K prev_action_types by total volume
        top = tos["prev"].value_counts().head(top_k).index.tolist()
        tos = tos[tos["prev"].isin(top)].copy()

        roles_present = [r for r in ROLE_COLORS if (tos["timeout_role"] == r).any()]
        ct = pd.crosstab(tos["prev"], tos["timeout_role"]).reindex(index=top, columns=roles_present, fill_value=0)

        if normalize == "within_role":
            mat = ct.div(ct.sum(axis=0), axis=1).fillna(0)
            ylabel = "share within role"
        elif normalize == "within_action":
            mat = ct.div(ct.sum(axis=1), axis=0).fillna(0)
            ylabel = "share within prev_action_type"
        else:
            mat = ct
            ylabel = "count"

        fig, ax = UKS_MPL.subplots(1, 1, width_per=width_per, height_per=height_per)
        x = np.arange(len(mat.index))
        n_roles = len(roles_present)
        bar_w = 0.8 / max(n_roles, 1)
        for i, role in enumerate(roles_present):
            ax.bar(
                x + (i - (n_roles - 1) / 2) * bar_w,
                mat[role].values,  # type: ignore
                bar_w,
                color=ROLE_COLORS[role],
                edgecolor="white",
                linewidth=0.4,
                label=role,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(mat.index, rotation=30, ha="right", fontsize=9)
        ax.set_xlabel("prev_action_type")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Timeout role density vs prev_action_type (normalize={normalize}, top {top_k})")
        ax.legend(loc="upper right", fontsize=8)
        return fig, ax, mat

    @staticmethod
    def plot_team_mandatory_share(
        classified_cdnnba: pl.DataFrame,
        *,
        action: str = "timeout",
        height_per: float = 5,
        width_per: float = 14,
    ):
        """Per-team share of predicted mandatory vs non-mandatory TOs.

        If the classifier (or the underlying ``qualifiers`` field) is
        unbiased, every team should split roughly the same way between
        mandatory-charged and purely discretionary. Outliers worth
        investigating: a team with much more discretionary share might
        be a fast-paced team that calls more clutch-time TOs.
        """
        tos = TimeoutInjectionPlots._classified_timeouts_pd(classified_cdnnba, action=action)
        if "teamTricode" not in tos.columns:
            raise ValueError("plot_team_mandatory_share requires teamTricode column in input")
        tos = tos[tos["teamTricode"].astype(str).str.strip() != ""]
        tos["is_mand"] = tos["timeout_role"].str.contains("_mandatory", na=False)

        by_team = tos.groupby("teamTricode")["is_mand"].agg(n_mand="sum", n_total="size")
        by_team["n_disc"] = by_team["n_total"] - by_team["n_mand"]
        by_team["mand_share"] = by_team["n_mand"] / by_team["n_total"]
        by_team = by_team.sort_values("mand_share", ascending=False)  # type: ignore

        fig, ax = UKS_MPL.subplots(1, 1, width_per=width_per, height_per=height_per)
        x = np.arange(len(by_team))
        ax.bar(x, by_team["n_mand"], color="tab:green", label="mandatory", edgecolor="white", linewidth=0.5)
        ax.bar(
            x,
            by_team["n_disc"],
            bottom=by_team["n_mand"],
            color="tab:gray",
            label="discretionary",
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(by_team.index, rotation=70, fontsize=8)
        ax.set_ylabel("predicted timeout count")
        ax.set_title("Per-team predicted mandatory vs discretionary timeouts")
        ax.legend(loc="upper right", fontsize=8)
        # Annotate mandatory-share %
        for xi, share in zip(x, by_team["mand_share"]):
            ax.text(xi, by_team["n_total"].iloc[xi] + 5, f"{share:.0%}", ha="center", va="bottom", fontsize=7)  # type: ignore
        return fig, ax, by_team

    @staticmethod
    def diagnose_cdnnba(
        classified_cdnnba: pl.DataFrame,
        full_pbp_pl: pl.DataFrame | None = None,
    ) -> dict[str, plt.Figure]:
        """Run all cdnnba diagnostic plots and return them by name.

        Convenience wrapper for the notebook: one call → four figures
        (sr distribution, wall-clock duration, per-period counts, per-team
        share). The caller can display each figure individually.
        """
        figs: dict[str, plt.Figure] = {}
        fig, _ = TimeoutInjectionPlots.plot_role_vs_sr(classified_cdnnba)
        figs["role_vs_sr"] = fig
        fig, _ = TimeoutInjectionPlots.plot_role_vs_duration(classified_cdnnba, full_pbp_pl)
        figs["role_vs_duration"] = fig
        fig, _ = TimeoutInjectionPlots.plot_role_counts_per_period(classified_cdnnba)
        figs["role_counts_per_period"] = fig
        try:
            fig, _, _ = TimeoutInjectionPlots.plot_role_vs_prev_action_type(classified_cdnnba, full_pbp_pl)
            figs["role_vs_prev_action_type"] = fig
        except ValueError:
            pass  # prev_action_type not available — skip
        try:
            fig, _, _ = TimeoutInjectionPlots.plot_team_mandatory_share(classified_cdnnba)
            figs["team_mandatory_share"] = fig
        except ValueError:
            pass  # teamTricode not present — skip
        return figs


def _annotate_metrics_box(ax: plt.Axes, *, title: str, tp: int, fp: int, fn: int, p: float, r: float, f1: float):
    text = (
        f"{title}\n"
        f"TP = {tp:,}\n"
        f"FP = {fp:,}\n"
        f"FN = {fn:,}\n"
        f"P  = {p:.3f}\n"
        f"R  = {r:.3f}\n"
        f"F1 = {f1:.3f}"
    )
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.85),
    )
