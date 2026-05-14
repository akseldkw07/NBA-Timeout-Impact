"""Plots for the TV / mandatory timeout reclassification analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from kret_matplotlib.UTILS_Matplotlib import UTILS_Plotting as UKS_MPL

from nba_timeout_impact.analyses.tv_timeout_validation import ValidationResult

# Pre-2017 trigger marks (Q2 / Q4): 8:59, 5:59, 2:59 → sr boundaries 540 / 360 / 180.
PRE_2017_TRIGGERS = [
    (540, "slot 1 (8:59)", "tab:red"),
    (360, "slot 2 (5:59)", "tab:purple"),
    (180, "slot 3 (2:59)", "tab:orange"),
]


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
        axes = list(np.atleast_1d(axes).ravel())
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

        _annotate_metrics_box(
            axes[0],
            title=f"per-event greedy match (tol={r_v3.tolerance_s}s)",
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
