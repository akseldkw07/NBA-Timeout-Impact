"""Plots for the TV / mandatory timeout reclassification analysis."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from nba_timeout_impact.analyses.tv_timeout_validation import (
    TVTimeoutValidation,
    ValidationResult,
)


# Pre-2017 trigger marks (Q2 / Q4): 8:59, 5:59, 2:59 → sr boundaries 540 / 360 / 180.
PRE_2017_TRIGGERS = [
    (540, "slot 1 (8:59)", "tab:red"),
    (360, "slot 2 (5:59)", "tab:purple"),
    (180, "slot 3 (2:59)", "tab:orange"),
]

_BIN_WIDTHS = (15, 30, 60)


class TimeoutInjectionPlots:
    """Static plotting helpers for `TVTimeoutValidation` outputs."""

    @staticmethod
    def _classify_v3_rows(
        memo_v3,
        seasons: tuple[int, int],
        pre_2017_mode: str,
        periods: tuple[int, ...] = (2, 4),
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (tp_sr, fp_sr, fn_sr) numpy arrays of seconds_remaining values.

        TP / FP / FN are computed **row-by-row** on the v3 timeout-row population:
            TP  = row predicted mandatory AND GT subType in (Official, Official TV)
            FP  = row predicted mandatory AND GT subType NOT in those
            FN  = row predicted NOT mandatory AND GT subType in those
        """
        v3_pl = TVTimeoutValidation._prep_v3(memo_v3, seasons)
        classified = TVTimeoutValidation.classify_timeouts(
            v3_pl, source="v3", seasons=seasons, pre_2017_mode=pre_2017_mode
        )
        tos = classified.filter(
            (pl.col("actionType").cast(pl.String).str.strip_chars() == "Timeout")
            & pl.col("period").is_in(list(periods))
        ).with_columns(
            pl.col("subType").cast(pl.String).str.strip_chars().alias("_gt_sub"),
        )
        is_gt = pl.col("_gt_sub").is_in(["Official", "Official TV"])
        is_pred = pl.col("timeout_role").str.contains("_mandatory")

        tp = tos.filter(is_gt & is_pred)["seconds_remaining"].to_numpy()
        fp = tos.filter(~is_gt & is_pred)["seconds_remaining"].to_numpy()
        fn = tos.filter(is_gt & ~is_pred)["seconds_remaining"].to_numpy()
        return tp, fp, fn

    @staticmethod
    def plot_gt_vs_predicted_overlap(
        memo_v3,
        seasons: tuple[int, int] = (2013, 2016),
        pre_2017_mode: str = "cascading",
        tolerance_s: int = 60,
        bin_widths: tuple[int, ...] = _BIN_WIDTHS,
        figsize: tuple[float, float] = (11, 9),
    ):
        """Two-color overlap histogram: GT mandatories vs predicted mandatories
        (slot_K_mandatory) in Q2/Q4, with greedy-match TP/FP/FN annotated.
        """
        r = TVTimeoutValidation.validate_against_v3(
            memo_v3, seasons=seasons, pre_2017_mode=pre_2017_mode, tolerance_s=tolerance_s
        )
        v3_pl = TVTimeoutValidation._prep_v3(memo_v3, seasons)
        classified = TVTimeoutValidation.classify_timeouts(
            v3_pl, source="v3", seasons=seasons, pre_2017_mode=pre_2017_mode
        )
        gt = classified.filter(
            (pl.col("actionType").cast(pl.String).str.strip_chars() == "Timeout")
            & pl.col("subType").cast(pl.String).str.strip_chars().is_in(["Official", "Official TV"])
            & pl.col("period").is_in([2, 4])
        )["seconds_remaining"].to_numpy()
        pred = classified.filter(
            pl.col("timeout_role").str.contains("_mandatory") & pl.col("period").is_in([2, 4])
        )["seconds_remaining"].to_numpy()

        fig, axes = plt.subplots(len(bin_widths), 1, figsize=figsize, sharex=True)
        if len(bin_widths) == 1:
            axes = [axes]
        for ax, width in zip(axes, bin_widths):
            bins = np.arange(0, 720 + width, width)
            ax.hist(gt, bins=bins, alpha=0.6, color="C0", label=f"v3 Official / Official TV (n={len(gt):,})")
            ax.hist(pred, bins=bins, alpha=0.6, color="C1", label=f"predicted slot_K_mandatory (n={len(pred):,})")
            for x, lbl, c in PRE_2017_TRIGGERS:
                ax.axvline(x, color=c, linestyle="--", linewidth=1, label=lbl)
            ax.set_ylabel(f"count (bin = {width}s)")
            ax.set_title(
                f"v3 mandatory timeouts in Q2/Q4 (seasons {seasons[0]}-{seasons[1]}), "
                f"sr_bin = {width}s, mode={pre_2017_mode}"
            )
            ax.legend(loc="upper right", fontsize=7)
        axes[-1].set_xlabel("seconds remaining in period (bin floor)")

        _annotate_metrics_box(
            axes[0],
            title=f"per-event greedy match (tol={tolerance_s}s)",
            tp=r.tp, fp=r.fp, fn=r.fn,
            p=r.precision, r=r.recall, f1=r.f1,
        )
        plt.tight_layout()
        return fig, axes, r

    @staticmethod
    def plot_stacked_tp_fp_fn(
        memo_v3,
        seasons: tuple[int, int] = (2013, 2016),
        pre_2017_mode: str = "cascading",
        bin_widths: tuple[int, ...] = _BIN_WIDTHS,
        figsize: tuple[float, float] = (11, 9),
        colors: tuple[str, str, str] = ("tab:green", "tab:red", "tab:gray"),
    ):
        """Stacked histogram: each (sr_bin, width) shows TP / FP / FN in three
        non-overlapping colors. TP / FP / FN are computed **row-by-row**
        on the v3 timeout population (no clock tolerance).
        """
        tp_sr, fp_sr, fn_sr = TimeoutInjectionPlots._classify_v3_rows(
            memo_v3, seasons, pre_2017_mode
        )
        tp_n, fp_n, fn_n = len(tp_sr), len(fp_sr), len(fn_sr)
        p = tp_n / max(tp_n + fp_n, 1)
        rec = tp_n / max(tp_n + fn_n, 1)
        f1 = 2 * p * rec / max(p + rec, 1e-9)

        c_tp, c_fp, c_fn = colors
        fig, axes = plt.subplots(len(bin_widths), 1, figsize=figsize, sharex=True)
        if len(bin_widths) == 1:
            axes = [axes]
        for ax, width in zip(axes, bin_widths):
            bins = np.arange(0, 720 + width, width)
            ax.hist(
                [tp_sr, fp_sr, fn_sr],
                bins=bins,
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
                f"v3 mandatory row-by-row outcomes in Q2/Q4 "
                f"(seasons {seasons[0]}-{seasons[1]}), sr_bin = {width}s, mode={pre_2017_mode}"
            )
            ax.legend(loc="upper right", fontsize=7)
        axes[-1].set_xlabel("seconds remaining in period (bin floor)")

        _annotate_metrics_box(
            axes[0],
            title="row-by-row (0s tol)",
            tp=tp_n, fp=fp_n, fn=fn_n, p=p, r=rec, f1=f1,
        )
        plt.tight_layout()
        return fig, axes, dict(tp=tp_n, fp=fp_n, fn=fn_n, precision=p, recall=rec, f1=f1)


def _annotate_metrics_box(ax, *, title, tp, fp, fn, p, r, f1):
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
        0.02, 0.98, text,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.85),
    )
