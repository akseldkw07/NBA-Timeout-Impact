"""Generalized likelihood ratio tests for the timeout-impact metrics.

We model each metric (excess net pts, excess PPP, WP added) as iid samples
from a Gaussian with unknown variance and test:

    H0: mu = 0   (timeout has no effect)
    H1: mu != 0  (timeout shifts the metric)

Derivation
----------
Let X_1, ..., X_n iid ~ N(mu, sigma^2). The Gaussian log-likelihood is

    log L(mu, sigma^2) = -n/2 log(2 pi sigma^2) - (1/(2 sigma^2)) sum (x_i - mu)^2.

Under H0 the MLE for sigma^2 is sigma_hat_0^2 = (1/n) sum x_i^2.
Under H1 the MLEs are mu_hat_1 = x_bar, sigma_hat_1^2 = (1/n) sum (x_i - x_bar)^2.

Both maximized log-likelihoods reduce to the same constant -n/2 (1 + log(2 pi sigma_hat^2))
because the residual sum of squares enters only via sigma_hat^2. The likelihood
ratio Lambda is therefore

    Lambda = sup_H0 L / sup_H1 L = (sigma_hat_1^2 / sigma_hat_0^2)^(n/2).

Since sigma_hat_0^2 = sigma_hat_1^2 + x_bar^2, the GLRT statistic simplifies to

    G = -2 log Lambda = n log(1 + x_bar^2 / sigma_hat_1^2).

In terms of the one-sample t-statistic t = x_bar / (s / sqrt(n)) with the
unbiased estimate s^2 = (n / (n - 1)) sigma_hat_1^2,

    G = n log(1 + t^2 / (n - 1)).

For large n, G is asymptotically chi-squared with 1 degree of freedom under H0;
we reject H0 at level alpha when G > chi2_{1, 1 - alpha}.
"""

from __future__ import annotations

import typing as t
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats as sp_stats

# --------------------------------------------------------------------------- #
#  Core test                                                                   #
# --------------------------------------------------------------------------- #


def glrt_zero_mean_gaussian(x: np.ndarray) -> dict[str, float]:
    """One-sample GLRT for H0: mu == 0 vs H1: mu != 0 with unknown sigma^2.

    Returns dict with keys:
      n, mean, std, sigma2_h0, sigma2_h1, glrt_stat, glrt_p,
      t_stat, t_p, ci_low, ci_high.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = int(x.size)
    if n < 2:
        return {
            "n": n,
            "mean": float("nan"),
            "std": float("nan"),
            "sigma2_h0": float("nan"),
            "sigma2_h1": float("nan"),
            "glrt_stat": float("nan"),
            "glrt_p": float("nan"),
            "t_stat": float("nan"),
            "t_p": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }

    mean = float(x.mean())
    s2_unbiased = float(x.var(ddof=1))
    sigma2_h1 = float(x.var(ddof=0))
    sigma2_h0 = sigma2_h1 + mean * mean

    # GLRT statistic (large-n chi^2(1) reference)
    if sigma2_h1 <= 0:
        glrt_stat = float("inf") if mean != 0 else 0.0
    else:
        glrt_stat = n * float(np.log1p(mean * mean / sigma2_h1))
    glrt_p = float(1.0 - sp_stats.chi2.cdf(glrt_stat, df=1))

    # Cross-check: one-sample Welch t-test (exact under Gaussian)
    if s2_unbiased > 0:
        t_stat = mean / np.sqrt(s2_unbiased / n)
        t_p = float(2.0 * (1.0 - sp_stats.t.cdf(abs(t_stat), df=n - 1)))
    else:
        t_stat = float("inf") if mean != 0 else 0.0
        t_p = 0.0 if mean != 0 else 1.0

    # 95% CI via t distribution
    if s2_unbiased > 0:
        half = sp_stats.t.ppf(0.975, df=n - 1) * np.sqrt(s2_unbiased / n)
    else:
        half = 0.0

    return {
        "n": n,
        "mean": mean,
        "std": float(np.sqrt(s2_unbiased)),
        "sigma2_h0": sigma2_h0,
        "sigma2_h1": sigma2_h1,
        "glrt_stat": float(glrt_stat),
        "glrt_p": glrt_p,
        "t_stat": float(t_stat),
        "t_p": t_p,
        "ci_low": mean - float(half),
        "ci_high": mean + float(half),
    }


def critical_value(alpha: float = 0.05) -> float:
    """chi^2(1) critical value at level alpha (default 0.05 -> 3.841)."""
    return float(sp_stats.chi2.ppf(1.0 - alpha, df=1))


# --------------------------------------------------------------------------- #
#  Per-group runner                                                            #
# --------------------------------------------------------------------------- #


def run_glrt_groups(
    events: pl.DataFrame,
    metric: str,
    group_cols: t.Sequence[str] = (),
    *,
    min_n: int = 30,
) -> pl.DataFrame:
    """Apply the GLRT per group on ``events[metric]``.

    Returns one row per group with the GLRT statistics, t-test cross-check,
    and 95% CI. Drops groups with fewer than ``min_n`` non-null values.
    """
    group_cols = list(group_cols)
    df = events.filter(pl.col(metric).is_not_null())

    rows: list[dict[str, t.Any]] = []
    if not group_cols:
        stats = glrt_zero_mean_gaussian(df[metric].to_numpy())
        rows.append({"group": "all", **stats})
    else:
        # Distinct group values
        keys = df.select(group_cols).unique().sort(group_cols)
        for key_row in keys.iter_rows(named=True):
            mask = pl.lit(True)
            for c in group_cols:
                mask = mask & (pl.col(c) == key_row[c])
            sub = df.filter(mask)
            if sub.height < min_n:
                continue
            stats = glrt_zero_mean_gaussian(sub[metric].to_numpy())
            rows.append({**key_row, **stats})

    out = pl.DataFrame(rows) if rows else pl.DataFrame(schema={"n": pl.Int64})
    return out


def add_decision_columns(table: pl.DataFrame, alpha: float = 0.05) -> pl.DataFrame:
    """Add ``crit``, ``reject_h0``, ``signif_label`` columns at level alpha."""
    crit = critical_value(alpha)
    return table.with_columns(
        pl.lit(crit).alias("crit"),
        (pl.col("glrt_stat") > crit).alias("reject_h0"),
        pl.when(pl.col("glrt_p") < 0.001)
        .then(pl.lit("***"))
        .when(pl.col("glrt_p") < 0.01)
        .then(pl.lit("**"))
        .when(pl.col("glrt_p") < 0.05)
        .then(pl.lit("*"))
        .otherwise(pl.lit("ns"))
        .alias("signif"),
    )


# --------------------------------------------------------------------------- #
#  Plotting                                                                    #
# --------------------------------------------------------------------------- #


_SUBTYPE_COLORS = {
    "full": "#1f77b4",
    "challenge": "#ff7f0e",
    "official_inferred": "#2ca02c",
}


def plot_glrt_overall(
    table: pl.DataFrame,
    *,
    title: str,
    metric_label: str,
    save_path: str | Path | None = None,
    alpha: float = 0.05,
) -> plt.Figure:
    """Bar chart of GLRT statistic by timeout subtype with critical-value line.

    ``table`` should have one row per ``timeout_subtype`` (output of
    ``run_glrt_groups`` with ``group_cols=['timeout_subtype']``).
    """
    table = add_decision_columns(table, alpha=alpha)
    subs = table["timeout_subtype"].to_list()
    stats = table["glrt_stat"].to_list()
    pvals = table["glrt_p"].to_list()
    ns = table["n"].to_list()
    means = table["mean"].to_list()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1]})
    colors = [_SUBTYPE_COLORS.get(s, "#999") for s in subs]

    bars = ax1.bar(subs, stats, color=colors)
    ax1.axhline(critical_value(alpha), color="red", linestyle="--", lw=1.0, label=f"chi^2(1) crit @ alpha={alpha}")
    ax1.set_yscale("log")
    ax1.set_ylabel("GLRT statistic G = -2 log Lambda")
    ax1.set_title(f"{title}\nGLRT against H0: mu=0")
    for bar, p, n in zip(bars, pvals, ns):
        sig = _sig_label(p)
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            f"{sig}\nn={n:,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax1.legend(loc="upper right", fontsize=8)

    # Right panel: mean +/- 95% CI
    lows = [m - lo for m, lo in zip(means, table["ci_low"].to_list())]
    highs = [hi - m for m, hi in zip(means, table["ci_high"].to_list())]
    yerr = np.array([lows, highs])
    ax2.errorbar(subs, means, yerr=yerr, fmt="o", capsize=4, color="black")
    ax2.axhline(0, color="red", linestyle="--", lw=1.0, label="H0: mu = 0")
    ax2.set_ylabel(metric_label)
    ax2.set_title("Sample mean +/- 95% CI")
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, format="jpg", dpi=200, bbox_inches="tight")
    return fig


def plot_glrt_by_slice(
    table: pl.DataFrame,
    *,
    slice_col: str,
    slice_order: t.Sequence[str],
    title: str,
    metric_label: str,
    save_path: str | Path | None = None,
    alpha: float = 0.05,
) -> plt.Figure:
    """Forest plot: mean +/- 95% CI per (subtype, slice_value).

    ``table`` should have rows for ``[timeout_subtype, slice_col]``.
    """
    table = add_decision_columns(table, alpha=alpha)

    subtypes = ["full", "challenge", "official_inferred"]
    fig, ax = plt.subplots(figsize=(10, max(4, len(slice_order) * 0.6 + 1)))

    n_subs = len(subtypes)
    y_pos_per_slice = np.arange(len(slice_order))
    offset = 0.22
    crit = critical_value(alpha)

    for i, sub in enumerate(subtypes):
        rows = table.filter(pl.col("timeout_subtype") == sub)
        means: list[float] = []
        ys: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        markers: list[bool] = []
        for s_idx, s_val in enumerate(slice_order):
            row = rows.filter(pl.col(slice_col) == s_val)
            if row.height == 0:
                continue
            r = row.row(0, named=True)
            means.append(r["mean"])
            ys.append(s_idx + (i - (n_subs - 1) / 2) * offset)
            lows.append(r["mean"] - r["ci_low"])
            highs.append(r["ci_high"] - r["mean"])
            markers.append(bool(r["glrt_stat"] > crit))
        if not means:
            continue
        yerr = np.array([lows, highs])
        c = _SUBTYPE_COLORS.get(sub, "#999")
        ax.errorbar(
            means,
            ys,
            xerr=yerr,
            fmt="o",
            capsize=3,
            color=c,
            label=sub,
            markersize=6,
            elinewidth=1.5,
        )
        # bold-fill significant points
        sig_means = [m for m, mk in zip(means, markers) if mk]
        sig_ys = [y for y, mk in zip(ys, markers) if mk]
        if sig_means:
            ax.scatter(sig_means, sig_ys, color=c, s=110, edgecolor="black", lw=1.2, zorder=3)

    ax.axvline(0, color="red", linestyle="--", lw=1.0, alpha=0.7, label="H0: mu=0")
    ax.set_yticks(y_pos_per_slice)
    ax.set_yticklabels(slice_order)
    ax.set_xlabel(metric_label)
    ax.set_title(f"{title}\nForest of mean +/- 95% CI; bold = reject H0 at alpha={alpha}")
    ax.legend(loc="best", fontsize=8)
    ax.invert_yaxis()
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, format="jpg", dpi=200, bbox_inches="tight")
    return fig


def plot_glrt_distribution(
    glrt_stats: t.Sequence[float],
    *,
    title: str,
    save_path: str | Path | None = None,
    alpha: float = 0.05,
    clip: float = 20.0,
) -> plt.Figure:
    """Histogram of observed GLRT statistics overlaid on the chi^2(1) reference.

    G is clipped to ``clip`` on the x-axis so the chi^2(1) reference and the
    bulk of the distribution remain visible; values exceeding the clip are
    reported in the legend (they all have p << alpha and are unambiguous
    rejections regardless of exact magnitude).
    """
    fig, ax = plt.subplots(figsize=(8, 4.2))
    stats_arr = np.asarray([s for s in glrt_stats if np.isfinite(s)])
    if stats_arr.size == 0:
        ax.set_title(f"{title}\n(no statistics available)")
        if save_path is not None:
            fig.savefig(save_path, format="jpg", dpi=200, bbox_inches="tight")
        return fig

    in_view = stats_arr[stats_arr <= clip]
    n_excluded = int((stats_arr > clip).sum())
    bins = np.linspace(0, clip, 40)
    obs_label = "observed G"
    if n_excluded:
        obs_label += f" ({n_excluded} of {stats_arr.size} > {clip:g}, not shown)"
    ax.hist(in_view, bins=bins, density=True, alpha=0.55, color="#4c72b0", label=obs_label)

    xs = np.linspace(0.05, clip, 200)
    ax.plot(xs, sp_stats.chi2.pdf(xs, df=1), "k-", lw=1.2, label="chi^2(1) (asymptotic null)")
    ax.axvline(critical_value(alpha), color="red", linestyle="--", lw=1.0, label=f"crit @ alpha={alpha}")
    ax.set_xlabel("GLRT statistic G")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.set_xlim(0, clip)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, format="jpg", dpi=200, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _sig_label(p: float) -> str:
    if not np.isfinite(p):
        return "?"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"
