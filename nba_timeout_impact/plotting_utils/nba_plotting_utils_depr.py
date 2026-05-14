from __future__ import annotations

import typing as t

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from kret_matplotlib.UTILS_Matplotlib import UTILS_Plotting as uks_mpl
from kret_np_pd.UTILS_np_pd import NP_PD_Utils as UKS_NP_PD
from plotly.subplots import make_subplots
from scipy import stats as sp_stats

if t.TYPE_CHECKING:
    from nba_timeout_impact.datasets.memo_cdnnba_pl import CDNNBAMemoPL


class NBAPlottingUtils:
    @classmethod
    def plot_lead_hist(
        cls,
        ax: plt.Axes,
        arr: pd.Series | np.ndarray,
        filter: np.ndarray | t.Sequence[np.ndarray] | None = None,
        **kwargs,
    ):
        filter = (
            UKS_NP_PD.mask_and(*filter)
            if isinstance(filter, t.Sequence)
            else UKS_NP_PD.process_filter(filter, len(arr))
        )
        ax.hist(arr[filter], **kwargs)
        ax.set_title(f"mean= {arr[filter].mean():.2f}, std={arr[filter].std():.2f}, n={filter.sum()}")

    @classmethod
    def plot_hist_many(cls, arr: pd.Series | np.ndarray, filters: t.Sequence[np.ndarray], **kwargs):
        ncols, nrows = uks_mpl.subplots_smart_dims(len(filters))
        fig, axes = uks_mpl.subplots(ncols=ncols, nrows=nrows)
        axes = axes.flatten()

        for i, (filt, ax) in enumerate(zip(filters, axes)):
            cls.plot_lead_hist(ax, arr, filter=filt, **kwargs)

        return fig

    # ------------------------------------------------------------------ #
    #  Post-timeout possession scoring analysis (Plotly)                   #
    # ------------------------------------------------------------------ #

    @classmethod
    def plot_timeout_possession_hist(
        cls,
        memo: CDNNBAMemoPL,
        n_possessions: int = 3,
        timeout_subtype: str | list[str] | None = None,
        *,
        team: t.Literal["calling", "opposing", "both"] = "calling",
        max_seconds_remaining: float | None = None,
        min_seconds_remaining: float | None = None,
        max_score_margin: int | None = None,
        min_score_margin: int | None = None,
        min_streak: int | None = None,
        season_type: t.Literal["rg", "po"] | None = None,
    ) -> go.Figure:
        """Compare points-per-possession after timeouts vs all possessions.

        Returns a single Plotly figure with one subplot per possession offset
        (1 through *n_possessions*), each showing two overlaid histograms:
        "Post-timeout" vs "All possessions".

        Parameters
        ----------
        memo : CDNNBAMemoPL
            Loaded memo object with all datasets.
        n_possessions : int
            Number of possessions after the timeout to show (1-5).
        timeout_subtype : str or list[str], optional
            Filter timeouts by subType (e.g. "full", "official_inferred", "challenge").
            None = all timeouts.
        team : "calling" | "opposing" | "both"
            Which team's possessions to include relative to the timeout-calling team.
            "calling" = only possessions by the team that called the timeout (default).
            "opposing" = only the opponent's possessions.
            "both" = all possessions regardless of team.
            Note: for official_inferred timeouts (no calling team), team filter is skipped.
        max_seconds_remaining, min_seconds_remaining : float, optional
            Filter by seconds remaining in the quarter at timeout time.
        max_score_margin, min_score_margin : int, optional
            Filter by score margin (home - away) at timeout time.
        min_streak : int, optional
            Filter to timeouts called during a scoring run of at least this size (either team).
        season_type : "rg" | "po" | None
            Filter by regular season or playoffs.
        """
        n_possessions = min(n_possessions, 5)

        # --- build filtered post-timeout data ---
        ptp = memo.post_timeout_possessions
        if timeout_subtype is not None:
            if isinstance(timeout_subtype, str):
                timeout_subtype = [timeout_subtype]
            ptp = ptp.filter(pl.col("timeout_subtype").is_in(timeout_subtype))
        if season_type is not None:
            ptp = ptp.filter(pl.col("season_type") == season_type)
        if max_seconds_remaining is not None:
            ptp = ptp.filter(pl.col("seconds_remaining") <= max_seconds_remaining)
        if min_seconds_remaining is not None:
            ptp = ptp.filter(pl.col("seconds_remaining") >= min_seconds_remaining)
        if max_score_margin is not None:
            ptp = ptp.filter(pl.col("score_margin") <= max_score_margin)
        if min_score_margin is not None:
            ptp = ptp.filter(pl.col("score_margin") >= min_score_margin)
        if min_streak is not None:
            poss_table = memo.possessions
            ptp = (
                ptp.join(
                    poss_table.select("gameId", "possession_id", pl.col("streak").alias("_poss_streak")),
                    on=["gameId", "possession_id"],
                    how="left",
                )
                .filter(pl.col("_poss_streak").abs() >= min_streak)
                .drop("_poss_streak")
            )

        # Filter by team relationship to timeout
        if team == "calling":
            # Keep only possessions by the calling team; for official_inferred (null team), keep all
            ptp = ptp.filter(pl.col("is_calling_team_poss") | pl.col("timeout_team").is_null())
        elif team == "opposing":
            ptp = ptp.filter(~pl.col("is_calling_team_poss") | pl.col("timeout_team").is_null())
        # "both" keeps everything

        # --- baseline: all possessions ---
        all_poss = memo.possessions
        if season_type is not None:
            all_poss = all_poss.filter(pl.col("season_type") == season_type)

        # --- build filter description ---
        desc_parts = []
        if timeout_subtype is not None:
            desc_parts.append(f"subtype={timeout_subtype}")
        if team != "both":
            desc_parts.append(f"team={team}")
        if season_type is not None:
            desc_parts.append("Playoffs" if season_type == "po" else "Regular Season")
        if max_seconds_remaining is not None or min_seconds_remaining is not None:
            lo = min_seconds_remaining or 0
            hi = max_seconds_remaining or 720
            desc_parts.append(f"sec_rem=[{lo:.0f},{hi:.0f}]")
        if max_score_margin is not None or min_score_margin is not None:
            lo = min_score_margin if min_score_margin is not None else "..."
            hi = max_score_margin if max_score_margin is not None else "..."
            desc_parts.append(f"margin=[{lo},{hi}]")
        if min_streak is not None:
            desc_parts.append(f"streak>={min_streak}")
        filter_desc = ", ".join(desc_parts) if desc_parts else "all timeouts"

        # --- create plotly figure ---
        fig = make_subplots(
            rows=1,
            cols=n_possessions,
            subplot_titles=[f"Possession +{i}" for i in range(1, n_possessions + 1)],
        )

        bins_dict = dict(start=-0.5, end=7.5, size=1)

        for i in range(1, n_possessions + 1):
            col_idx = i
            post = ptp.filter(pl.col("offset") == i)["possession_points"].to_numpy()
            base = all_poss["possession_points"].to_numpy()

            if len(post) == 0:
                continue

            post_mean, post_std = float(np.mean(post)), float(np.std(post))
            base_mean, base_std = float(np.mean(base)), float(np.std(base))

            hover_tpl = "Points: %{x}<br>Probability: %{y:.3f}<extra>%{fullData.name}</extra>"

            fig.add_trace(
                go.Histogram(
                    x=base,
                    xbins=bins_dict,
                    histnorm="probability",
                    name=f"All poss (n={len(base):,}, \u03bc={base_mean:.3f}, \u03c3={base_std:.3f})",
                    marker_color="rgba(100,149,237,0.5)",
                    hovertemplate=hover_tpl,
                    showlegend=(i == 1),
                    legendgroup="baseline",
                ),
                row=1,
                col=col_idx,
            )
            fig.add_trace(
                go.Histogram(
                    x=post,
                    xbins=bins_dict,
                    histnorm="probability",
                    name=f"Post-TO (n={len(post):,}, \u03bc={post_mean:.3f}, \u03c3={post_std:.3f})",
                    marker_color="rgba(255,99,71,0.6)",
                    hovertemplate=hover_tpl,
                    showlegend=(i == 1),
                    legendgroup="post_timeout",
                ),
                row=1,
                col=col_idx,
            )

            fig.update_xaxes(title_text="Points", row=1, col=col_idx)
            fig.update_yaxes(title_text="Probability" if i == 1 else "", row=1, col=col_idx)

            fig.add_annotation(
                text=f"Post-TO: \u03bc={post_mean:.3f} \u03c3={post_std:.3f} n={len(post):,}<br>"
                f"All: \u03bc={base_mean:.3f} \u03c3={base_std:.3f} n={len(base):,}",
                xref=f"x{col_idx}" if col_idx > 1 else "x",
                yref=f"y{col_idx}" if col_idx > 1 else "y",
                x=5,
                y=0.95,
                xanchor="right",
                yanchor="top",
                showarrow=False,
                font=dict(size=10),
                bgcolor="rgba(255,255,255,0.7)",
            )

        fig.update_layout(
            title_text=f"Points per Possession: Post-Timeout vs Baseline<br>" f"<sup>Filters: {filter_desc}</sup>",
            barmode="overlay",
            template="plotly_dark",
            width=400 * n_possessions,
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5),
        )

        return fig

    # ------------------------------------------------------------------ #
    #  Counterfactual timeout impact (Plotly)                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def plot_timeout_causal_effect(cls, memo: CDNNBAMemoPL) -> go.Figure:
        """Visualize the causal effect of timeouts vs matched controls.

        Top panel: forward N-possession net for timeout vs control groups,
        plotted against trailing N-possession net (the "situation" at timeout).
        N is even (default 6) so each team gets exactly N/2 possessions.
        Bottom panel: causal effect (timeout - control) per bucket with
        significance markers.

        Uses ``memo.timeout_counterfactual`` for the data.
        """
        cf = memo.timeout_counterfactual
        W = memo._COUNTERFACTUAL_WINDOW
        half = W // 2

        buckets = cf["trail_bucket"].to_numpy()
        to_mean = cf["to_fwd_mean"].to_numpy()
        ctrl_mean = cf["ctrl_fwd_mean"].to_numpy()
        to_std = cf["to_fwd_std"].to_numpy()
        ctrl_std = cf["ctrl_fwd_std"].to_numpy()
        to_n = cf["to_n"].to_numpy()
        ctrl_n = cf["ctrl_n"].to_numpy()
        effect = cf["causal_effect"].to_numpy()
        pvals = cf["p_value"].to_numpy()

        # Standard error for CI bands
        to_se = to_std / np.sqrt(to_n)
        ctrl_se = ctrl_std / np.sqrt(ctrl_n)

        # Weighted overall effect
        weighted_effect = float(np.sum(effect * to_n) / np.sum(to_n))

        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.6, 0.4],
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=[
                f"Forward {W}-Possession Net: Timeout vs Matched Control ({half} per team)",
                "Causal Effect (Timeout \u2212 Control)",
            ],
        )

        # --- Top panel: two lines with CI bands ---

        # Control CI band
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([buckets, buckets[::-1]]),
                y=np.concatenate([ctrl_mean + 1.96 * ctrl_se, (ctrl_mean - 1.96 * ctrl_se)[::-1]]),
                fill="toself",
                fillcolor="rgba(100,149,237,0.15)",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

        # Timeout CI band
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([buckets, buckets[::-1]]),
                y=np.concatenate([to_mean + 1.96 * to_se, (to_mean - 1.96 * to_se)[::-1]]),
                fill="toself",
                fillcolor="rgba(255,99,71,0.15)",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

        # Control line
        fig.add_trace(
            go.Scatter(
                x=buckets,
                y=ctrl_mean,
                mode="lines+markers",
                name="No timeout (control)",
                line=dict(color="cornflowerblue", width=2),
                marker=dict(size=5),
                hovertemplate="Trailing net: %{x}<br>Forward net: %{y:.3f}<br>"
                "n=%{customdata:,}<extra>Control</extra>",
                customdata=ctrl_n,
            ),
            row=1,
            col=1,
        )

        # Timeout line
        fig.add_trace(
            go.Scatter(
                x=buckets,
                y=to_mean,
                mode="lines+markers",
                name="After timeout",
                line=dict(color="tomato", width=2),
                marker=dict(size=5),
                hovertemplate="Trailing net: %{x}<br>Forward net: %{y:.3f}<br>"
                "n=%{customdata:,}<extra>Timeout</extra>",
                customdata=to_n,
            ),
            row=1,
            col=1,
        )

        # Zero line
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, row=1, col=1)  # type: ignore

        # --- Bottom panel: causal effect bars ---
        bar_colors = ["rgba(34,139,34,0.7)" if p < 0.05 else "rgba(150,150,150,0.5)" for p in pvals]
        sig_text = ["*" if p < 0.05 else "" for p in pvals]

        fig.add_trace(
            go.Bar(
                x=buckets,
                y=effect,
                marker_color=bar_colors,
                hovertemplate="Trailing net: %{x}<br>Causal effect: %{y:+.3f}<br>" "p=%{customdata:.4f}<extra></extra>",
                customdata=pvals,
                showlegend=False,
                text=sig_text,
                textposition="outside",
                textfont=dict(size=10),
            ),
            row=2,
            col=1,
        )

        # Weighted average line
        fig.add_hline(
            y=weighted_effect,
            line_dash="dash",
            line_color="tomato",
            annotation_text=f"Weighted avg: {weighted_effect:+.3f} pts/{W}poss",
            annotation_position="top right",
            row=2,  # type: ignore
            col=1,  # type: ignore
        )
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, row=2, col=1)  # type: ignore

        fig.update_xaxes(title_text=f"Trailing {W}-Possession Net (caller's perspective)", row=2, col=1)
        fig.update_yaxes(title_text=f"Forward {W}-Poss Net", row=1, col=1)
        fig.update_yaxes(title_text="Effect (pts)", row=2, col=1)

        fig.update_layout(
            title_text="Causal Effect of Coach Timeouts on Scoring<br>"
            f"<sup>Matched by trailing {W}-possession net ({half} per team) | 95% CI bands | * = p < 0.05</sup>",
            template="plotly_dark",
            width=900,
            height=700,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )

        return fig

    # ------------------------------------------------------------------ #
    #  Stoppage impact on runs (Plotly)                                    #
    # ------------------------------------------------------------------ #

    @classmethod
    def plot_stoppage_run_impact(
        cls,
        memo: CDNNBAMemoPL,
        run_size: int = 5,
        minutes: float = 3.0,
        location: t.Literal["home", "away"] | None = None,
        margin: t.Literal["ahead", "behind"] | None = None,
    ) -> go.Figure:
        """Compare recovery from a scoring run after endogenous timeout,
        exogenous stoppage, or no stoppage.

        Shows overlaid histograms of the lead change (from the suffering
        team's perspective) over the next *minutes*, plus a summary with
        means, stds, and a significance test.

        Parameters
        ----------
        memo : CDNNBAMemoPL
        run_size : int
            Minimum |streak| to qualify as a "run" (default 5).
        minutes : float
            Forward window in minutes to measure recovery (default 3).
        location : "home" | "away" | None
            Filter to runs where the suffering team is home or away.
            None (default) includes both.
        margin : "ahead" | "behind" | None
            Filter by the suffering team's score margin at the time of the event.
            "ahead" = suffering team is still winning despite the run.
            "behind" = suffering team is trailing.
            None (default) includes both.
        """
        filter_parts = []
        if location:
            filter_parts.append(f"suffering={location}")
        if margin:
            filter_parts.append(f"margin={margin}")
        filter_label = f", {', '.join(filter_parts)}" if filter_parts else ""
        print(f"Stoppage run impact: run>={run_size}, forward={minutes}min{filter_label}")
        data = memo.stoppage_run_impact(run_size=run_size, minutes=minutes)

        if location is not None:
            data = data.filter(pl.col("suffering_location") == location)
        if margin == "ahead":
            data = data.filter(pl.col("suffering_margin") > 0)
        elif margin == "behind":
            data = data.filter(pl.col("suffering_margin") < 0)

        endo = data.filter(pl.col("group") == "endogenous")["recovery"].to_numpy()
        exo = data.filter(pl.col("group") == "exogenous")["recovery"].to_numpy()
        ctrl = data.filter(pl.col("group") == "control")["recovery"].to_numpy()

        groups = [
            ("endogenous", endo, "rgba(255,99,71,0.6)", "Coach Timeout"),
            ("exogenous", exo, "rgba(100,149,237,0.6)", "TV/Official Stoppage"),
            ("control", ctrl, "rgba(150,150,150,0.4)", "No Stoppage"),
        ]
        groups = [(name, arr, color, label) for name, arr, color, label in groups if len(arr) > 0]

        fig = make_subplots(
            rows=1,
            cols=2,
            column_widths=[0.65, 0.35],
            horizontal_spacing=0.12,
            subplot_titles=[
                f"Recovery Distribution ({minutes}min after run >= {run_size})",
                "Mean Recovery (95% CI)",
            ],
        )

        # --- Left panel: overlaid histograms ---
        for name, arr, color, label in groups:
            mu, sigma = float(np.mean(arr)), float(np.std(arr))
            fig.add_trace(
                go.Histogram(
                    x=arr,
                    histnorm="probability",
                    name=f"{label} (n={len(arr):,}, \u03bc={mu:.3f}, \u03c3={sigma:.3f})",
                    marker_color=color,
                    opacity=0.7,
                    hovertemplate="Recovery: %{x:.1f}<br>Probability: %{y:.3f}<extra>%{fullData.name}</extra>",
                ),
                row=1,
                col=1,
            )

        fig.update_xaxes(title_text="Lead Change (suffering team's perspective)", row=1, col=1)
        fig.update_yaxes(title_text="Probability", row=1, col=1)

        # --- Right panel: mean + CI bar chart ---
        labels = []
        means = []
        ci_lo = []
        ci_hi = []
        colors = []
        for name, arr, color, label in groups:
            mu = float(np.mean(arr))
            se = float(np.std(arr) / np.sqrt(len(arr)))
            labels.append(label)
            means.append(mu)
            ci_lo.append(mu - 1.96 * se)
            ci_hi.append(mu + 1.96 * se)
            colors.append(color)

        fig.add_trace(
            go.Bar(
                x=labels,
                y=means,
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[h - m for h, m in zip(ci_hi, means)],
                    arrayminus=[m - l for m, l in zip(means, ci_lo)],
                ),
                marker_color=colors,
                hovertemplate="%{x}<br>Mean: %{y:.3f}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, row=1, col=2)  # type: ignore
        fig.update_yaxes(title_text="Mean Recovery (pts)", row=1, col=2)

        # --- Significance tests ---
        annotations = []
        test_pairs = [
            ("endogenous", "control", endo, ctrl),
            ("exogenous", "control", exo, ctrl),
            ("endogenous", "exogenous", endo, exo),
        ]
        for name_a, name_b, arr_a, arr_b in test_pairs:
            if len(arr_a) > 0 and len(arr_b) > 0:
                t_stat, p_val = sp_stats.ttest_ind(arr_a, arr_b, equal_var=False)
                diff = float(np.mean(arr_a) - np.mean(arr_b))
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."  # type: ignore
                annotations.append(f"{name_a} vs {name_b}: diff={diff:+.3f}, p={p_val:.4f} {sig}")

        annotation_text = "<br>".join(annotations)
        fig.add_annotation(
            text=annotation_text,
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.22,
            xanchor="center",
            yanchor="top",
            showarrow=False,
            font=dict(size=13, family="monospace"),
            bgcolor="rgba(0,0,0,0.6)",
            bordercolor="gray",
            borderwidth=1,
        )

        fig.update_layout(
            title_text=f"Scoring Run Recovery: Endogenous vs Exogenous vs No Stoppage<br>"
            f"<sup>Run >= {run_size} pts | {minutes}min forward | "
            f"Suffering team: {location or 'all'}"
            f"{f' | {margin}' if margin else ''}</sup>",
            barmode="overlay",
            template="plotly_dark",
            width=1400,
            height=750,
            font=dict(size=14),
            title_font=dict(size=18),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=1.12,
                xanchor="center",
                x=0.35,
                font=dict(size=13),
            ),
            margin=dict(b=180, t=140, l=80, r=60),
        )

        # Increase subplot title font
        for ann in fig.layout.annotations:  # type: ignore
            if ann.text and ("Recovery" in ann.text or "Mean" in ann.text):
                ann.font = dict(size=15)

        return fig

    # ------------------------------------------------------------------ #
    #  Stoppage run impact — 2x2 grid (Plotly)                             #
    # ------------------------------------------------------------------ #

    @classmethod
    def plot_stoppage_run_impact_grid(
        cls,
        memo: CDNNBAMemoPL,
        run_size: int = 5,
        minutes: float = 3.0,
        max_abs_margin: int | None = None,
        metric: t.Literal["recovery", "running_team_pts"] = "recovery",
    ) -> go.Figure:
        """2x2 grid of bar charts, split by home/away and ahead/behind.

        Each subplot shows Coach Timeout vs TV/Official Stoppage vs No Stoppage
        with 95% CI error bars.

        Parameters
        ----------
        memo : CDNNBAMemoPL
        run_size : int
            Minimum |streak| to qualify as a "run" (default 5).
        minutes : float
            Forward window in minutes to measure recovery (default 3).
        max_abs_margin : int or None
            If set, only include events where |suffering_margin| <= this value
            (i.e. close games only).
        """
        metric_labels = {
            "recovery": (
                "Mean Recovery (pts)",
                "Scoring Run Recovery by Situation",
                "Suffering team's net lead change",
            ),
            "running_team_pts": (
                "Running Team Pts Scored",
                "Running Team Scoring After Stoppage",
                "Points scored by the team on the run",
            ),
        }
        y_label, title_prefix, metric_desc = metric_labels[metric]

        print(
            f"Stoppage run impact grid: run>={run_size}, forward={minutes}min, metric={metric}"
            f"{f', |margin|<={max_abs_margin}' if max_abs_margin else ''}"
        )
        data = memo.stoppage_run_impact(run_size=run_size, minutes=minutes)

        if max_abs_margin is not None:
            data = data.filter(pl.col("suffering_margin").abs() <= max_abs_margin)

        panel_specs = [
            ("home", "ahead", "Home & Ahead"),
            ("away", "ahead", "Away & Ahead"),
            ("home", "behind", "Home & Behind"),
            ("away", "behind", "Away & Behind"),
        ]

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[s[2] for s in panel_specs],
            vertical_spacing=0.22,
            horizontal_spacing=0.14,
        )

        group_defs = [
            ("endogenous", "rgba(255,99,71,0.7)", "Coach TO"),
            ("exogenous", "rgba(100,149,237,0.7)", "TV/Official"),
            ("control", "rgba(150,150,150,0.5)", "No Stoppage"),
        ]

        sig_lines: list[str] = []
        all_y_lo: list[float] = []
        all_y_hi: list[float] = []

        for idx, (loc, mar, title) in enumerate(panel_specs):
            row = idx // 2 + 1
            col = idx % 2 + 1

            if mar == "ahead":
                panel_data = data.filter((pl.col("suffering_location") == loc) & (pl.col("suffering_margin") > 0))
            else:
                panel_data = data.filter((pl.col("suffering_location") == loc) & (pl.col("suffering_margin") < 0))

            labels = []
            means = []
            errors_plus = []
            errors_minus = []
            colors = []
            arrays: dict[str, np.ndarray] = {}

            for gname, gcolor, glabel in group_defs:
                arr = panel_data.filter(pl.col("group") == gname)[metric].to_numpy()
                arrays[gname] = arr
                if len(arr) == 0:
                    continue
                mu = float(np.mean(arr))
                se = float(np.std(arr) / np.sqrt(len(arr)))
                labels.append(f"{glabel}\nn={len(arr):,}")
                means.append(mu)
                errors_plus.append(1.96 * se)
                errors_minus.append(1.96 * se)
                colors.append(gcolor)

            # Track global y range
            for m, ep, em in zip(means, errors_plus, errors_minus):
                all_y_hi.append(m + ep)
                all_y_lo.append(m - em)

            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=means,
                    error_y=dict(type="data", symmetric=False, array=errors_plus, arrayminus=errors_minus),
                    marker_color=colors,
                    hovertemplate="%{x}<br>Mean: %{y:.3f}<extra></extra>",
                    showlegend=(idx == 0),
                    name="Recovery",
                ),
                row=row,
                col=col,
            )

            fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, row=row, col=col)  # type: ignore
            fig.update_yaxes(title_text=y_label if col == 1 else "", row=row, col=col)

            # Significance: endo vs ctrl
            endo_arr = arrays.get("endogenous", np.array([]))
            ctrl_arr = arrays.get("control", np.array([]))
            exo_arr = arrays.get("exogenous", np.array([]))
            if len(endo_arr) > 30 and len(ctrl_arr) > 30:
                _, p_ec = sp_stats.ttest_ind(endo_arr, ctrl_arr, equal_var=False)
                diff_ec = float(np.mean(endo_arr) - np.mean(ctrl_arr))
                sig_ec = "***" if p_ec < 0.001 else "**" if p_ec < 0.01 else "*" if p_ec < 0.05 else "n.s."  # type: ignore
                sig_lines.append(
                    f"{title}: TO vs Ctrl {diff_ec:+.3f} (p={p_ec:.3f}{sig_ec})"
                    f"  |  TV vs Ctrl {float(np.mean(exo_arr) - np.mean(ctrl_arr)):+.3f}"
                )

        # Set consistent y-axis range across all panels
        if all_y_lo and all_y_hi:
            y_pad = (max(all_y_hi) - min(all_y_lo)) * 0.15
            y_range = [min(all_y_lo) - y_pad, max(all_y_hi) + y_pad]
            for r in range(1, 3):
                for c in range(1, 3):
                    fig.update_yaxes(range=y_range, row=r, col=c)

        margin_label = f" | |margin| <= {max_abs_margin}" if max_abs_margin else ""
        fig.update_layout(
            title_text=f"{title_prefix}<br>"
            f"<sup>{metric_desc} | Run >= {run_size} pts | {minutes}min forward{margin_label} | 95% CI</sup>",
            template="plotly_dark",
            width=1200,
            height=900,
            font=dict(size=14),
            title_font=dict(size=18),
            showlegend=False,
            margin=dict(b=160, t=120, l=80, r=60),
        )

        # Significance summary at bottom
        fig.add_annotation(
            text="<br>".join(sig_lines),
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.15,
            xanchor="center",
            yanchor="top",
            showarrow=False,
            font=dict(size=12, family="monospace"),
            bgcolor="rgba(0,0,0,0.6)",
            bordercolor="gray",
            borderwidth=1,
        )

        # Increase subplot title font
        for ann in fig.layout.annotations:  # type: ignore
            for keyword in panel_specs:
                if ann.text == keyword[2]:
                    ann.font = dict(size=15)

        return fig
