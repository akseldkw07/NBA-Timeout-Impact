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
