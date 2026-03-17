import typing as t

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kret_matplotlib.UTILS_Matplotlib import UTILS_Plotting as uks_mpl
from kret_np_pd.UTILS_np_pd import NP_PD_Utils as UKS_NP_PD


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
