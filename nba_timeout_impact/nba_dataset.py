import pandas as pd

from kret_np_pd.enriched_df import Enriched_DF


class NBADataset(Enriched_DF):
    actionNumber: pd.Series
    teamId: pd.Categorical
