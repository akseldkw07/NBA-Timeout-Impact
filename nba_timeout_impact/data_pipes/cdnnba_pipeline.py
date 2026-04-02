import time

from nba_api.stats.endpoints import LeagueGameLog
import tarfile
import typing as t
from pathlib import Path

import pandas as pd

from nba_timeout_impact.data_pipes.load_data_utils import NBADataLoader, DATA_SRC_TYPE, SeasonsArg


class CDNBAPipelineHelper:
    @classmethod
    def load_and_stack_all(cls): ...

    @classmethod
    def clean_stacked(cls): ...
