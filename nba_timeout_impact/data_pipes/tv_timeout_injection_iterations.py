from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

# -------------------- source-specific column conventions ----------------- #


SOURCE_CONFIGS: dict[str, dict] = {
    "v3": {
        # nbastatsv3 schema (1998+; mandatories explicit only through 2016)
        "timeout_action": "Timeout",
        "coach_subtypes": ["Regular", "Short", "Coach Challenge"],
        "challenge_subtypes": ["Coach Challenge"],
        "order_col": "actionNumber",
    },
    "cdnnba": {
        # cdnnba schema (2020+; mandatories implicit, charged to team)
        "timeout_action": "timeout",
        "coach_subtypes": ["full", "challenge"],
        "challenge_subtypes": ["challenge"],
        "order_col": "orderNumber",
    },
}


# -------------------- era-specific rulebook windows ---------------------- #


# Trigger marks in seconds-remaining. Pre-2017: Q2/Q4 only, three triggers
# at 8:59 / 5:59 / 2:59. Post-2017: Q1-Q4, two triggers at 6:59 / 2:59.
PRE_2017_TRIGGERS = [540, 360, 180]
PRE_2017_PERIODS = [2, 4]
POST_2017_TRIGGERS = [420, 180]
POST_2017_PERIODS = [1, 2, 3, 4]


# -------------------- public result type --------------------------------- #


@dataclass
class ValidationResult:
    label: str
    seasons: tuple[int, int]
    tolerance_s: int
    n_gt: int
    n_pred: int
    tp: int
    fp: int
    fn: int
    per_season: pd.DataFrame = field(repr=False)
    per_period: pd.DataFrame = field(repr=False)

    @property
    def precision(self) -> float:
        return self.tp / max(self.tp + self.fp, 1)

    @property
    def recall(self) -> float:
        return self.tp / max(self.tp + self.fn, 1)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / max(p + r, 1e-9)

    def summary(self) -> str:
        return (
            f"{self.label}: "
            f"n_gt={self.n_gt:,} n_pred={self.n_pred:,} "
            f"TP={self.tp:,} FP={self.fp:,} FN={self.fn:,} | "
            f"P={self.precision:.3f} R={self.recall:.3f} F1={self.f1:.3f}"
        )

    def __repr__(self) -> str:
        return f"<ValidationResult {self.summary()}>"


class TVTimeoutValidation:
    """Static-method container for mandatory-timeout reclassification + validation."""

    # ---------- source dispatch ----------

    @staticmethod
    def get_source_config(source: Literal["v3", "cdnnba"]) -> dict:
        if source not in SOURCE_CONFIGS:
            raise ValueError(f"unknown source {source!r}, expected one of {list(SOURCE_CONFIGS)}")
        return SOURCE_CONFIGS[source]

    # ---------- classification ----------
