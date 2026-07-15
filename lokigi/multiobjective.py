from dataclasses import dataclass
from typing import Callable, Literal
import pandas as pd

Direction = Literal["higher_better", "lower_better", "closest_to_target"]


@dataclass
class ParetoMetric:
    """
    Describes how one column in the solutions dataframe should be treated
    when comparing solutions.

    column:  name of the column in the solutions dataframe
    direction: "higher_better" (e.g. coverage), "lower_better"
        (e.g. average travel time), or "closest_to_target" (e.g. an
        inter-tertile ratio where 1.0 = perfectly even, and both >1 and <1
        represent moving away from equity in different directions)
    target: required if direction == "closest_to_target"
    label: human-readable name, defaults to a title-cased column name
    """

    column: str
    direction: Direction = "lower_better"
    target: float | None = None
    label: str | None = None
    unit: str = ""
    as_percentage: bool = False
    decimals: int = 1
    phrasing: "Callable[[ParetoMetric, float, bool], str] | None" = None

    def __post_init__(self):
        if self.direction == "closest_to_target" and self.target is None:
            raise ValueError(
                f"ParetoMetric '{self.column}' uses closest_to_target but no target was given."
            )
        if self.label is None:
            self.label = self.column.replace("_", " ").capitalize()

    def normalise(self, series: pd.Series) -> pd.Series:
        """Return a version of this metric where lower is always better."""
        if self.direction == "higher_better":
            return -series
        if self.direction == "closest_to_target":
            return (series - self.target).abs()
        return series

    def format_delta(self, delta: float) -> str:
        """Render an absolute change in this metric for stakeholder-facing text."""
        v = abs(delta) * 100 if self.as_percentage else abs(delta)
        unit = (
            " percentage points"
            if self.as_percentage
            else (f" {self.unit}" if self.unit else "")
        )
        return f"{v:.{self.decimals}f}{unit}"

    def phrase(self, delta: float, better: bool) -> str:
        """
        Full clause describing a change in this metric, e.g.
        "average journey time is 0.2 minutes worse".

        Override by passing `phrasing` -- a callable taking
        (metric, delta, better) and returning a full clause -- for metrics
        where the generic "improves by / is worse by" wording reads
        awkwardly (e.g. counts of people, rather than continuous measures).
        """
        if self.phrasing is not None:
            return self.phrasing(self, delta, better)
        if better:
            return f"{self.label} improves by {self.format_delta(delta)}"
        return f"{self.label} is {self.format_delta(delta)} worse"
