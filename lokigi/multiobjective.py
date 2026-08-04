from dataclasses import dataclass
from typing import Callable, Literal
import numpy as np
import pandas as pd

Direction = Literal["higher_better", "lower_better", "closest_to_target"]


@dataclass
class Metric:
    """
    Describes how one column in the solutions dataframe should be treated
    when comparing solutions.

    Used in two places: `compute_pareto_front()` and friends, where a list
    of these describes a multi-objective trade-off; and `solve(rank_on=...)`,
    where a single one names the metric the search itself should optimise.

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
    phrasing: "Callable[[Metric, float, bool], str] | None" = None

    def __post_init__(self):
        if self.direction == "closest_to_target" and self.target is None:
            raise ValueError(
                f"Metric '{self.column}' uses closest_to_target but no target was given."
            )
        if self.label is None:
            self.label = self.column.replace("_", " ").capitalize()

    def normalise(self, values):
        """
        Return a version of this metric where lower is always better.

        Accepts either a `pandas.Series` (the batch comparisons: Pareto
        dominance, cost weighting, the final solution sort) or a single
        float (`solve()`'s brute-force streaming heap, which scores one
        combination at a time and never materialises a batch). `np.abs`
        rather than `Series.abs()` is what makes the scalar case work;
        both return the same thing for a Series.
        """
        if self.direction == "higher_better":
            return -values
        if self.direction == "closest_to_target":
            return np.abs(values - self.target)
        return values

    def format_value(self, value: float) -> str:
        """
        Render a single value of this metric for stakeholder-facing text.

        The level counterpart to `format_delta` below, which renders a
        *change*: "23.4 minutes" rather than "0.2 minutes worse". Used when
        labelling what a solution was ranked on.
        """
        if pd.isna(value):
            return "n/a"
        v = value * 100 if self.as_percentage else value
        unit = "%" if self.as_percentage else (f" {self.unit}" if self.unit else "")
        return f"{v:.{self.decimals}f}{unit}"

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
