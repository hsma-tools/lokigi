import pandas as pd
import pytest

from lokigi.multiobjective import ParetoMetric

# All tests in this file written by

# --- __post_init__ ---


def test_closest_to_target_requires_target():
    """closest_to_target without a target should raise immediately."""
    with pytest.raises(ValueError) as exc_info:
        ParetoMetric(column="equity_ratio", direction="closest_to_target")

    assert "equity_ratio" in str(exc_info.value)
    assert "closest_to_target" in str(exc_info.value)


def test_closest_to_target_with_target_is_valid():
    """Providing a target alongside closest_to_target should not raise."""
    metric = ParetoMetric(
        column="equity_ratio", direction="closest_to_target", target=1.0
    )
    assert metric.target == 1.0


def test_label_defaults_to_title_cased_column():
    """No label given -> underscores replaced and first letter capitalised."""
    metric = ParetoMetric(column="average_travel_time")
    assert metric.label == "Average travel time"


def test_label_is_not_overridden_when_given():
    metric = ParetoMetric(column="average_travel_time", label="Avg. journey time")
    assert metric.label == "Avg. journey time"


def test_default_direction_is_lower_better():
    metric = ParetoMetric(column="cost")
    assert metric.direction == "lower_better"


# --- normalise ---


def test_normalise_higher_better_negates_series():
    metric = ParetoMetric(column="coverage", direction="higher_better")
    series = pd.Series([10.0, 20.0, 30.0])

    result = metric.normalise(series)

    pd.testing.assert_series_equal(result, -series)


def test_normalise_lower_better_returns_series_unchanged():
    metric = ParetoMetric(column="travel_time", direction="lower_better")
    series = pd.Series([10.0, 20.0, 30.0])

    result = metric.normalise(series)

    pd.testing.assert_series_equal(result, series)


def test_normalise_closest_to_target_returns_absolute_deviation():
    metric = ParetoMetric(
        column="equity_ratio", direction="closest_to_target", target=1.0
    )
    series = pd.Series([0.7, 1.0, 1.4])

    result = metric.normalise(series)

    pd.testing.assert_series_equal(result, pd.Series([0.3, 0.0, 0.4]))


def test_normalise_result_always_makes_lower_better():
    """Whatever the direction, the best value in `series` should end up smallest after normalising."""
    higher = ParetoMetric(column="coverage", direction="higher_better")
    series = pd.Series([10.0, 50.0, 30.0])  # 50 is the best (highest) value

    normed = higher.normalise(series)

    assert normed.idxmin() == series.idxmax()


# --- format_delta ---


def test_format_delta_plain_number():
    metric = ParetoMetric(column="travel_time", decimals=1)
    assert metric.format_delta(2.34) == "2.3"


def test_format_delta_takes_absolute_value():
    metric = ParetoMetric(column="travel_time", decimals=1)
    assert metric.format_delta(-2.34) == "2.3"


def test_format_delta_with_unit():
    metric = ParetoMetric(column="travel_time", unit="minutes", decimals=1)
    assert metric.format_delta(2.0) == "2.0 minutes"


def test_format_delta_as_percentage_scales_and_labels():
    metric = ParetoMetric(column="coverage", as_percentage=True, decimals=1)
    assert metric.format_delta(0.042) == "4.2 percentage points"


def test_format_delta_as_percentage_ignores_unit():
    """as_percentage should take priority over a configured `unit`."""
    metric = ParetoMetric(column="coverage", as_percentage=True, unit="km", decimals=0)
    assert metric.format_delta(0.05) == "5 percentage points"


def test_format_delta_respects_decimals():
    metric = ParetoMetric(column="travel_time", decimals=3)
    assert metric.format_delta(1.23456) == "1.235"


# --- phrase ---


def test_phrase_default_wording_when_better():
    metric = ParetoMetric(
        column="travel_time", label="Average journey time", unit="minutes"
    )
    assert (
        metric.phrase(0.2, better=True)
        == "Average journey time improves by 0.2 minutes"
    )


def test_phrase_default_wording_when_worse():
    metric = ParetoMetric(
        column="travel_time", label="Average journey time", unit="minutes"
    )
    assert (
        metric.phrase(0.2, better=False) == "Average journey time is 0.2 minutes worse"
    )


def test_phrase_uses_custom_phrasing_callable_when_given():
    def custom_phrasing(metric, delta, better):
        verb = "gains" if better else "loses"
        return f"{metric.label} {verb} {abs(delta):.0f} people"

    metric = ParetoMetric(
        column="people_covered", label="Coverage", phrasing=custom_phrasing
    )

    assert metric.phrase(150, better=True) == "Coverage gains 150 people"
    assert metric.phrase(-150, better=False) == "Coverage loses 150 people"
