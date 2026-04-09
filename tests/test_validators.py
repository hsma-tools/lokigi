import pytest
import pandas as pd
from lokigi.utils import _validate_columns

# AI use disclosure
# Gemini 3.1 pro used for drafting tests
# All tests have been checked by a human





# --- Tests ---


def test_successful_validation(manager, valid_df):
    """Test the happy path where everything is correct."""
    # Should run without raising any exceptions
    manager.add_data(valid_df, id_col="site_id")
    assert manager.success is True


def test_missing_column_raises_value_error(manager, valid_df):
    """Test that missing columns trigger our formatted ValueError."""
    # Drop the 'lat' column to trigger the error
    bad_df = valid_df.drop(columns=["lat"])

    with pytest.raises(ValueError) as exc_info:
        manager.add_data(bad_df, id_col="site_id")

    # Check that our custom message formatted correctly
    assert "Missing columns: ['lat']" in str(exc_info.value)


def test_string_in_numeric_column_raises_type_error(manager, invalid_df):
    """Test that passing text to a numeric column triggers a TypeError."""
    with pytest.raises(TypeError) as exc_info:
        manager.add_data(invalid_df, id_col="site_id")

    assert "must contain numbers" in str(exc_info.value)
    assert "['long']" in str(exc_info.value)
