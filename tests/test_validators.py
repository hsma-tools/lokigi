import pytest
import pandas as pd
from lokigi.utils import _validate_columns

# AI use disclosure
# Gemini 3.1 pro used for drafting tests
# All tests have been checked by a human


class DummyManager:
    """A minimal class to test the decorator."""

    @_validate_columns(
        df_arg_name="candidate_df",
        col_arg_names=["id_col", "lat_col", "long_col"],
        numeric_col_args=[
            "lat_col",
            "long_col",
        ],  # We only care that coords are numeric
        msg_template="Missing: {missing}. Found: {available}",
    )
    def add_data(self, candidate_df, id_col, lat_col="lat", long_col="long"):
        # If it reaches here, validation passed!
        self.success = True
        return True


# --- Fixtures ---
# Fixtures provide reusable test data


@pytest.fixture
def manager():
    return DummyManager()


@pytest.fixture
def valid_df():
    return pd.DataFrame(
        {
            "site_id": ["Site A", "Site B"],
            "lat": [51.5074, 53.4808],
            "long": [-0.1278, -2.2426],
        }
    )


@pytest.fixture
def invalid_df():
    return pd.DataFrame(
        {
            "site_id": ["Site A", "Site B"],
            "lat": [51.5074, 53.4808],
            "long": ["-0.1278", -2.2426],
        }
    )


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
    assert "Missing: ['lat']" in str(exc_info.value)
    assert "Found: ['site_id', 'long']" in str(exc_info.value)


def test_string_in_numeric_column_raises_type_error(manager, invalid_df):
    """Test that passing text to a numeric column triggers a TypeError."""
    with pytest.raises(TypeError) as exc_info:
        manager.add_data(invalid_df, id_col="site_id")

    assert "must contain numbers" in str(exc_info.value)
    assert "['long']" in str(exc_info.value)
