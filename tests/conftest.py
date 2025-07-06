"""Pytest configuration and shared fixtures for the stranalyzer test suite."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_returns_data() -> dict[str, pd.Series]:
    """Generate sample returns data for testing."""
    np.random.seed(42)  # For reproducible tests

    # Generate date range
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    # Generate different strategy returns
    strategies = {}

    # Conservative strategy: low volatility, steady growth
    conservative_returns = np.random.normal(0.0005, 0.01, len(date_range))
    strategies["Conservative"] = pd.Series(conservative_returns, index=date_range)

    # Aggressive strategy: high volatility, higher growth potential
    aggressive_returns = np.random.normal(0.001, 0.03, len(date_range))
    strategies["Aggressive"] = pd.Series(aggressive_returns, index=date_range)

    # Market neutral: very low volatility
    neutral_returns = np.random.normal(0.0002, 0.005, len(date_range))
    strategies["Market_Neutral"] = pd.Series(neutral_returns, index=date_range)

    return strategies


@pytest.fixture
def sample_single_strategy() -> pd.Series:
    """Generate a single strategy for testing."""
    np.random.seed(42)
    date_range = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    returns = np.random.normal(0.001, 0.02, len(date_range))
    return pd.Series(returns, index=date_range)


@pytest.fixture
def empty_series() -> pd.Series:
    """Generate an empty series for testing edge cases."""
    return pd.Series(dtype=float)


@pytest.fixture
def series_with_nans() -> pd.Series:
    """Generate a series with NaN values for testing."""
    np.random.seed(42)
    date_range = pd.date_range(start="2023-01-01", end="2023-03-31", freq="D")
    returns = np.random.normal(0.001, 0.02, len(date_range))

    # Add some NaN values
    returns[10:15] = np.nan
    returns[50:55] = np.nan

    return pd.Series(returns, index=date_range)


@pytest.fixture
def sample_weights() -> dict[str, float]:
    """Generate sample weights for portfolio composition."""
    return {"Conservative": 0.4, "Aggressive": 0.4, "Market_Neutral": 0.2}


@pytest.fixture
def invalid_weights() -> dict[str, float]:
    """Generate invalid weights for testing error handling."""
    return {
        "Conservative": -0.2,  # Negative weight
        "Aggressive": 0.8,
        "Market_Neutral": 0.4,  # Total > 1.0
    }


@pytest.fixture
def sample_csv_content() -> str:
    """Generate sample CSV content for testing file upload."""
    return """date,strategy_1,strategy_2
2023-01-01,0.012,0.008
2023-01-02,-0.005,0.002
2023-01-03,0.008,0.012
2023-01-04,0.015,-0.001
2023-01-05,-0.002,0.007"""


@pytest.fixture
def invalid_csv_content() -> str:
    """Generate invalid CSV content for testing error handling."""
    return """date
2023-01-01
2023-01-02
2023-01-03"""  # Missing data columns


@pytest.fixture
def mock_uploaded_file():
    """Mock uploaded file for testing."""

    class MockUploadedFile:
        def __init__(self, content: str, name: str = "test.csv"):
            self.content = content
            self.name = name

        def read(self):
            return self.content.encode("utf-8")

        def seek(self, position):
            pass

    return MockUploadedFile


# Test data validation helpers
def assert_series_equal_approx(
    series1: pd.Series, series2: pd.Series, tolerance: float = 1e-6
):
    """Assert that two series are approximately equal."""
    assert len(series1) == len(series2), "Series have different lengths"
    assert (series1.index == series2.index).all(), "Series have different indices"
    np.testing.assert_allclose(series1.values, series2.values, atol=tolerance)


def assert_metrics_valid(metrics: dict[str, float]):
    """Assert that metrics dictionary contains valid values."""
    for key, value in metrics.items():
        assert isinstance(value, (int, float)), f"Metric {key} is not numeric: {value}"
        assert not np.isnan(value), f"Metric {key} is NaN"
        assert np.isfinite(value), f"Metric {key} is not finite: {value}"
