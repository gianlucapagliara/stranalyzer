"""Tests for the DataValidator class."""

import numpy as np
import pandas as pd

from stranalyzer.data.validator import DataValidator


class TestDataValidator:
    """Test cases for DataValidator class."""

    def test_validate_series_empty(self, empty_series):
        """Test validation of empty series."""
        result = DataValidator.validate_series(empty_series)

        assert result["is_valid"] is False
        assert "Series is empty" in result["errors"]
        assert len(result["warnings"]) == 0

    def test_validate_series_valid(self, sample_single_strategy):
        """Test validation of valid series."""
        result = DataValidator.validate_series(sample_single_strategy)

        assert result["is_valid"] is True
        assert len(result["errors"]) == 0

        # Check statistics are calculated
        stats = result["statistics"]
        assert "count" in stats
        assert "mean" in stats
        assert "std" in stats
        assert stats["count"] == len(sample_single_strategy)

    def test_validate_series_insufficient_data(self):
        """Test validation with insufficient data points."""
        # Create series with less than 30 data points
        short_series = pd.Series([0.01, 0.02, -0.01, 0.005] * 5)  # 20 points

        result = DataValidator.validate_series(short_series)

        assert result["is_valid"] is True  # Still valid, just warning
        assert any("only 20 data points" in warning for warning in result["warnings"])

    def test_validate_series_with_missing_values(self, series_with_nans):
        """Test validation of series with missing values."""
        result = DataValidator.validate_series(series_with_nans)

        assert result["is_valid"] is True  # Valid but with warnings
        assert any("missing values" in warning for warning in result["warnings"])

        # Check that missing count is correct
        expected_missing = series_with_nans.isna().sum()
        assert result["statistics"]["missing"] == expected_missing

    def test_validate_series_constant_values(self):
        """Test validation of series with constant values."""
        constant_series = pd.Series(
            [0.01] * 100, index=pd.date_range("2023-01-01", periods=100)
        )

        result = DataValidator.validate_series(constant_series)

        assert result["is_valid"] is True
        assert any("constant values" in warning for warning in result["warnings"])

    def test_validate_series_with_outliers(self):
        """Test validation of series with outliers."""
        np.random.seed(42)
        normal_data = np.random.normal(0.001, 0.01, 100)

        # Add some extreme outliers
        outlier_data = normal_data.copy()
        outlier_data[50] = 0.5  # Extreme positive outlier
        outlier_data[75] = -0.5  # Extreme negative outlier

        outlier_series = pd.Series(
            outlier_data, index=pd.date_range("2023-01-01", periods=100)
        )

        result = DataValidator.validate_series(outlier_series)

        assert result["is_valid"] is True
        assert any("outliers" in warning for warning in result["warnings"])

    def test_validate_series_statistics_calculation(self, sample_single_strategy):
        """Test that statistics are correctly calculated."""
        result = DataValidator.validate_series(sample_single_strategy)

        stats = result["statistics"]
        clean_data = sample_single_strategy.dropna()

        # Verify statistics match pandas calculations
        assert abs(stats["mean"] - float(clean_data.mean())) < 1e-10
        assert abs(stats["std"] - float(clean_data.std())) < 1e-10
        assert abs(stats["min"] - float(clean_data.min())) < 1e-10
        assert abs(stats["max"] - float(clean_data.max())) < 1e-10
        assert stats["count"] == len(sample_single_strategy)
        assert stats["missing"] == sample_single_strategy.isna().sum()

    def test_validate_dataframe_empty(self):
        """Test validation of empty DataFrame."""
        empty_df = pd.DataFrame()

        result = DataValidator.validate_dataframe(empty_df)

        assert result["is_valid"] is False
        assert "DataFrame is empty" in result["errors"]

    def test_validate_dataframe_valid(self, sample_returns_data):
        """Test validation of valid DataFrame."""
        df = pd.DataFrame(sample_returns_data)

        result = DataValidator.validate_dataframe(df)

        assert result["is_valid"] is True
        assert len(result["errors"]) == 0
        assert len(result["column_validations"]) == len(df.columns)

        # Check that each column validation is included
        for col in df.columns:
            assert col in result["column_validations"]
            assert result["column_validations"][col]["is_valid"] is True

    def test_validate_dataframe_invalid_column(self):
        """Test validation of DataFrame with invalid column."""
        # Create DataFrame with one invalid (empty) column
        df = pd.DataFrame(
            {
                "valid_col": [0.01, 0.02, -0.01],
                "invalid_col": [np.nan, np.nan, np.nan],  # All NaN
            },
            index=pd.date_range("2023-01-01", periods=3),
        )

        result = DataValidator.validate_dataframe(df)

        # Should still be valid overall, but with warnings for the invalid column
        assert "invalid_col" in result["column_validations"]

    def test_validate_dataframe_non_datetime_index(self, sample_returns_data):
        """Test validation of DataFrame with non-datetime index."""
        df = pd.DataFrame(sample_returns_data)
        df = df.reset_index(drop=True)  # Convert to integer index

        result = DataValidator.validate_dataframe(df)

        assert any("not a DatetimeIndex" in warning for warning in result["warnings"])

    def test_validate_dataframe_duplicate_index(self, sample_returns_data):
        """Test validation of DataFrame with duplicate index values."""
        df = pd.DataFrame(sample_returns_data)

        # Create duplicate index by reindexing
        duplicate_index = df.index.tolist()
        duplicate_index[10] = duplicate_index[5]  # Create duplicate
        df = df.reindex(pd.DatetimeIndex(duplicate_index))

        result = DataValidator.validate_dataframe(df)

        assert any("duplicate values" in warning for warning in result["warnings"])

    def test_check_data_alignment_empty(self):
        """Test data alignment check with empty data."""
        result = DataValidator.check_data_alignment({})

        assert result["is_aligned"] is True
        assert len(result["date_ranges"]) == 0
        assert result["common_period"] is None

    def test_check_data_alignment_single_series(self, sample_single_strategy):
        """Test data alignment check with single series."""
        data_dict = {"strategy1": sample_single_strategy}

        result = DataValidator.check_data_alignment(data_dict)

        assert result["is_aligned"] is True
        assert len(result["date_ranges"]) == 1
        assert "strategy1" in result["date_ranges"]

    def test_check_data_alignment_overlapping(self, sample_returns_data):
        """Test data alignment check with overlapping series."""
        result = DataValidator.check_data_alignment(sample_returns_data)

        assert result["is_aligned"] is True
        assert len(result["date_ranges"]) == len(sample_returns_data)
        assert result["common_period"] is not None

        # All series should have the same date range in this case
        start_dates = [info["start"] for info in result["date_ranges"].values()]
        end_dates = [info["end"] for info in result["date_ranges"].values()]

        assert len(set(start_dates)) == 1  # All start dates are the same
        assert len(set(end_dates)) == 1  # All end dates are the same

    def test_check_data_alignment_non_overlapping(self):
        """Test data alignment check with non-overlapping series."""
        # Create two series with different date ranges
        series1 = pd.Series([0.01, 0.02], index=pd.date_range("2023-01-01", periods=2))
        series2 = pd.Series([0.03, 0.04], index=pd.date_range("2023-06-01", periods=2))

        data_dict = {"series1": series1, "series2": series2}

        result = DataValidator.check_data_alignment(data_dict)

        assert result["is_aligned"] is False
        assert any("No overlapping period" in warning for warning in result["warnings"])

    def test_suggest_data_improvements_no_issues(self, sample_single_strategy):
        """Test suggestions when no issues are found."""
        validation_result = DataValidator.validate_series(sample_single_strategy)
        suggestions = DataValidator.suggest_data_improvements(validation_result)

        # Should have no suggestions for valid data
        assert isinstance(suggestions, list)

    def test_suggest_data_improvements_with_issues(self, series_with_nans):
        """Test suggestions when issues are found."""
        validation_result = DataValidator.validate_series(series_with_nans)
        suggestions = DataValidator.suggest_data_improvements(validation_result)

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert any("missing values" in suggestion for suggestion in suggestions)

    def test_suggest_data_improvements_invalid_data(self, empty_series):
        """Test suggestions for invalid data."""
        validation_result = DataValidator.validate_series(empty_series)
        suggestions = DataValidator.suggest_data_improvements(validation_result)

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert any("Fix data errors" in suggestion for suggestion in suggestions)
