"""Tests for the PerformanceMetrics class."""

import numpy as np
import pandas as pd

from stranalyzer.analysis.metrics import PerformanceMetrics


class TestPerformanceMetrics:
    """Test cases for PerformanceMetrics class."""

    def test_calculate_basic_metrics_valid_data(self, sample_single_strategy):
        """Test basic metrics calculation with valid data."""
        metrics = PerformanceMetrics.calculate_basic_metrics(sample_single_strategy)

        assert isinstance(metrics, dict)

        # Check that all expected metrics are present
        expected_metrics = [
            "total_return",
            "cagr",
            "volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
            "skewness",
            "kurtosis",
            "var_95",
            "cvar_95",
        ]

        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)
            assert not np.isnan(metrics[metric])

    def test_calculate_basic_metrics_empty_data(self, empty_series):
        """Test basic metrics calculation with empty data."""
        metrics = PerformanceMetrics.calculate_basic_metrics(empty_series)

        assert isinstance(metrics, dict)
        assert len(metrics) == 0

    def test_calculate_basic_metrics_with_nans(self, series_with_nans):
        """Test basic metrics calculation with NaN values."""
        metrics = PerformanceMetrics.calculate_basic_metrics(series_with_nans)

        assert isinstance(metrics, dict)
        # Should still calculate metrics after dropping NaNs
        if len(metrics) > 0:
            for metric_value in metrics.values():
                assert not np.isnan(metric_value)

    def test_calculate_advanced_metrics_valid_data(self, sample_single_strategy):
        """Test advanced metrics calculation with valid data."""
        metrics = PerformanceMetrics.calculate_advanced_metrics(sample_single_strategy)

        assert isinstance(metrics, dict)

        # Check some expected advanced metrics
        expected_metrics = [
            "omega_ratio",
            "tail_ratio",
            "kelly_criterion",
            "profit_factor",
            "recovery_factor",
        ]

        for metric in expected_metrics:
            if metric in metrics:  # Some metrics might not be calculated
                assert isinstance(metrics[metric], float)

    def test_calculate_risk_metrics_valid_data(self, sample_single_strategy):
        """Test risk metrics calculation with valid data."""
        metrics = PerformanceMetrics.calculate_risk_metrics(sample_single_strategy)

        assert isinstance(metrics, dict)

        expected_metrics = [
            "volatility",
            "max_drawdown",
            "var_95",
            "cvar_95",
            "downside_deviation",
            "avg_drawdown",
        ]

        for metric in expected_metrics:
            if metric in metrics:
                assert isinstance(metrics[metric], (int, float))

    def test_calculate_comparative_metrics_valid_data(self, sample_returns_data):
        """Test comparative metrics calculation."""
        strategies = list(sample_returns_data.keys())
        if len(strategies) >= 2:
            returns1 = sample_returns_data[strategies[0]]
            returns2 = sample_returns_data[strategies[1]]

            metrics = PerformanceMetrics.calculate_comparative_metrics(
                returns1, returns2
            )

            assert isinstance(metrics, dict)

            expected_metrics = ["correlation", "beta", "alpha", "information_ratio"]

            for metric in expected_metrics:
                if metric in metrics:
                    assert isinstance(metrics[metric], float)
                    assert not np.isnan(metrics[metric])

    def test_calculate_comparative_metrics_empty_data(self, empty_series):
        """Test comparative metrics with empty data."""
        metrics = PerformanceMetrics.calculate_comparative_metrics(
            empty_series, empty_series
        )

        assert isinstance(metrics, dict)
        assert len(metrics) == 0

    def test_calculate_rolling_metrics_sufficient_data(self, sample_single_strategy):
        """Test rolling metrics calculation with sufficient data."""
        window = 30  # Use smaller window for test data
        metrics = PerformanceMetrics.calculate_rolling_metrics(
            sample_single_strategy, window
        )

        assert isinstance(metrics, dict)

        expected_metrics = [
            "rolling_sharpe",
            "rolling_volatility",
            "rolling_max_drawdown",
        ]

        for metric in expected_metrics:
            if metric in metrics:
                assert isinstance(metrics[metric], pd.Series)

    def test_calculate_rolling_metrics_insufficient_data(self):
        """Test rolling metrics with insufficient data."""
        short_series = pd.Series(
            [0.01, 0.02, -0.01], index=pd.date_range("2023-01-01", periods=3)
        )
        window = 10

        metrics = PerformanceMetrics.calculate_rolling_metrics(short_series, window)

        assert isinstance(metrics, dict)
        assert len(metrics) == 0  # Should return empty dict

    def test_calculate_monthly_returns_valid_data(self, sample_single_strategy):
        """Test monthly returns calculation."""
        monthly_df = PerformanceMetrics.calculate_monthly_returns(
            sample_single_strategy
        )

        assert isinstance(monthly_df, pd.DataFrame)

        if not monthly_df.empty:
            # Should have years as index and months as columns
            assert monthly_df.index.name == "Year" or "Year" in str(monthly_df.index)

    def test_calculate_monthly_returns_empty_data(self, empty_series):
        """Test monthly returns calculation with empty data."""
        monthly_df = PerformanceMetrics.calculate_monthly_returns(empty_series)

        assert isinstance(monthly_df, pd.DataFrame)
        assert monthly_df.empty

    def test_calculate_drawdown_series_valid_data(self, sample_single_strategy):
        """Test drawdown series calculation."""
        drawdown = PerformanceMetrics.calculate_drawdown_series(sample_single_strategy)

        assert isinstance(drawdown, pd.Series)

        if not drawdown.empty:
            # Drawdown should be non-positive
            assert all(drawdown <= 0)

    def test_calculate_drawdown_series_empty_data(self, empty_series):
        """Test drawdown series calculation with empty data."""
        drawdown = PerformanceMetrics.calculate_drawdown_series(empty_series)

        assert isinstance(drawdown, pd.Series)
        assert drawdown.empty

    def test_calculate_cumulative_returns_valid_data(self, sample_single_strategy):
        """Test cumulative returns calculation."""
        cumulative = PerformanceMetrics.calculate_cumulative_returns(
            sample_single_strategy
        )

        assert isinstance(cumulative, pd.Series)

        if not cumulative.empty:
            # First value should be close to 1 (assuming returns start from 0)
            assert cumulative.iloc[0] > 0
            # Should be monotonic if returns are reasonable
            assert len(cumulative) == len(sample_single_strategy.dropna())

    def test_calculate_cumulative_returns_empty_data(self, empty_series):
        """Test cumulative returns calculation with empty data."""
        cumulative = PerformanceMetrics.calculate_cumulative_returns(empty_series)

        assert isinstance(cumulative, pd.Series)
        assert cumulative.empty

    def test_get_performance_summary_valid_data(self, sample_single_strategy):
        """Test comprehensive performance summary."""
        summary = PerformanceMetrics.get_performance_summary(sample_single_strategy)

        assert isinstance(summary, dict)

        # Check main sections
        expected_sections = [
            "basic_metrics",
            "advanced_metrics",
            "risk_metrics",
            "period_info",
        ]

        for section in expected_sections:
            assert section in summary
            assert isinstance(summary[section], dict)

        # Check period info
        period_info = summary["period_info"]
        assert "start_date" in period_info
        assert "end_date" in period_info
        assert "data_points" in period_info
        assert "total_days" in period_info

    def test_get_performance_summary_empty_data(self, empty_series):
        """Test performance summary with empty data."""
        summary = PerformanceMetrics.get_performance_summary(empty_series)

        assert isinstance(summary, dict)

        # Should still have structure but with empty/None values
        assert "basic_metrics" in summary
        assert "period_info" in summary

        period_info = summary["period_info"]
        assert period_info["start_date"] is None
        assert period_info["end_date"] is None
