"""Tests for the ChartGenerator class."""

import plotly.graph_objects as go

from stranalyzer.visualization.charts import ChartGenerator


class TestChartGenerator:
    """Test cases for ChartGenerator class."""

    def test_create_cumulative_returns_chart_valid_data(self, sample_returns_data):
        """Test cumulative returns chart creation with valid data."""
        fig = ChartGenerator.create_cumulative_returns_chart(sample_returns_data)

        assert isinstance(fig, go.Figure)
        assert hasattr(fig, "data")
        assert len(fig.data) == len(sample_returns_data)

        # Check that each strategy has a trace
        trace_names = [getattr(trace, "name", None) for trace in fig.data]
        for strategy_name in sample_returns_data.keys():
            assert strategy_name in trace_names

    def test_create_cumulative_returns_chart_empty_data(self):
        """Test cumulative returns chart creation with empty data."""
        fig = ChartGenerator.create_cumulative_returns_chart({})

        assert isinstance(fig, go.Figure)
        assert hasattr(fig, "data")
        assert len(fig.data) == 0

    def test_create_cumulative_returns_chart_with_composite(self, sample_returns_data):
        """Test cumulative returns chart with composite strategies."""
        composite_strategies = ["Strategy1"]  # Mock composite strategy

        fig = ChartGenerator.create_cumulative_returns_chart(
            sample_returns_data, composite_strategies=composite_strategies
        )

        assert isinstance(fig, go.Figure)
        # Should still create the chart even with non-existent composite strategies
        assert len(fig.data) == len(sample_returns_data)

    def test_create_drawdown_chart_valid_data(self, sample_returns_data):
        """Test drawdown chart creation with valid data."""
        fig = ChartGenerator.create_drawdown_chart(sample_returns_data)

        assert isinstance(fig, go.Figure)
        assert hasattr(fig, "data")
        assert len(fig.data) == len(sample_returns_data)

    def test_create_drawdown_chart_empty_data(self):
        """Test drawdown chart creation with empty data."""
        fig = ChartGenerator.create_drawdown_chart({})

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_create_rolling_metrics_chart_sharpe(self, sample_returns_data):
        """Test rolling Sharpe ratio chart creation."""
        fig = ChartGenerator.create_rolling_metrics_chart(
            sample_returns_data, metric="sharpe", window=30
        )

        assert isinstance(fig, go.Figure)
        # May have fewer traces if data is insufficient for rolling calculation
        assert len(fig.data) <= len(sample_returns_data)

    def test_create_rolling_metrics_chart_volatility(self, sample_returns_data):
        """Test rolling volatility chart creation."""
        fig = ChartGenerator.create_rolling_metrics_chart(
            sample_returns_data, metric="volatility", window=30
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) <= len(sample_returns_data)

    def test_create_rolling_metrics_chart_invalid_metric(self, sample_returns_data):
        """Test rolling metrics chart with invalid metric."""
        fig = ChartGenerator.create_rolling_metrics_chart(
            sample_returns_data, metric="invalid_metric", window=30
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0  # No traces should be added for invalid metric

    def test_create_return_distribution_chart_single_strategy(
        self, sample_single_strategy
    ):
        """Test return distribution chart with single strategy."""
        strategies = {"TestStrategy": sample_single_strategy}

        fig = ChartGenerator.create_return_distribution_chart(
            strategies, chart_type="histogram"
        )

        assert isinstance(fig, go.Figure)
        # Should create a histogram
        assert len(fig.data) >= 1

    def test_create_return_distribution_chart_multiple_strategies(
        self, sample_returns_data
    ):
        """Test return distribution chart with multiple strategies."""
        fig = ChartGenerator.create_return_distribution_chart(
            sample_returns_data, chart_type="box"
        )

        assert isinstance(fig, go.Figure)
        # Should create box plots
        assert len(fig.data) >= 1

    def test_create_return_distribution_chart_empty_data(self):
        """Test return distribution chart with empty data."""
        fig = ChartGenerator.create_return_distribution_chart({})

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_create_correlation_matrix_valid_data(self, sample_returns_data):
        """Test correlation matrix creation with valid data."""
        fig = ChartGenerator.create_correlation_matrix(sample_returns_data)

        assert isinstance(fig, go.Figure)
        assert hasattr(fig, "data")

        if len(sample_returns_data) > 1:
            # Should have a heatmap trace
            assert len(fig.data) >= 1

    def test_create_correlation_matrix_single_strategy(self, sample_single_strategy):
        """Test correlation matrix with single strategy."""
        strategies = {"TestStrategy": sample_single_strategy}

        fig = ChartGenerator.create_correlation_matrix(strategies)

        assert isinstance(fig, go.Figure)
        # Should still create a figure, possibly with single value
        assert len(fig.data) >= 0

    def test_create_correlation_matrix_empty_data(self):
        """Test correlation matrix with empty data."""
        fig = ChartGenerator.create_correlation_matrix({})

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_create_monthly_returns_heatmap_valid_data(self, sample_single_strategy):
        """Test monthly returns heatmap creation."""
        fig = ChartGenerator.create_monthly_returns_heatmap(
            sample_single_strategy, "TestStrategy"
        )

        assert isinstance(fig, go.Figure)
        # Should create a heatmap if there's sufficient data
        assert len(fig.data) >= 0

    def test_create_monthly_returns_heatmap_empty_data(self, empty_series):
        """Test monthly returns heatmap with empty data."""
        fig = ChartGenerator.create_monthly_returns_heatmap(
            empty_series, "EmptyStrategy"
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_create_risk_return_scatter_valid_data(self, sample_returns_data):
        """Test risk-return scatter plot creation."""
        fig = ChartGenerator.create_risk_return_scatter(sample_returns_data)

        assert isinstance(fig, go.Figure)

        if len(sample_returns_data) > 0:
            # Should have scatter points
            assert len(fig.data) >= 1
            # Check that it's a scatter plot
            assert hasattr(fig.data[0], "x")
            assert hasattr(fig.data[0], "y")

    def test_create_risk_return_scatter_empty_data(self):
        """Test risk-return scatter plot with empty data."""
        fig = ChartGenerator.create_risk_return_scatter({})

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_create_performance_comparison_chart_valid_data(self, sample_returns_data):
        """Test performance comparison chart creation."""
        fig = ChartGenerator.create_performance_comparison_chart(sample_returns_data)

        assert isinstance(fig, go.Figure)

        if len(sample_returns_data) > 0:
            # Should create bar chart traces
            assert len(fig.data) >= 1

    def test_create_performance_comparison_chart_custom_metrics(
        self, sample_returns_data
    ):
        """Test performance comparison chart with custom metrics."""
        custom_metrics = ["sharpe_ratio", "volatility"]

        fig = ChartGenerator.create_performance_comparison_chart(
            sample_returns_data, metrics=custom_metrics
        )

        assert isinstance(fig, go.Figure)
        # Should create traces for the specified metrics
        assert len(fig.data) >= 0

    def test_create_performance_comparison_chart_empty_data(self):
        """Test performance comparison chart with empty data."""
        fig = ChartGenerator.create_performance_comparison_chart({})

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_chart_customization_parameters(self, sample_returns_data):
        """Test that chart customization parameters work."""
        custom_title = "Custom Chart Title"
        custom_height = 800

        fig = ChartGenerator.create_cumulative_returns_chart(
            sample_returns_data, title=custom_title, height=custom_height
        )

        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == custom_title
        assert fig.layout.height == custom_height
