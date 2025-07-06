"""Integration tests for the stranalyzer package."""

import numpy as np
import pandas as pd

from stranalyzer import (
    ChartGenerator,
    DataLoader,
    PortfolioComposer,
    StrategyAnalyzer,
    TearsheetGenerator,
)


class TestIntegration:
    """Integration tests for the complete workflow."""

    def test_full_workflow_sample_data(self):
        """Test complete workflow with sample data."""
        # 1. Load sample data
        loader = DataLoader()
        sample_data = loader.generate_sample_data()

        assert len(sample_data) == 5
        assert all(isinstance(series, pd.Series) for series in sample_data.values())

        # 2. Create composite portfolio
        composer = PortfolioComposer()

        weights = {
            "Conservative_Strategy": 0.4,
            "Aggressive_Strategy": 0.4,
            "Balanced_Strategy": 0.2,
        }

        result = composer.create_composite_strategy(
            base_strategies=sample_data, weights=weights, name="TestPortfolio"
        )

        assert result["success"] is True
        assert "TestPortfolio" in composer.composite_strategies

        # 3. Analyze strategies
        analyzer = StrategyAnalyzer()

        # Include both original and composite strategies
        all_strategies = sample_data.copy()
        all_strategies["TestPortfolio"] = result["data"]

        selected_strategies = [
            "Conservative_Strategy",
            "Aggressive_Strategy",
            "TestPortfolio",
        ]
        analysis_results = analyzer.analyze_multiple_strategies(
            all_strategies, selected_strategies
        )

        assert "individual_analyses" in analysis_results
        assert "comparative_analysis" in analysis_results
        assert len(analysis_results["individual_analyses"]) == 3

        # 4. Generate charts
        chart_generator = ChartGenerator()

        chart_data = {name: all_strategies[name] for name in selected_strategies}

        # Test various chart types
        cumulative_fig = chart_generator.create_cumulative_returns_chart(chart_data)
        assert cumulative_fig is not None
        assert hasattr(cumulative_fig, "data")
        assert len(cumulative_fig.data) == 3

        drawdown_fig = chart_generator.create_drawdown_chart(chart_data)
        assert drawdown_fig is not None

        correlation_fig = chart_generator.create_correlation_matrix(chart_data)
        assert correlation_fig is not None

        # 5. Generate tearsheet
        tearsheet_generator = TearsheetGenerator()

        html_report = tearsheet_generator.generate_html_tearsheet(
            returns=result["data"],
            strategy_name="TestPortfolio",
            title="Integration Test Portfolio Report",
        )

        assert html_report is not None
        assert isinstance(html_report, str)
        assert len(html_report) > 0

    def test_error_handling_workflow(self):
        """Test error handling throughout the workflow."""
        # 1. Test with empty data
        composer = PortfolioComposer()

        result = composer.create_composite_strategy({}, {}, "EmptyPortfolio")
        assert result["success"] is False
        assert len(result["errors"]) > 0

        # 2. Test chart generation with empty data
        chart_generator = ChartGenerator()
        empty_fig = chart_generator.create_cumulative_returns_chart({})
        assert empty_fig is not None
        assert hasattr(empty_fig, "data")
        assert len(empty_fig.data) == 0

    def test_memory_efficiency_large_dataset(self):
        """Test memory efficiency with larger datasets."""
        # Create larger dataset
        np.random.seed(42)
        large_date_range = pd.date_range("2020-01-01", "2023-12-31", freq="D")

        large_strategies = {}
        for i in range(10):  # 10 strategies
            returns = np.random.normal(0.001, 0.02, len(large_date_range))
            large_strategies[f"Strategy_{i}"] = pd.Series(
                returns, index=large_date_range
            )

        # Test that operations complete without memory issues
        analyzer = StrategyAnalyzer()

        # Analyze subset to avoid excessive computation in tests
        selected = [f"Strategy_{i}" for i in range(3)]
        analysis = analyzer.analyze_multiple_strategies(large_strategies, selected)

        assert len(analysis["individual_analyses"]) == 3

        # Test chart generation
        chart_generator = ChartGenerator()
        subset_data = {name: large_strategies[name] for name in selected}

        fig = chart_generator.create_cumulative_returns_chart(subset_data)
        assert fig is not None
        assert hasattr(fig, "data")
        assert len(fig.data) == 3
