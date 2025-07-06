"""Tests for the TearsheetGenerator class."""

from stranalyzer.reports.tearsheet import TearsheetGenerator


class TestTearsheetGenerator:
    """Test cases for TearsheetGenerator class."""

    def test_generate_csv_export_basic(self, sample_single_strategy):
        """Test basic CSV export generation."""
        csv_data = TearsheetGenerator.generate_csv_export(
            returns=sample_single_strategy,
            strategy_name="TestStrategy",
            include_cumulative=False,
            include_drawdown=False,
        )

        assert isinstance(csv_data, str)
        assert len(csv_data) > 0

        # Check CSV structure
        lines = csv_data.strip().split("\n")
        assert len(lines) > 1  # Header + data

        header = lines[0]
        assert "date" in header
        assert "TestStrategy_returns" in header

    def test_generate_csv_export_full(self, sample_single_strategy):
        """Test full CSV export generation with all columns."""
        csv_data = TearsheetGenerator.generate_csv_export(
            returns=sample_single_strategy,
            strategy_name="TestStrategy",
            include_cumulative=True,
            include_drawdown=True,
        )

        assert isinstance(csv_data, str)
        assert len(csv_data) > 0

        # Check that all columns are present
        header = csv_data.split("\n")[0]
        assert "TestStrategy_returns" in header
        assert "TestStrategy_cumulative" in header
        assert "TestStrategy_drawdown" in header

    def test_generate_csv_export_empty_data(self, empty_series):
        """Test CSV export with empty data."""
        csv_data = TearsheetGenerator.generate_csv_export(
            returns=empty_series, strategy_name="EmptyStrategy"
        )

        assert isinstance(csv_data, str)
        # Should still have header even with no data
        assert "EmptyStrategy_returns" in csv_data

    def test_generate_summary_report_single_strategy(self, sample_single_strategy):
        """Test summary report generation with single strategy."""
        strategies = {"TestStrategy": sample_single_strategy}

        html_report = TearsheetGenerator.generate_summary_report(strategies)

        assert isinstance(html_report, str)
        assert len(html_report) > 0
        assert "Strategy Summary Report" in html_report
        assert "Performance Comparison" in html_report
        assert "TestStrategy" in html_report

    def test_generate_summary_report_multiple_strategies(self, sample_returns_data):
        """Test summary report generation with multiple strategies."""
        html_report = TearsheetGenerator.generate_summary_report(sample_returns_data)

        assert isinstance(html_report, str)
        assert len(html_report) > 0
        assert "Strategy Summary Report" in html_report

        # Check that all strategies are included
        for strategy_name in sample_returns_data.keys():
            assert strategy_name in html_report

        # Check that performance metrics are included
        assert "Total Return" in html_report
        assert "CAGR" in html_report
        assert "Sharpe Ratio" in html_report

    def test_generate_summary_report_custom_title(self, sample_returns_data):
        """Test summary report generation with custom title."""
        custom_title = "Custom Portfolio Analysis Report"

        html_report = TearsheetGenerator.generate_summary_report(
            sample_returns_data, title=custom_title
        )

        assert isinstance(html_report, str)
        assert custom_title in html_report

    def test_generate_summary_report_empty_data(self):
        """Test summary report generation with empty data."""
        html_report = TearsheetGenerator.generate_summary_report({})

        assert isinstance(html_report, str)
        assert len(html_report) > 0
        assert "Strategy Summary Report" in html_report
        # Should handle empty data gracefully

    def test_css_styling_included(self, sample_single_strategy):
        """Test that CSS styling is included in reports."""
        html_report = TearsheetGenerator.generate_html_tearsheet(
            returns=sample_single_strategy, strategy_name="TestStrategy"
        )

        assert "<style>" in html_report
        assert "</style>" in html_report
        assert "font-family" in html_report
        assert "margin" in html_report
