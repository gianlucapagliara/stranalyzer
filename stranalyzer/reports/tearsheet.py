"""Tearsheet report generation functionality."""

import os
import tempfile
from datetime import datetime

import pandas as pd

# Import quantstats-lumi for better compatibility
try:
    import quantstats_lumi as qs
except ImportError:
    import quantstats as qs


class TearsheetGenerator:
    """Generates comprehensive tearsheet reports for strategy analysis using QuantStats."""

    @staticmethod
    def generate_quantstats_tearsheet(
        returns: pd.Series,
        benchmark: pd.Series | None = None,
        output_file: str | None = None,
        title: str = "Dynamic Carry Trading Strategy Performance Report",
        periods_per_year: int = 365,
        compounded: bool = True,
    ) -> str:
        """
        Generate a complete QuantStats tearsheet.

        This matches your original example:
        qs.reports.html(
            self.returns_series,
            benchmark=benchmark,
            output=output_file,
            title="Dynamic Carry Trading Strategy Performance Report"
        )

        Args:
            returns: Strategy returns series
            benchmark: Benchmark returns series (optional)
            output_file: Output file path (optional)
            title: Report title

        Returns:
            File path if output_file specified, otherwise HTML string
        """
        # Clean the returns data
        clean_returns = returns.dropna()
        if len(clean_returns) == 0:
            raise ValueError("No valid returns data available")

        # Clean benchmark data if provided
        clean_benchmark = None
        if benchmark is not None:
            clean_benchmark = benchmark.dropna()
            if len(clean_benchmark) > 0:
                # Align benchmark with returns
                common_dates = clean_returns.index.intersection(clean_benchmark.index)
                if len(common_dates) > 0:
                    clean_returns = clean_returns.reindex(common_dates)
                    clean_benchmark = clean_benchmark.reindex(common_dates)
                else:
                    clean_benchmark = None

        if output_file:
            # Generate and save to file
            qs.reports.html(
                clean_returns,
                benchmark=clean_benchmark,
                output=output_file,
                title=title,
                compounded=compounded,
                periods_per_year=periods_per_year,
            )
            return output_file
        else:
            # For returning HTML string, use temporary file approach
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".html", delete=False
            ) as tmp_file:
                temp_path = tmp_file.name

            try:
                # Generate to temporary file
                qs.reports.html(
                    clean_returns,
                    benchmark=clean_benchmark,
                    output=temp_path,
                    title=title,
                )

                # Read the generated HTML
                with open(temp_path, encoding="utf-8") as f:
                    html_content = f.read()

                return html_content

            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

    @staticmethod
    def generate_html_tearsheet(
        returns: pd.Series,
        benchmark: pd.Series | None = None,
        strategy_name: str = "Strategy",
        title: str | None = None,
        output_file: str | None = None,
    ) -> str:
        """
        Generate HTML tearsheet report using QuantStats.

        Args:
            returns: Strategy returns series
            benchmark: Benchmark returns series (optional)
            strategy_name: Name of the strategy
            title: Report title (optional)
            output_file: Output file path (optional, if None returns HTML string)

        Returns:
            HTML content as string or file path if output_file is specified
        """
        if title is None:
            title = f"{strategy_name} Performance Report"

        return TearsheetGenerator.generate_quantstats_tearsheet(
            returns=returns, benchmark=benchmark, output_file=output_file, title=title
        )

    @staticmethod
    def generate_csv_export(
        returns: pd.Series,
        strategy_name: str,
        include_cumulative: bool = True,
        include_drawdown: bool = True,
    ) -> str:
        """Generate CSV export of strategy data."""

        export_data = pd.DataFrame(
            {"date": returns.index, f"{strategy_name}_returns": returns.values}
        )

        if include_cumulative:
            cumulative_returns = (1 + returns).cumprod()
            export_data[f"{strategy_name}_cumulative"] = cumulative_returns.values

        if include_drawdown:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            export_data[f"{strategy_name}_drawdown"] = drawdown.values

        return str(export_data.to_csv(index=False))

    @staticmethod
    def generate_summary_report(
        strategies: dict[str, pd.Series], title: str = "Strategy Summary Report"
    ) -> str:
        """Generate summary report comparing multiple strategies."""

        from ..analysis.metrics import PerformanceMetrics

        # Calculate metrics for each strategy
        strategy_metrics = {}
        for name, returns in strategies.items():
            strategy_metrics[name] = PerformanceMetrics.calculate_basic_metrics(returns)

        # Create comparison table
        table_rows = ""
        for strategy_name, metrics in strategy_metrics.items():
            table_rows += f"""
            <tr>
                <td>{strategy_name}</td>
                <td>{metrics.get("total_return", 0):.2%}</td>
                <td>{metrics.get("cagr", 0):.2%}</td>
                <td>{metrics.get("volatility", 0):.2%}</td>
                <td>{metrics.get("sharpe_ratio", 0):.2f}</td>
                <td>{metrics.get("max_drawdown", 0):.2%}</td>
                <td>{metrics.get("calmar_ratio", 0):.2f}</td>
            </tr>
            """

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ color: #333; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
                .section {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .positive {{ color: #388e3c; }}
                .negative {{ color: #d32f2f; }}
            </style>
        </head>
        <body>
            <h1 class="header">{title}</h1>
            
            <div class="section">
                <h2>Performance Comparison</h2>
                <table>
                    <tr>
                        <th>Strategy</th>
                        <th>Total Return</th>
                        <th>CAGR</th>
                        <th>Volatility</th>
                        <th>Sharpe Ratio</th>
                        <th>Max Drawdown</th>
                        <th>Calmar Ratio</th>
                    </tr>
                    {table_rows}
                </table>
            </div>
            
            <div class="section">
                <h2>Report Generated</h2>
                <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p>Number of strategies: {len(strategies)}</p>
            </div>
            
        </body>
        </html>
        """

        return html_content
