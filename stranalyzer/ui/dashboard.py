"""Main dashboard orchestration for the strategy analyzer."""

from typing import Any

import pandas as pd
import streamlit as st

from ..analysis import StrategyAnalyzer
from ..data import DataLoader, DataValidator
from ..portfolio import PortfolioComposer
from ..reports import TearsheetGenerator
from ..visualization import ChartGenerator
from .components import UIComponents


class Dashboard:
    """Main dashboard class that orchestrates the entire application."""

    def __init__(self) -> None:
        """Initialize the dashboard with all components."""
        self.data_loader = DataLoader()
        self.data_validator = DataValidator()
        self.portfolio_composer = PortfolioComposer()
        self.strategy_analyzer = StrategyAnalyzer()
        self.chart_generator = ChartGenerator()
        self.tearsheet_generator = TearsheetGenerator()
        self.ui_components = UIComponents()

        # Initialize session state
        self._initialize_session_state()

    def _initialize_session_state(self) -> None:
        """Initialize Streamlit session state."""
        if "uploaded_data" not in st.session_state:
            st.session_state.uploaded_data = {}
        if "composite_strategies" not in st.session_state:
            st.session_state.composite_strategies = {}
        if "use_sample_data" not in st.session_state:
            st.session_state.use_sample_data = False

    def _render_home_section(self) -> None:
        """Render the home section with dashboard introduction."""

        st.markdown("""### 🚀 Welcome to the Strategies Analyzer Dashboard""")

        col1, col2 = st.columns(2)

        with col2:
            st.markdown("""
            #### 📋 What You Can Do
            
            ##### 📁 File Upload & Data Management
            - **Upload CSV Files**: Import your strategy returns data in a simple CSV format
            - **Sample Data Generation**: Generate realistic sample data for testing and exploration
            - **Data Validation**: Automatic validation ensures your data is properly formatted
            - **Multiple Formats**: Support for single-strategy or multi-strategy CSV files
            
            ##### 📈 Strategy Analysis
            - **Performance Metrics**: Calculate comprehensive performance statistics including:
            - Sharpe Ratio, Sortino Ratio, Calmar Ratio
            - Maximum Drawdown, Volatility, Beta
            - Value at Risk (VaR) and Expected Shortfall
            - Interactive Charts:
                - Cumulative returns comparison
                - Drawdown analysis
                - Rolling metrics (Sharpe, volatility)
                - Return distribution analysis
                - Correlation matrices
                - Risk-return scatter plots
                - Monthly returns heatmaps
            
            ##### 🏗️ Portfolio Composition
            - **Custom Portfolios**: Create weighted combinations of your strategies
            - **Weight Optimization**: Flexible weight assignment with normalization
            - **Composite Strategies**: Build and manage multiple portfolio combinations
            - **Dynamic Rebalancing**: Analyze how different weightings affect performance
            
            ##### 📊 Tearsheet Reports
            - **Professional Reports**: Generate comprehensive PDF tearsheets
            - **Benchmark Comparison**: Compare strategies against benchmarks
            - **Detailed Analytics**: In-depth analysis with tables and charts
            - **Export Options**: Download reports for presentations and documentation
            """)

        with col1:
            st.markdown("""
            #### 🎯 Getting Started
            
            1. **Start with Data**: Go to the "📁 File Upload" tab to upload your strategy data or generate sample data
            2. **Analyze Performance**: Use the "📈 Strategy Analysis" tab to compare strategies and view detailed metrics
            3. **Build Portfolios**: Create custom portfolio combinations in the "🏗️ Portfolio Composition" tab
            4. **Generate Reports**: Create professional tearsheets in the "📊 Tearsheet Reports" tab
            
            #### 📊 Data Format
            
            Your CSV files should have the following structure:
            ```
            date,strategy_name
            2023-01-01,0.015
            2023-01-02,-0.008
            2023-01-03,0.012
            ...
            ```
            
            Or for multiple strategies:
            ```
            date,strategy_1,strategy_2,strategy_3
            2023-01-01,0.015,-0.008,0.012
            2023-01-02,-0.008,0.021,-0.005
            ...
            ```
            
            """)

    def run(self) -> None:
        """Run the main dashboard application."""
        # Configure streamlit page
        st.set_page_config(page_title="Strategies Analyzer", layout="wide")
        st.title("Strategies Analyzer Dashboard")

        # Main sections in tabs
        tab_home, tab_upload, tab_analysis, tab_portfolio, tab_tearsheet = st.tabs(
            [
                "🏠 Home",
                "📁 File Upload",
                "📈 Strategy Analysis",
                "🏗️ Portfolio Composition",
                "📊 Tearsheet Reports",
            ]
        )

        # Home Section
        with tab_home:
            self._render_home_section()

        # File Upload Section
        with tab_upload:
            self._render_file_upload_section()

        # Get available data
        available_data = self._get_available_data()

        if not available_data:
            st.warning(
                "📊 No data available for analysis. Please upload files or generate sample data first."
            )
            return

        # Strategy Analysis Section
        with tab_analysis:
            self._render_main_analysis_section(available_data)

        # Portfolio Composition Section
        with tab_portfolio:
            self._render_portfolio_composition_section(available_data)

        # Tearsheet Reports Section
        with tab_tearsheet:
            self._render_tearsheet_section(available_data)

    def _render_file_upload_section(self) -> None:
        """Render the file upload section."""
        upload_results = self.ui_components.render_file_upload_section(
            self.data_loader, "uploaded_data"
        )

        # No need to update session state or trigger rerun here since it's done in the UI component
        # Streamlit will automatically refresh when session state changes

        # Render data summary
        # self.ui_components.render_data_summary(
        #     self.data_loader, st.session_state.uploaded_data
        # )

    def _get_available_data(self) -> dict[str, pd.Series]:
        """Get all available data (uploaded + composite strategies)."""
        available_data = st.session_state.uploaded_data.copy()

        # Add composite strategies
        for name, info in st.session_state.composite_strategies.items():
            available_data[name] = info["data"]

        return available_data

    def _render_portfolio_composition_section(
        self, available_data: dict[str, pd.Series]
    ) -> None:
        """Render portfolio composition section."""
        base_strategies = [
            name
            for name, data in st.session_state.uploaded_data.items()
            if not data.empty
        ]

        if not base_strategies:
            st.info(
                "📊 No strategies available for portfolio composition. Please upload data first."
            )
            return

        composition_results = self.ui_components.render_portfolio_composition_section(
            self.portfolio_composer, base_strategies, "composite_strategies"
        )

        # Create composite strategy if requested
        if composition_results["new_composite"]:
            self._create_composite_strategy(
                composition_results["composite_data"], st.session_state.uploaded_data
            )

        # Render existing composite strategies
        removed_strategies = self.ui_components.render_existing_composite_strategies(
            st.session_state.composite_strategies, "composite_strategies"
        )

        # Remove strategies if requested
        for strategy_name in removed_strategies:
            if strategy_name in st.session_state.composite_strategies:
                del st.session_state.composite_strategies[strategy_name]
                st.rerun()

    def _create_composite_strategy(
        self, composite_config: dict[str, Any], base_strategies: dict[str, pd.Series]
    ) -> None:
        """Create a composite strategy."""
        result = self.portfolio_composer.create_composite_strategy(
            base_strategies=base_strategies,
            weights=composite_config["weights"],
            name=composite_config["name"],
            normalize_weights=True,
            enable_rebalancing=composite_config.get("enable_rebalancing", False),
            rebalancing_tolerance=composite_config.get("rebalancing_tolerance", 0.05),
            rebalancing_cost=composite_config.get("rebalancing_cost", 0.001),
            cost_on_rebalanced_amount=composite_config.get(
                "cost_on_rebalanced_amount", True
            ),
        )

        if result["success"]:
            strategy_info = {
                "data": result["data"],
                "weights": result["info"]["weights"],
                "created_at": pd.Timestamp.now(),
                "contributions": result["info"]["contributions"],
                "enable_rebalancing": result["info"].get("enable_rebalancing", False),
            }

            if result["info"].get("enable_rebalancing"):
                strategy_info["rebalancing_info"] = result["info"].get(
                    "rebalancing", {}
                )

            st.session_state.composite_strategies[composite_config["name"]] = (
                strategy_info
            )
            st.rerun()
        else:
            for error in result["errors"]:
                st.error(f"❌ {error}")

    def _render_tearsheet_section(self, available_data: dict[str, pd.Series]) -> None:
        """Render tearsheet generation section."""
        available_strategies = [
            name for name, data in available_data.items() if not data.empty
        ]

        if not available_strategies:
            st.info(
                "📊 No strategies available for tearsheet generation. Please upload data first."
            )
            return

        self.ui_components.render_tearsheet_section(
            self.tearsheet_generator,
            available_strategies,
            st.session_state.uploaded_data,
            st.session_state.composite_strategies,
        )

    def _render_main_analysis_section(
        self, available_data: dict[str, pd.Series]
    ) -> None:
        """Render the main analysis section."""
        # Filter out empty strategies
        valid_strategies = [
            name for name, data in available_data.items() if not data.empty
        ]

        if not valid_strategies:
            st.info(
                "📊 No valid strategies available for analysis. Please upload data first."
            )
            return

        # Strategy selector
        selected_strategies = self.ui_components.render_strategy_selector(
            valid_strategies,
            "strategy_selector",
            "Select Strategies to Analyze:",
            "Select one or more strategies to compare (includes composite strategies)",
        )

        if not selected_strategies:
            st.warning("Please select at least one strategy to analyze.")
            return

        # Perform analysis
        analysis_results = self.strategy_analyzer.analyze_multiple_strategies(
            available_data, selected_strategies
        )

        if not analysis_results:
            st.error("Analysis failed. Please check your data.")
            return

        # Render analysis results
        self._render_analysis_results(analysis_results, selected_strategies)

        # Render data export section
        self.ui_components.render_data_export_section(
            available_data, selected_strategies
        )

    def _render_analysis_results(
        self, analysis_results: dict[str, Any], selected_strategies: list[str]
    ) -> None:
        """Render comprehensive analysis results."""

        # Performance metrics table
        self.ui_components.render_performance_metrics_table(
            analysis_results.get("individual_analyses", {}),
            "Strategy Performance Comparison",
        )

        # Charts section
        self._render_charts_section(analysis_results, selected_strategies)

        # Detailed analysis section
        self._render_detailed_analysis_section(analysis_results)

    def _render_charts_section(
        self, analysis_results: dict[str, Any], selected_strategies: list[str]
    ) -> None:
        """Render charts section."""

        combined_data = analysis_results.get("combined_data", pd.DataFrame())

        if combined_data.empty:
            st.warning("No data available for charts.")
            return

        # Prepare data for charts
        chart_data = {
            col: combined_data[col].dropna()
            for col in selected_strategies
            if col in combined_data.columns
        }

        composite_strategy_names = list(st.session_state.composite_strategies.keys())

        # Cumulative Returns Chart
        st.subheader("📈 Cumulative Returns")
        fig_cumulative = self.chart_generator.create_cumulative_returns_chart(
            chart_data, composite_strategies=composite_strategy_names
        )
        st.plotly_chart(fig_cumulative, use_container_width=True)

        # Drawdown Chart
        st.subheader("📉 Drawdown Analysis")
        fig_drawdown = self.chart_generator.create_drawdown_chart(
            chart_data, composite_strategies=composite_strategy_names
        )
        st.plotly_chart(fig_drawdown, use_container_width=True)

        # Rolling Metrics Charts
        st.subheader("🔄 Rolling Metrics Analysis")

        # Rolling Sharpe Ratio
        fig_rolling_sharpe = self.chart_generator.create_rolling_metrics_chart(
            chart_data,
            metric="sharpe",
            window=365,
            composite_strategies=composite_strategy_names,
        )
        st.plotly_chart(fig_rolling_sharpe, use_container_width=True)

        # Rolling Volatility
        fig_rolling_vol = self.chart_generator.create_rolling_metrics_chart(
            chart_data,
            metric="volatility",
            window=365,
            composite_strategies=composite_strategy_names,
        )
        st.plotly_chart(fig_rolling_vol, use_container_width=True)

        # Return Distribution
        st.subheader("📊 Return Distribution")
        fig_distribution = self.chart_generator.create_return_distribution_chart(
            chart_data
        )
        st.plotly_chart(fig_distribution, use_container_width=True)

        # Correlation Matrix (if multiple strategies)
        if len(selected_strategies) > 1:
            st.subheader("🔗 Strategy Correlation Matrix")
            fig_correlation = self.chart_generator.create_correlation_matrix(chart_data)
            st.plotly_chart(fig_correlation, use_container_width=True)

            # Risk-Return Scatter
            st.subheader("⚖️ Risk-Return Analysis")
            fig_risk_return = self.chart_generator.create_risk_return_scatter(
                chart_data, composite_strategies=composite_strategy_names
            )
            st.plotly_chart(fig_risk_return, use_container_width=True)

        # Monthly Returns Heatmaps
        st.subheader("🗓️ Monthly Returns Analysis")
        for strategy in selected_strategies:
            if strategy in chart_data:
                strategy_type = (
                    " (Composite)" if strategy in composite_strategy_names else ""
                )
                fig_monthly = self.chart_generator.create_monthly_returns_heatmap(
                    chart_data[strategy], f"{strategy}{strategy_type}"
                )
                st.plotly_chart(fig_monthly, use_container_width=True)

    def _render_detailed_analysis_section(
        self, analysis_results: dict[str, Any]
    ) -> None:
        """Render detailed analysis section."""

        with st.expander("📋 Detailed Analysis", expanded=False):
            # Summary table
            summary_table = analysis_results.get("summary_table", pd.DataFrame())
            if not summary_table.empty:
                st.subheader("📊 Detailed Performance Metrics")
                st.dataframe(summary_table, use_container_width=True)

            # Comparative analysis
            comparative_analysis = analysis_results.get("comparative_analysis", {})
            if comparative_analysis:
                st.subheader("🔍 Comparative Analysis")

                # Correlation matrix
                corr_matrix = comparative_analysis.get(
                    "correlation_matrix", pd.DataFrame()
                )
                if not corr_matrix.empty:
                    st.write("**Correlation Matrix:**")
                    st.dataframe(corr_matrix, use_container_width=True)

                # Pairwise metrics
                pairwise_metrics = comparative_analysis.get("pairwise_metrics", {})
                if pairwise_metrics:
                    st.write("**Pairwise Comparison Metrics:**")
                    for pair, metrics in pairwise_metrics.items():
                        if metrics:
                            st.write(f"**{pair}:**")
                            try:
                                # Filter out any non-serializable values
                                filtered_pair_metrics = {}
                                for key, value in metrics.items():
                                    if (
                                        isinstance(value, (int, float, str, bool))
                                        or value is None
                                    ):
                                        filtered_pair_metrics[key] = value
                                    else:
                                        # Convert complex objects to string representation
                                        filtered_pair_metrics[key] = str(value)

                                metrics_df = pd.DataFrame([filtered_pair_metrics]).T
                                metrics_df.columns = ["Value"]
                                st.dataframe(metrics_df)
                            except Exception as e:
                                st.error(
                                    f"Error displaying pairwise metrics for {pair}: {str(e)}"
                                )
                                # Fallback: display as text
                                for key, value in metrics.items():
                                    st.write(f"  - **{key}**: {value}")

            # Individual strategy details
            individual_analyses = analysis_results.get("individual_analyses", {})
            if individual_analyses:
                st.subheader("📈 Individual Strategy Details")

                for strategy_name, analysis in individual_analyses.items():
                    with st.expander(f"📊 {strategy_name} Details", expanded=False):
                        # Basic metrics
                        basic_metrics = analysis.get("summary", {}).get(
                            "basic_metrics", {}
                        )
                        if basic_metrics:
                            st.write("**Basic Metrics:**")
                            try:
                                # Filter out any non-serializable values
                                filtered_metrics = {}
                                for key, value in basic_metrics.items():
                                    if (
                                        isinstance(value, (int, float, str, bool))
                                        or value is None
                                    ):
                                        filtered_metrics[key] = value
                                    else:
                                        # Convert complex objects to string representation
                                        filtered_metrics[key] = str(value)

                                basic_df = pd.DataFrame([filtered_metrics]).T
                                basic_df.columns = ["Value"]
                                st.dataframe(basic_df)
                            except Exception as e:
                                st.error(f"Error displaying basic metrics: {str(e)}")
                                # Fallback: display as text
                                for key, value in basic_metrics.items():
                                    st.write(f"- **{key}**: {value}")

                        # Return distribution
                        return_dist = analysis.get("return_distribution", {})
                        if return_dist:
                            st.write("**Return Distribution:**")
                            try:
                                # Filter out any non-serializable values
                                filtered_dist = {}
                                for key, value in return_dist.items():
                                    if (
                                        isinstance(value, (int, float, str, bool))
                                        or value is None
                                    ):
                                        filtered_dist[key] = value
                                    else:
                                        # Convert complex objects to string representation
                                        filtered_dist[key] = str(value)

                                dist_df = pd.DataFrame([filtered_dist]).T
                                dist_df.columns = ["Value"]
                                st.dataframe(dist_df)
                            except Exception as e:
                                st.error(
                                    f"Error displaying return distribution: {str(e)}"
                                )
                                # Fallback: display as text
                                for key, value in return_dist.items():
                                    st.write(f"- **{key}**: {value}")


def main() -> None:
    """Main function to run the dashboard."""
    dashboard = Dashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
