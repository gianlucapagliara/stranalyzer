"""UI components for the strategy analyzer dashboard."""

from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from stranalyzer.data.loader import DataLoader
from stranalyzer.portfolio import PortfolioComposer
from stranalyzer.reports import TearsheetGenerator


class UIComponents:
    """Provides reusable UI components for the dashboard."""

    @staticmethod
    def render_file_upload_section(
        data_loader: DataLoader,
        session_state_key: str = "uploaded_data",
    ) -> dict[str, Any]:
        """Render file upload section with sample data generation."""

        st.write("Upload your CSV files to analyze custom strategies")

        upload_results = {"new_data": False, "data": {}}

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📤 Upload Strategy Data")

            # File upload widget
            uploaded_files = st.file_uploader(
                "Choose CSV files",
                accept_multiple_files=True,
                type=["csv"],
                help="Upload CSV files with format: date,label or date,label_1,label_2,...",
                key="main_file_uploader",
            )

            # Clear all uploaded data option and download sample CSV format
            if st.button(
                "🗑️ Clear All Data", type="secondary", key="clear_all_data_btn"
            ):
                st.session_state.uploaded_data = {}
                st.session_state.composite_strategies = {}
                st.session_state.use_sample_data = False
                st.rerun()

            if uploaded_files:
                for uploaded_file in uploaded_files:
                    result = data_loader.process_uploaded_file(uploaded_file)

                    if result["success"]:
                        # Directly update session state
                        st.session_state.uploaded_data.update(result["data"])
                        upload_results["new_data"] = True

                    else:
                        for error in result["errors"]:
                            st.error(f"❌ {uploaded_file.name}: {error}")

        with col2:
            st.subheader("🧪 Sample Data")
            st.write("Generate sample data for testing")

            # Download sample CSV format
            sample_csv = data_loader.get_sample_csv_format()
            st.download_button(
                label="📥 Download Sample CSV Format",
                data=sample_csv,
                file_name="sample_format.csv",
                mime="text/csv",
                help="Download a sample CSV file showing the expected format",
                key="download_sample_csv_btn",
            )

            if st.button(
                "🎲 Generate Sample Data",
                type="secondary",
                key="generate_sample_data_btn",
            ):
                sample_data = data_loader.generate_sample_data()
                # Directly update session state
                st.session_state.uploaded_data.update(sample_data)
                st.session_state.use_sample_data = True
                upload_results["new_data"] = True
                st.success("✅ Sample data generated!")
                st.rerun()

            if st.session_state.get("use_sample_data", False):
                st.info("📊 Sample data active")
                if st.button(
                    "🗑️ Clear Sample Data",
                    type="secondary",
                    key="clear_sample_data_btn",
                ):
                    # Remove sample data keys
                    sample_keys = [
                        "Conservative_Strategy",
                        "Balanced_Strategy",
                        "Aggressive_Strategy",
                        "Market_Neutral",
                        "Momentum_Strategy",
                    ]
                    for key in sample_keys:
                        if key in st.session_state.uploaded_data:
                            del st.session_state.uploaded_data[key]
                    st.session_state.use_sample_data = False
                    st.rerun()

        if st.session_state.uploaded_data:
            with st.expander("Data Summary", expanded=True):
                summary_df = data_loader.get_data_summary(
                    st.session_state.uploaded_data
                )
                st.dataframe(summary_df, use_container_width=True)

        return upload_results

    @staticmethod
    def render_portfolio_composition_section(
        portfolio_composer: PortfolioComposer,
        available_strategies: list[str],
        session_state_key: str = "composite_strategies",
    ) -> dict[str, Any]:
        """Render portfolio composition section."""

        st.write("Create custom weighted combinations of strategies")

        composition_results = {"new_composite": False, "composite_data": {}}

        st.subheader("Strategy Weights")

        # Strategy selection for composition
        selected_for_composition = st.multiselect(
            "Select strategies to combine:",
            available_strategies,
            key="composition_selector",
            help="Choose strategies to include in your portfolio",
        )

        if selected_for_composition:
            # Weight inputs
            weights = {}
            total_weight = 0.0

            st.write("**Assign weights to each strategy:**")
            cols = st.columns(min(len(selected_for_composition), 4))

            for i, strategy in enumerate(selected_for_composition):
                with cols[i % 4]:
                    weight = st.number_input(
                        f"{strategy}",
                        min_value=0.0,
                        max_value=1.0,
                        value=1.0 / len(selected_for_composition),
                        step=0.01,
                        format="%.2f",
                        key=f"weight_{strategy}",
                    )
                    weights[strategy] = weight
                    total_weight += weight

            # Show total weight
            if total_weight > 0:
                st.metric("Total Weight", f"{total_weight:.2f}")

                # Normalize weights option
                normalize_weights = st.checkbox(
                    "Normalize weights to sum to 1.0", value=True
                )

                # Calculate normalized weights if needed
                if normalize_weights and total_weight != 1.0:
                    normalized_weights = {
                        k: v / total_weight for k, v in weights.items()
                    }
                    st.info(
                        f"Weights normalized. New weights: {', '.join([f'{k}: {v:.3f}' for k, v in normalized_weights.items()])}"
                    )
                else:
                    normalized_weights = weights

            # Composite strategy name
            composite_name = st.text_input(
                "Composite Strategy Name:",
                value=f"Portfolio_{len(st.session_state.get(session_state_key, {})) + 1}",
                key="composite_name",
            )

            # Create composite strategy
            if st.button("Create Composite Strategy", type="primary"):
                if composite_name and total_weight > 0:
                    # This would need access to the actual strategy data
                    # For now, we'll return the configuration
                    composition_results["new_composite"] = True
                    composition_results["composite_data"] = {
                        "name": composite_name,
                        "weights": normalized_weights,
                        "selected_strategies": selected_for_composition,
                    }
                    st.success(
                        f"✅ Composite strategy '{composite_name}' created successfully!"
                    )
                else:
                    st.error("Please provide a name and ensure total weight > 0")

        return composition_results

    @staticmethod
    def render_existing_composite_strategies(
        composite_strategies: dict[str, Any],
        session_state_key: str = "composite_strategies",
    ) -> list[str]:
        """Render existing composite strategies section."""

        removed_strategies = []

        if composite_strategies:
            st.subheader("📊 Existing Composite Strategies")

            for name, info in composite_strategies.items():
                with st.expander(f"📈 {name}", expanded=True):
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.write("**Composition:**")
                        weights_df = pd.DataFrame(
                            [
                                {
                                    "Strategy": k,
                                    "Weight": f"{v:.3f}",
                                    "Percentage": f"{v * 100:.1f}%",
                                }
                                for k, v in info.get("weights", {}).items()
                            ]
                        )
                        st.dataframe(weights_df, use_container_width=True)

                    with col2:
                        st.write("**Actions:**")
                        if st.button(f"Remove {name}", key=f"remove_{name}"):
                            removed_strategies.append(name)

                        # Export data
                        if "data" in info:
                            csv_data = info["data"].to_csv()
                            st.download_button(
                                label="💾 Download CSV",
                                data=csv_data,
                                file_name=f"{name}.csv",
                                mime="text/csv",
                                key=f"download_{name}",
                            )

        return removed_strategies

    @staticmethod
    def render_strategy_selector(
        available_strategies: list[str],
        key: str = "strategy_selector",
        label: str = "Select Strategies to Analyze:",
        help_text: str = "Select one or more strategies to compare",
    ) -> list[str]:
        """Render strategy selector."""

        return st.multiselect(
            label,
            available_strategies,
            default=[available_strategies[0]] if available_strategies else [],
            help=help_text,
            key=key,
        )

    @staticmethod
    def render_tearsheet_section(
        tearsheet_generator: TearsheetGenerator,
        available_strategies: list[str],
        strategy_data: dict[str, pd.Series],
        composite_strategies: dict[str, Any],
    ) -> str | None:
        """Render tearsheet generation section."""

        st.write("Generate comprehensive QuantStats tearsheet reports")

        # Strategy selection for tearsheet
        tearsheet_strategy = st.selectbox(
            "Select strategy for tearsheet:",
            available_strategies,
            key="tearsheet_selector",
            help="Choose a strategy to generate a comprehensive tearsheet report",
        )

        if tearsheet_strategy:
            # Benchmark selection (optional)
            include_benchmark = st.checkbox("Include Benchmark Comparison", value=False)
            benchmark_strategy = None

            if include_benchmark:
                benchmark_options = [
                    s for s in available_strategies if s != tearsheet_strategy
                ]
                if benchmark_options:
                    benchmark_strategy = st.selectbox(
                        "Select benchmark strategy:",
                        benchmark_options,
                        key="benchmark_selector",
                    )

            # Additional options
            title = st.text_input(
                "Report Title (optional):",
                value=f"{tearsheet_strategy} Performance Report",
                key="report_title",
            )

            # Generate report button
            if st.button("🎯 Generate Tearsheet Report", type="primary"):
                return UIComponents._generate_tearsheet_report(
                    tearsheet_generator,
                    tearsheet_strategy,
                    strategy_data,
                    composite_strategies,
                    title,
                    benchmark_strategy,
                )

        return None

    @staticmethod
    def _generate_tearsheet_report(
        tearsheet_generator: TearsheetGenerator,
        strategy_name: str,
        strategy_data: dict[str, pd.Series],
        composite_strategies: dict[str, Any],
        title: str,
        benchmark_strategy: str | None,
    ) -> str:
        """Generate tearsheet report."""

        try:
            with st.spinner("Generating tearsheet report..."):
                # Get strategy data
                if strategy_name in composite_strategies:
                    returns = composite_strategies[strategy_name]["data"]
                    strategy_info = {
                        "type": "composite",
                        "weights": composite_strategies[strategy_name]["weights"],
                    }
                else:
                    returns = strategy_data[strategy_name]
                    strategy_info = {"type": "original"}

                # Get benchmark data if selected
                benchmark_returns = None
                if benchmark_strategy:
                    if benchmark_strategy in composite_strategies:
                        benchmark_returns = composite_strategies[benchmark_strategy][
                            "data"
                        ]
                    else:
                        benchmark_returns = strategy_data[benchmark_strategy]

                # Generate report
                html_content = tearsheet_generator.generate_html_tearsheet(
                    returns=returns,
                    benchmark=benchmark_returns,
                    strategy_name=strategy_name,
                    title=title,
                )

                # Display success message
                st.success("✅ Tearsheet report generated successfully!")

                # Download button
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{strategy_name}_tearsheet_{timestamp}.html"

                st.download_button(
                    label="📥 Download Tearsheet Report",
                    data=html_content,
                    file_name=filename,
                    mime="text/html",
                    key="download_tearsheet",
                )

                return html_content

        except Exception as e:
            st.error(f"❌ Error generating tearsheet: {str(e)}")
            st.info(
                "💡 Tip: Make sure the strategy has sufficient data points for analysis."
            )
            return ""

    @staticmethod
    def render_performance_metrics_table(
        strategy_analyses: dict[str, Any], title: str = "Performance Comparison"
    ) -> None:
        """Render performance metrics comparison table."""

        st.subheader(title)

        if not strategy_analyses:
            st.warning("No strategy analyses available")
            return

        # Create summary table
        summary_data = []
        for name, analysis in strategy_analyses.items():
            basic_metrics = analysis.get("summary", {}).get("basic_metrics", {})
            period_info = analysis.get("summary", {}).get("period_info", {})

            summary_data.append(
                {
                    "Strategy": name,
                    "Start Date": period_info.get("start_date", "N/A"),
                    "End Date": period_info.get("end_date", "N/A"),
                    "Data Points": period_info.get("data_points", 0),
                    "Total Return": f"{basic_metrics.get('total_return', 0):.2%}",
                    "APR": f"{basic_metrics.get('apr', 0):.2%}",
                    # "CAGR": f"{basic_metrics.get('cagr', 0):.2%}",
                    "Volatility": f"{basic_metrics.get('volatility', 0):.2%}",
                    "Max Drawdown": f"{basic_metrics.get('max_drawdown', 0):.2%}",
                    "Sharpe Ratio": f"{basic_metrics.get('sharpe_ratio', 0):.2f}",
                    "Sortino Ratio": f"{basic_metrics.get('sortino_ratio', 0):.2f}",
                    "Calmar Ratio": f"{basic_metrics.get('calmar_ratio', 0):.2f}",
                }
            )

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

    @staticmethod
    def render_error_message(message: str, error_type: str = "warning") -> None:
        """Render error message."""

        if error_type == "error":
            st.error(message)
        elif error_type == "warning":
            st.warning(message)
        else:
            st.info(message)

    @staticmethod
    def render_data_export_section(
        data_dict: dict[str, pd.Series], selected_strategies: list[str]
    ) -> None:
        """Render data export section."""

        with st.expander("Export Data"):
            if selected_strategies:
                # Prepare export data
                export_data = pd.DataFrame(
                    {
                        name: data
                        for name, data in data_dict.items()
                        if name in selected_strategies
                    }
                ).dropna()

                if not export_data.empty:
                    st.dataframe(export_data, use_container_width=True)

                    # Export options
                    csv_data = export_data.to_csv()
                    st.download_button(
                        label="💾 Download Selected Data as CSV",
                        data=csv_data,
                        file_name="selected_strategies.csv",
                        mime="text/csv",
                    )
                else:
                    st.warning("No data available for export")
            else:
                st.info("Select strategies to export data")
