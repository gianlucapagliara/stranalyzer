"""Chart generation functionality for strategy analysis."""

from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class ChartGenerator:
    """Generates various charts and visualizations for strategy analysis."""

    @staticmethod
    def create_cumulative_returns_chart(
        strategies: dict[str, pd.Series],
        title: str = "Cumulative Returns Comparison",
        height: int = 600,
        composite_strategies: list[str] | None = None,
    ) -> go.Figure:
        """Create cumulative returns chart."""

        fig = go.Figure()

        for strategy_name, returns in strategies.items():
            clean_returns = returns.dropna()
            if len(clean_returns) == 0:
                continue

            cumulative_returns = (1 + clean_returns).cumprod() - 1

            # Different line style for composite strategies
            line_style = (
                dict(dash="dash")
                if composite_strategies and strategy_name in composite_strategies
                else dict()
            )

            fig.add_trace(
                go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns.values * 100.0,
                    mode="lines",
                    name=strategy_name,
                    line=line_style,
                    hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>",
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            hovermode="x unified",
            height=height,
        )

        return fig

    @staticmethod
    def create_drawdown_chart(
        strategies: dict[str, pd.Series],
        title: str = "Drawdown Comparison",
        height: int = 400,
        composite_strategies: list[str] | None = None,
    ) -> go.Figure:
        """Create drawdown chart."""

        fig = go.Figure()

        for i, (strategy_name, returns) in enumerate(strategies.items()):
            clean_returns = returns.dropna()
            if len(clean_returns) == 0:
                continue

            # Calculate drawdown
            cumulative = (1 + clean_returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max

            # Different line style for composite strategies
            line_style = (
                dict(dash="dash")
                if composite_strategies and strategy_name in composite_strategies
                else dict()
            )

            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values * 100.0,
                    mode="lines",
                    name=strategy_name,
                    line=line_style,
                    fill="tonexty" if i == 0 else None,
                    fillcolor="rgba(255,0,0,0.1)" if i == 0 else None,
                    hovertemplate="%{x}<br>%{y:.2%}<extra></extra>",
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            hovermode="x unified",
            height=height,
        )

        return fig

    @staticmethod
    def create_rolling_metrics_chart(
        strategies: dict[str, pd.Series],
        metric: Literal["sharpe", "volatility", "returns"] = "sharpe",
        window: int = 252,
        title: str | None = None,
        height: int = 400,
        composite_strategies: list[str] | None = None,
    ) -> go.Figure:
        """Create rolling metrics chart."""

        if title is None:
            title = f"Rolling {metric.title()} Ratio ({window} days)"

        metric_title = {
            "sharpe": "Sharpe Ratio",
            "volatility": "Volatility (%)",
            "returns": "Returns (%)",
        }

        fig = go.Figure()

        for strategy_name, returns in strategies.items():
            clean_returns = returns.dropna()
            if len(clean_returns) < window:
                continue

            if metric == "sharpe":
                rolling_metric = (
                    clean_returns.rolling(window).mean()
                    / clean_returns.rolling(window).std()
                    * np.sqrt(365)
                )
            elif metric == "volatility":
                rolling_metric = (
                    clean_returns.rolling(window).std() * np.sqrt(365) * 100.0
                )
            elif metric == "returns":
                rolling_metric = clean_returns.rolling(window).mean() * 365 * 100.0
            else:
                continue

            # Different line style for composite strategies
            line_style = (
                dict(dash="dash")
                if composite_strategies and strategy_name in composite_strategies
                else dict()
            )

            fig.add_trace(
                go.Scatter(
                    x=rolling_metric.index,
                    y=rolling_metric.values,
                    mode="lines",
                    name=strategy_name,
                    line=line_style,
                    hovertemplate="%{x}<br>%{y:.2f}<extra></extra>",
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title=metric.title(),
            hovermode="x unified",
            height=height,
        )

        return fig

    @staticmethod
    def create_return_distribution_chart(
        strategies: dict[str, pd.Series],
        chart_type: str = "histogram",
        title: str = "Return Distribution",
        height: int = 400,
    ) -> go.Figure:
        """Create return distribution chart."""

        if chart_type == "histogram" and len(strategies) == 1:
            # Single strategy histogram
            strategy_name, returns = next(iter(strategies.items()))
            clean_returns = returns.dropna()

            fig = px.histogram(
                x=clean_returns.values,
                nbins=50,
                title=f"{title} - {strategy_name}",
                labels={"x": "Returns (%)", "y": "Frequency"},
            )

        elif chart_type == "box" or len(strategies) > 1:
            # Multiple strategies box plot
            dist_data = []
            for strategy_name, returns in strategies.items():
                clean_returns = returns.dropna()
                dist_data.extend(
                    [(strategy_name, ret * 100.0) for ret in clean_returns.values]
                )

            if dist_data:
                dist_df = pd.DataFrame(dist_data, columns=["Strategy", "Returns"])
                fig = px.box(
                    dist_df,
                    x="Strategy",
                    y="Returns",
                    title=title,
                    hover_data=["Returns"],
                )
            else:
                fig = go.Figure()
        else:
            fig = go.Figure()

        fig.update_layout(height=height)
        return fig

    @staticmethod
    def create_correlation_matrix(
        strategies: dict[str, pd.Series],
        title: str = "Strategy Correlation Matrix",
        height: int = 500,
    ) -> go.Figure:
        """Create correlation matrix heatmap."""

        # Align all strategies
        aligned_data = pd.DataFrame(strategies).dropna()

        if aligned_data.empty:
            return go.Figure()

        corr_matrix = aligned_data.corr()

        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title=title,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
        )

        fig.update_layout(height=height)
        return fig

    @staticmethod
    def create_monthly_returns_heatmap(
        returns: pd.Series,
        strategy_name: str,
        title: str | None = None,
        height: int = 400,
    ) -> go.Figure:
        """Create monthly returns heatmap for a single strategy."""

        if title is None:
            title = f"Monthly Returns Heatmap - {strategy_name}"

        clean_returns = returns.dropna()
        if len(clean_returns) == 0:
            return go.Figure()

        try:
            # Calculate monthly returns
            monthly_returns = clean_returns.resample("ME").apply(
                lambda x: (1 + x).prod() - 1
            )

            # Create pivot table for heatmap
            monthly_data = []
            for date, ret in monthly_returns.items():
                if isinstance(date, pd.Timestamp):
                    ts = date
                else:
                    ts = pd.Timestamp(str(date))
                monthly_data.append(
                    {"Year": ts.year, "Month": ts.month, "Return": ret * 100.0}
                )

            if not monthly_data:
                return go.Figure()

            monthly_df = pd.DataFrame(monthly_data)
            monthly_pivot = monthly_df.pivot(
                index="Year", columns="Month", values="Return"
            )

            # Month names for better readability
            month_names = [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]

            # Only rename columns that exist
            existing_months = [
                col
                for col in monthly_pivot.columns
                if isinstance(col, int) and col <= 12
            ]

            if existing_months:
                monthly_pivot.columns = [
                    month_names[int(i) - 1] for i in existing_months
                ]

            fig = px.imshow(
                monthly_pivot,
                text_auto=True,
                aspect="auto",
                title=title,
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
                labels={"x": "Month", "y": "Year", "color": "Return (%)"},
            )

            fig.update_layout(height=height)
            return fig

        except Exception:
            return go.Figure()

    @staticmethod
    def create_risk_return_scatter(
        strategies: dict[str, pd.Series],
        title: str = "Risk-Return Analysis",
        height: int = 500,
        composite_strategies: list[str] | None = None,
    ) -> go.Figure:
        """Create risk-return scatter plot."""

        scatter_data = []

        for strategy_name, returns in strategies.items():
            clean_returns = returns.dropna()
            if len(clean_returns) == 0:
                continue

            annual_return = clean_returns.mean() * 365
            annual_volatility = clean_returns.std() * np.sqrt(365)

            scatter_data.append(
                {
                    "Strategy": strategy_name,
                    "Annual Return": annual_return,
                    "Annual Volatility": annual_volatility,
                    "Type": (
                        "Composite"
                        if composite_strategies
                        and strategy_name in composite_strategies
                        else "Original"
                    ),
                }
            )

        if not scatter_data:
            return go.Figure()

        scatter_df = pd.DataFrame(scatter_data)

        fig = px.scatter(
            scatter_df,
            x="Annual Volatility",
            y="Annual Return",
            text="Strategy",
            color="Type",
            title=title,
            hover_data=["Strategy", "Annual Return", "Annual Volatility"],
        )

        fig.update_traces(textposition="top center")
        fig.update_layout(
            height=height,
            xaxis_title="Annual Volatility",
            yaxis_title="Annual Return",
        )

        return fig

    @staticmethod
    def create_performance_comparison_chart(
        strategies: dict[str, pd.Series],
        metrics: list[str] = ["sharpe_ratio", "sortino_ratio", "calmar_ratio"],
        title: str = "Performance Metrics Comparison",
        height: int = 500,
    ) -> go.Figure:
        """Create performance comparison chart."""

        from ..analysis.metrics import PerformanceMetrics

        comparison_data = []

        for strategy_name, returns in strategies.items():
            basic_metrics = PerformanceMetrics.calculate_basic_metrics(returns)

            for metric in metrics:
                if metric in basic_metrics:
                    comparison_data.append(
                        {
                            "Strategy": strategy_name,
                            "Metric": metric.replace("_", " ").title(),
                            "Value": basic_metrics[metric],
                        }
                    )

        if not comparison_data:
            return go.Figure()

        comparison_df = pd.DataFrame(comparison_data)

        fig = px.bar(
            comparison_df,
            x="Strategy",
            y="Value",
            color="Metric",
            title=title,
            barmode="group",
            hover_data=["Strategy", "Metric", "Value"],
        )

        fig.update_layout(height=height)
        return fig
