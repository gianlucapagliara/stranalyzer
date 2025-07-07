"""Strategy analysis orchestration."""

from typing import Any

import pandas as pd

from .metrics import PerformanceMetrics


class StrategyAnalyzer:
    """Orchestrates comprehensive strategy analysis."""

    def __init__(self):
        self.metrics_calculator = PerformanceMetrics()

    def analyze_single_strategy(
        self, returns: pd.Series, strategy_name: str
    ) -> dict[str, Any]:
        """Perform comprehensive analysis of a single strategy."""

        analysis = {
            "strategy_name": strategy_name,
            "summary": self.metrics_calculator.get_performance_summary(returns),
            "cumulative_returns": self.metrics_calculator.calculate_cumulative_returns(
                returns
            ),
            "drawdown_series": self.metrics_calculator.calculate_drawdown_series(
                returns
            ),
            "monthly_returns": self.metrics_calculator.calculate_monthly_returns(
                returns
            ),
            "rolling_metrics": self.metrics_calculator.calculate_rolling_metrics(
                returns
            ),
            "return_distribution": self._analyze_return_distribution(returns),
        }

        return analysis

    def analyze_multiple_strategies(
        self, strategies: dict[str, pd.Series], selected_strategies: list[str]
    ) -> dict[str, Any]:
        """Analyze and compare multiple strategies."""

        # Filter selected strategies
        filtered_strategies = {
            name: data
            for name, data in strategies.items()
            if name in selected_strategies
        }

        if not filtered_strategies:
            return {}

        # Individual analyses
        individual_analyses = {}
        for name, returns in filtered_strategies.items():
            individual_analyses[name] = self.analyze_single_strategy(returns, name)

        # Comparative analysis
        comparative_analysis = self._perform_comparative_analysis(filtered_strategies)

        # Combined data for visualization
        combined_data = self._prepare_combined_data(filtered_strategies)

        return {
            "individual_analyses": individual_analyses,
            "comparative_analysis": comparative_analysis,
            "combined_data": combined_data,
            "summary_table": self._create_summary_table(individual_analyses),
        }

    def _analyze_return_distribution(self, returns: pd.Series) -> dict[str, Any]:
        """Analyze return distribution characteristics."""
        clean_returns = returns.dropna()

        if len(clean_returns) == 0:
            return {}

        try:
            # Calculate percentiles separately to avoid nested dict issues
            percentiles = {
                "percentile_1": float(clean_returns.quantile(0.01)),
                "percentile_5": float(clean_returns.quantile(0.05)),
                "percentile_25": float(clean_returns.quantile(0.25)),
                "percentile_50": float(clean_returns.quantile(0.50)),
                "percentile_75": float(clean_returns.quantile(0.75)),
                "percentile_95": float(clean_returns.quantile(0.95)),
                "percentile_99": float(clean_returns.quantile(0.99)),
            }

            base_stats = {
                "mean": float(clean_returns.mean()),
                "std": float(clean_returns.std()),
                "skewness": float(clean_returns.skew()),
                "kurtosis": float(clean_returns.kurtosis()),
                "min": float(clean_returns.min()),
                "max": float(clean_returns.max()),
                "positive_returns": int((clean_returns > 0).sum()),
                "negative_returns": int((clean_returns < 0).sum()),
                "zero_returns": int((clean_returns == 0).sum()),
                "win_rate": float((clean_returns > 0).mean()),
            }

            # Combine base stats with flattened percentiles
            return {**base_stats, **percentiles}
        except Exception:
            return {}

    def _perform_comparative_analysis(
        self, strategies: dict[str, pd.Series]
    ) -> dict[str, Any]:
        """Perform comparative analysis between strategies."""

        if len(strategies) < 2:
            return {}

        strategy_names = list(strategies.keys())
        comparative_metrics = {}

        # Calculate pairwise comparisons
        for i, strategy1 in enumerate(strategy_names):
            for j, strategy2 in enumerate(strategy_names[i + 1 :], i + 1):
                pair_key = f"{strategy1}_vs_{strategy2}"
                comparative_metrics[pair_key] = (
                    self.metrics_calculator.calculate_comparative_metrics(
                        strategies[strategy1], strategies[strategy2]
                    )
                )

        # Calculate correlation matrix
        aligned_data = pd.DataFrame(strategies).dropna()
        correlation_matrix = (
            aligned_data.corr() if not aligned_data.empty else pd.DataFrame()
        )

        return {
            "pairwise_metrics": comparative_metrics,
            "correlation_matrix": correlation_matrix,
            "alignment_info": self._check_data_alignment(strategies),
        }

    def _prepare_combined_data(self, strategies: dict[str, pd.Series]) -> pd.DataFrame:
        """Prepare combined data for visualization."""
        try:
            return pd.DataFrame(strategies).dropna()
        except Exception:
            return pd.DataFrame()

    def _create_summary_table(
        self, individual_analyses: dict[str, Any]
    ) -> pd.DataFrame:
        """Create summary table for comparison."""
        summary_data = []

        for name, analysis in individual_analyses.items():
            basic_metrics = analysis["summary"]["basic_metrics"]
            period_info = analysis["summary"]["period_info"]

            summary_data.append(
                {
                    "Strategy": name,
                    "Start Date": period_info.get("start_date", "N/A"),
                    "End Date": period_info.get("end_date", "N/A"),
                    "Data Points": period_info.get("data_points", 0),
                    "Total Return": f"{basic_metrics.get('total_return', 0):.2%}",
                    "APR": f"{basic_metrics.get('apr', 0):.2%}",
                    "CAGR": f"{basic_metrics.get('cagr', 0):.2%}",
                    "Volatility": f"{basic_metrics.get('volatility', 0):.2%}",
                    "Sharpe Ratio": f"{basic_metrics.get('sharpe_ratio', 0):.2f}",
                    "Sortino Ratio": f"{basic_metrics.get('sortino_ratio', 0):.2f}",
                    "Max Drawdown": f"{basic_metrics.get('max_drawdown', 0):.2%}",
                    "Calmar Ratio": f"{basic_metrics.get('calmar_ratio', 0):.2f}",
                    "Skewness": f"{basic_metrics.get('skewness', 0):.2f}",
                    "Kurtosis": f"{basic_metrics.get('kurtosis', 0):.2f}",
                    "VaR (95%)": f"{basic_metrics.get('var_95', 0):.2%}",
                    "CVaR (95%)": f"{basic_metrics.get('cvar_95', 0):.2%}",
                }
            )

        return pd.DataFrame(summary_data)

    def _check_data_alignment(self, strategies: dict[str, pd.Series]) -> dict[str, Any]:
        """Check data alignment across strategies."""
        alignment_info = {"date_ranges": {}, "common_period": None, "data_points": {}}

        # Get date ranges
        for name, returns in strategies.items():
            clean_returns = returns.dropna()
            if len(clean_returns) > 0:
                alignment_info["date_ranges"][name] = {
                    "start": clean_returns.index[0],
                    "end": clean_returns.index[-1],
                }
                alignment_info["data_points"][name] = len(clean_returns)

        # Find common period
        if len(alignment_info["date_ranges"]) > 1:
            start_dates = [
                info["start"] for info in alignment_info["date_ranges"].values()
            ]
            end_dates = [info["end"] for info in alignment_info["date_ranges"].values()]

            common_start = max(start_dates)
            common_end = min(end_dates)

            if common_start <= common_end:
                alignment_info["common_period"] = {
                    "start": common_start,
                    "end": common_end,
                }

        return alignment_info

    def get_strategy_ranking(
        self, strategies: dict[str, pd.Series], ranking_metric: str = "sharpe_ratio"
    ) -> list[dict[str, Any]]:
        """Rank strategies based on specified metric."""

        rankings = []

        for name, returns in strategies.items():
            basic_metrics = self.metrics_calculator.calculate_basic_metrics(returns)

            if ranking_metric in basic_metrics:
                rankings.append(
                    {
                        "strategy": name,
                        "value": basic_metrics[ranking_metric],
                        "formatted_value": (
                            f"{basic_metrics[ranking_metric]:.2f}"
                            if ranking_metric
                            in ["sharpe_ratio", "sortino_ratio", "calmar_ratio"]
                            else f"{basic_metrics[ranking_metric]:.2%}"
                        ),
                    }
                )

        # Sort by metric value (descending for most metrics)
        reverse_sort = ranking_metric not in [
            "volatility",
            "max_drawdown",
            "var_95",
            "cvar_95",
        ]
        rankings.sort(key=lambda x: x["value"], reverse=reverse_sort)

        # Add rank
        for i, ranking in enumerate(rankings, 1):
            ranking["rank"] = i

        return rankings
