"""Performance metrics calculation for strategy analysis."""

from typing import Any

import numpy as np
import pandas as pd
import quantstats as qs


class PerformanceMetrics:
    """Calculates various performance metrics for strategy analysis."""

    @staticmethod
    def calculate_basic_metrics(returns: pd.Series) -> dict[str, float]:
        """Calculate basic performance metrics."""
        clean_returns = returns.dropna()

        if len(clean_returns) == 0:
            return {}

        try:
            return {
                "total_return": float(qs.stats.comp(clean_returns)),
                "apr": float(
                    qs.stats.cagr(clean_returns, periods=365, compounded=False)
                ),
                "cagr": float(
                    qs.stats.cagr(clean_returns, periods=365, compounded=True)
                ),
                "volatility": float(qs.stats.volatility(clean_returns, periods=365)),
                "sharpe_ratio": float(qs.stats.sharpe(clean_returns, periods=365)),
                "sortino_ratio": float(qs.stats.sortino(clean_returns, periods=365)),
                "calmar_ratio": float(qs.stats.calmar(clean_returns)),
                "max_drawdown": float(qs.stats.max_drawdown(clean_returns)),
                "skewness": float(qs.stats.skew(clean_returns)),
                "kurtosis": float(qs.stats.kurtosis(clean_returns)),
                "var_95": float(qs.stats.var(clean_returns)),
                "cvar_95": float(qs.stats.cvar(clean_returns)),
            }
        except Exception:
            return {}

    @staticmethod
    def calculate_advanced_metrics(returns: pd.Series) -> dict[str, float]:
        """Calculate advanced performance metrics."""
        clean_returns = returns.dropna()

        if len(clean_returns) == 0:
            return {}

        try:
            return {
                "omega_ratio": float(qs.stats.omega(clean_returns)),
                "tail_ratio": float(qs.stats.tail_ratio(clean_returns)),
                "common_sense_ratio": float(qs.stats.common_sense_ratio(clean_returns)),
                "kelly_criterion": float(qs.stats.kelly_criterion(clean_returns)),
                "payoff_ratio": float(qs.stats.payoff_ratio(clean_returns)),
                "profit_factor": float(qs.stats.profit_factor(clean_returns)),
                "cpc_index": float(qs.stats.cpc_index(clean_returns)),
                "outlier_win_ratio": float(qs.stats.outlier_win_ratio(clean_returns)),
                "outlier_loss_ratio": float(qs.stats.outlier_loss_ratio(clean_returns)),
                "recovery_factor": float(qs.stats.recovery_factor(clean_returns)),
            }
        except Exception:
            return {}

    @staticmethod
    def calculate_risk_metrics(returns: pd.Series) -> dict[str, float]:
        """Calculate risk-specific metrics."""
        clean_returns = returns.dropna()

        if len(clean_returns) == 0:
            return {}

        try:
            return {
                "volatility": float(qs.stats.volatility(clean_returns, periods=365)),
                "max_drawdown": float(qs.stats.max_drawdown(clean_returns)),
                "var_95": float(qs.stats.var(clean_returns)),
                "cvar_95": float(qs.stats.cvar(clean_returns)),
                "downside_deviation": float(qs.stats.downside_deviation(clean_returns)),
                "max_drawdown_duration": int(
                    qs.stats.max_drawdown_duration(clean_returns)
                ),
                "avg_drawdown": float(qs.stats.avg_drawdown(clean_returns)),
                "avg_drawdown_duration": float(
                    qs.stats.avg_drawdown_duration(clean_returns)
                ),
            }
        except Exception:
            return {}

    @staticmethod
    def calculate_comparative_metrics(
        returns1: pd.Series, returns2: pd.Series
    ) -> dict[str, float]:
        """Calculate comparative metrics between two strategies."""
        clean_returns1 = returns1.dropna()
        clean_returns2 = returns2.dropna()

        if len(clean_returns1) == 0 or len(clean_returns2) == 0:
            return {}

        try:
            # Align series
            aligned_returns1, aligned_returns2 = clean_returns1.align(
                clean_returns2, join="inner"
            )

            if len(aligned_returns1) == 0:
                return {}

            correlation = float(aligned_returns1.corr(aligned_returns2))

            return {
                "correlation": correlation,
                "beta": float(qs.stats.beta(aligned_returns1, aligned_returns2)),
                "alpha": float(qs.stats.alpha(aligned_returns1, aligned_returns2)),
                "information_ratio": float(
                    qs.stats.information_ratio(aligned_returns1, aligned_returns2)
                ),
                "tracking_error": float(
                    qs.stats.tracking_error(aligned_returns1, aligned_returns2)
                ),
            }
        except Exception:
            return {}

    @staticmethod
    def calculate_rolling_metrics(
        returns: pd.Series, window: int = 252
    ) -> dict[str, pd.Series]:
        """Calculate rolling metrics."""
        clean_returns = returns.dropna()

        if len(clean_returns) < window:
            return {}

        try:
            return {
                "rolling_sharpe": clean_returns.rolling(window).apply(
                    lambda x: qs.stats.sharpe(x) if len(x) == window else np.nan
                ),
                "rolling_volatility": clean_returns.rolling(window).std()
                * np.sqrt(365),
                "rolling_max_drawdown": clean_returns.rolling(window).apply(
                    lambda x: qs.stats.max_drawdown(x) if len(x) == window else np.nan
                ),
                "rolling_sortino": clean_returns.rolling(window).apply(
                    lambda x: qs.stats.sortino(x, periods=365)
                    if len(x) == window
                    else np.nan
                ),
            }
        except Exception:
            return {}

    @staticmethod
    def calculate_monthly_returns(returns: pd.Series) -> pd.DataFrame:
        """Calculate monthly returns matrix."""
        clean_returns = returns.dropna()

        if len(clean_returns) == 0:
            return pd.DataFrame()

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
                monthly_data.append({"Year": ts.year, "Month": ts.month, "Return": ret})

            if not monthly_data:
                return pd.DataFrame()

            monthly_df = pd.DataFrame(monthly_data)
            return monthly_df.pivot(index="Year", columns="Month", values="Return")
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def calculate_drawdown_series(returns: pd.Series) -> pd.Series:
        """Calculate drawdown series."""
        clean_returns = returns.dropna()

        if len(clean_returns) == 0:
            return pd.Series()

        try:
            return qs.stats.to_drawdown_series(clean_returns)
        except Exception:
            return pd.Series()

    @staticmethod
    def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
        """Calculate cumulative returns."""
        clean_returns = returns.dropna()

        if len(clean_returns) == 0:
            return pd.Series()

        try:
            return (1 + clean_returns).cumprod()
        except Exception:
            return pd.Series()

    @staticmethod
    def get_performance_summary(returns: pd.Series) -> dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            "basic_metrics": PerformanceMetrics.calculate_basic_metrics(returns),
            "advanced_metrics": PerformanceMetrics.calculate_advanced_metrics(returns),
            "risk_metrics": PerformanceMetrics.calculate_risk_metrics(returns),
            "period_info": {
                "start_date": str(returns.index[0])[:10] if len(returns) > 0 else None,
                "end_date": str(returns.index[-1])[:10] if len(returns) > 0 else None,
                "data_points": len(returns.dropna()),
                "total_days": len(returns),
            },
        }

        return summary
