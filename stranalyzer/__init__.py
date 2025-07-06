"""
Strategy Analyzer Package

A comprehensive toolkit for analyzing and comparing trading strategies with features including:
- Data loading and validation
- Portfolio composition and rebalancing
- Performance metrics calculation
- Interactive visualizations
- Tearsheet report generation
- Streamlit dashboard interface
"""

from typing import Any

import pandas as pd

from .analysis import PerformanceMetrics, StrategyAnalyzer
from .data import DataLoader, DataValidator
from .portfolio import PortfolioComposer
from .reports import TearsheetGenerator
from .ui import Dashboard, UIComponents
from .visualization import ChartGenerator

__all__ = [
    # Data handling
    "DataLoader",
    "DataValidator",
    # Portfolio management
    "PortfolioComposer",
    # Analysis
    "StrategyAnalyzer",
    "PerformanceMetrics",
    # Visualization
    "ChartGenerator",
    # Reports
    "TearsheetGenerator",
    # UI Components
    "Dashboard",
    "UIComponents",
]


# Convenience functions
def create_dashboard() -> Dashboard:
    """Create a new dashboard instance."""
    return Dashboard()


def run_dashboard() -> None:
    """Run the dashboard application."""
    dashboard = create_dashboard()
    dashboard.run()


def analyze_strategies(
    strategy_data: dict[str, pd.Series], selected_strategies: list[str] | None = None
) -> dict[str, Any]:
    """Convenience function to analyze strategies."""
    analyzer = StrategyAnalyzer()
    if selected_strategies is None:
        selected_strategies = list(strategy_data.keys())
    return analyzer.analyze_multiple_strategies(strategy_data, selected_strategies)
