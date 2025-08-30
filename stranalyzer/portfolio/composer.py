"""Portfolio composition functionality for strategy analysis."""

from typing import Any

import numpy as np
import pandas as pd


class PortfolioComposer:
    """Handles creation and management of composite strategies."""

    def __init__(self):
        self.composite_strategies = {}

    def create_composite_strategy(
        self,
        base_strategies: dict[str, pd.Series],
        weights: dict[str, float],
        name: str,
        normalize_weights: bool = True,
        enable_rebalancing: bool = False,
        rebalancing_tolerance: float = 0.05,
        rebalancing_cost: float = 0.001,
        cost_on_rebalanced_amount: bool = True,
    ) -> dict[str, Any]:
        """Create a composite strategy from base strategies with given weights."""

        result = {"success": True, "data": None, "info": {}, "errors": []}

        # Validate inputs
        if not base_strategies:
            result["success"] = False
            result["errors"].append("No base strategies provided")
            return result

        if not weights:
            result["success"] = False
            result["errors"].append("No weights provided")
            return result

        if not name or name.strip() == "":
            result["success"] = False
            result["errors"].append("Strategy name cannot be empty")
            return result

        # Check if all strategies in weights exist in base_strategies
        missing_strategies = [s for s in weights.keys() if s not in base_strategies]
        if missing_strategies:
            result["success"] = False
            result["errors"].append(
                f"Strategies not found: {', '.join(missing_strategies)}"
            )
            return result

        # Check for negative weights
        negative_weights = [s for s, w in weights.items() if w < 0]
        if negative_weights:
            result["success"] = False
            result["errors"].append(
                f"Negative weights not allowed: {', '.join(negative_weights)}"
            )
            return result

        # Validate rebalancing parameters
        if enable_rebalancing:
            if not (0.001 <= rebalancing_tolerance <= 0.5):
                result["success"] = False
                result["errors"].append(
                    "Rebalancing tolerance must be between 0.1% and 50%"
                )
                return result

            if not (0.0 <= rebalancing_cost <= 0.1):
                result["success"] = False
                result["errors"].append("Rebalancing cost must be between 0% and 10%")
                return result

        # Normalize weights if requested
        total_weight = sum(weights.values())
        if total_weight == 0:
            result["success"] = False
            result["errors"].append("Total weight cannot be zero")
            return result

        if normalize_weights and total_weight != 1.0:
            weights = {k: v / total_weight for k, v in weights.items()}

        try:
            # Find common date range
            date_ranges = []
            for strategy_name in weights.keys():
                series = base_strategies[strategy_name].dropna()
                if len(series) > 0:
                    date_ranges.append((series.index[0], series.index[-1]))

            if not date_ranges:
                result["success"] = False
                result["errors"].append("No valid data in selected strategies")
                return result

            # Get common period
            common_start = max(start for start, _ in date_ranges)
            common_end = min(end for _, end in date_ranges)

            if common_start > common_end:
                result["success"] = False
                result["errors"].append(
                    "No overlapping period found between selected strategies"
                )
                return result

            # Create composite strategy with or without rebalancing
            if enable_rebalancing:
                composite_data, rebalancing_info = self._create_rebalanced_composite(
                    base_strategies,
                    weights,
                    common_start,
                    common_end,
                    rebalancing_tolerance,
                    rebalancing_cost,
                    cost_on_rebalanced_amount,
                )
                strategy_contributions = rebalancing_info["strategy_contributions"]
            else:
                # Create static portfolio (weights drift over time, no rebalancing)
                composite_data, static_info = self._create_static_composite(
                    base_strategies, weights, common_start, common_end
                )
                strategy_contributions = static_info["strategy_contributions"]
                rebalancing_info = static_info  # Store static info for later use

            # Store composite strategy
            strategy_info = {
                "data": composite_data,
                "weights": weights,
                "contributions": strategy_contributions,
                "created_at": pd.Timestamp.now(),
                "period": {"start": common_start, "end": common_end},
                "enable_rebalancing": enable_rebalancing,
            }

            if enable_rebalancing:
                strategy_info.update(
                    {
                        "rebalancing_tolerance": rebalancing_tolerance,
                        "rebalancing_cost": rebalancing_cost,
                        "cost_on_rebalanced_amount": cost_on_rebalanced_amount,
                        "rebalancing_info": rebalancing_info,
                    }
                )
            else:
                # Store static portfolio information
                strategy_info["static_info"] = rebalancing_info

            self.composite_strategies[name] = strategy_info

            result["data"] = composite_data
            result["info"] = {
                "weights": weights,
                "contributions": strategy_contributions,
                "period": f"{common_start.strftime('%Y-%m-%d')} to {common_end.strftime('%Y-%m-%d')}",
                "data_points": len(composite_data),
                "enable_rebalancing": enable_rebalancing,
            }

            if enable_rebalancing and rebalancing_info:
                result["info"]["rebalancing"] = {
                    "tolerance": rebalancing_tolerance,
                    "cost": rebalancing_cost,
                    "cost_on_rebalanced_amount": cost_on_rebalanced_amount,
                    "total_rebalancing_events": rebalancing_info.get(
                        "total_rebalancing_events", 0
                    ),
                    "total_rebalancing_cost": rebalancing_info.get(
                        "total_rebalancing_cost", 0.0
                    ),
                    "average_drift": rebalancing_info.get("average_drift", 0.0),
                }

            return result

        except Exception as e:
            result["success"] = False
            result["errors"].append(f"Error creating composite strategy: {str(e)}")
            return result

    def get_composite_strategy(self, name: str) -> dict[str, Any] | None:
        """Get a composite strategy by name."""
        return self.composite_strategies.get(name)

    def list_composite_strategies(self) -> list[str]:
        """List all composite strategy names."""
        return list(self.composite_strategies.keys())

    def remove_composite_strategy(self, name: str) -> bool:
        """Remove a composite strategy."""
        if name in self.composite_strategies:
            del self.composite_strategies[name]
            return True
        return False

    def get_strategy_composition(self, name: str) -> dict[str, Any] | None:
        """Get the composition details of a composite strategy."""
        if name not in self.composite_strategies:
            return None

        strategy_info = self.composite_strategies[name]
        return {
            "name": name,
            "weights": strategy_info["weights"],
            "contributions": strategy_info["contributions"],
            "created_at": strategy_info["created_at"],
            "period": strategy_info["period"],
            "data_points": len(strategy_info["data"]),
        }

    def export_composite_strategy(self, name: str) -> str | None:
        """Export a composite strategy to CSV format."""
        if name not in self.composite_strategies:
            return None

        strategy_data = self.composite_strategies[name]["data"]
        return strategy_data.to_csv()

    def optimize_weights(
        self,
        base_strategies: dict[str, pd.Series],
        selected_strategies: list[str],
        optimization_method: str = "equal_weight",
    ) -> dict[str, float]:
        """Optimize weights for selected strategies."""

        if optimization_method == "equal_weight":
            weight = 1.0 / len(selected_strategies)
            return {strategy: weight for strategy in selected_strategies}

        elif optimization_method == "inverse_volatility":
            # Calculate inverse volatility weights
            volatilities = {}
            for strategy in selected_strategies:
                if strategy in base_strategies:
                    vol = base_strategies[strategy].std()
                    volatilities[strategy] = (
                        vol if vol > 0 else 0.01
                    )  # Avoid division by zero

            inv_vol_sum = sum(1 / vol for vol in volatilities.values())
            return {
                strategy: (1 / vol) / inv_vol_sum
                for strategy, vol in volatilities.items()
            }

        elif optimization_method == "risk_parity":
            # Simplified risk parity (equal risk contribution)
            # This is a simplified version - full risk parity requires iterative optimization
            return self.optimize_weights(
                base_strategies, selected_strategies, "inverse_volatility"
            )

        else:
            # Default to equal weight
            return self.optimize_weights(
                base_strategies, selected_strategies, "equal_weight"
            )

    def validate_weights(self, weights: dict[str, float]) -> dict[str, Any]:
        """Validate strategy weights."""
        validation_result = {"is_valid": True, "warnings": [], "errors": []}

        # Check for negative weights
        negative_weights = [s for s, w in weights.items() if w < 0]
        if negative_weights:
            validation_result["is_valid"] = False
            validation_result["errors"].append(
                f"Negative weights not allowed: {', '.join(negative_weights)}"
            )

        # Check total weight
        total_weight = sum(weights.values())
        if total_weight == 0:
            validation_result["is_valid"] = False
            validation_result["errors"].append("Total weight cannot be zero")
        elif abs(total_weight - 1.0) > 0.01:
            validation_result["warnings"].append(
                f"Total weight is {total_weight:.3f}, consider normalizing to 1.0"
            )

        # Check for extreme weights
        max_weight = max(weights.values()) if weights else 0
        if max_weight > 0.8:
            validation_result["warnings"].append(
                f"Maximum weight is {max_weight:.1%}, consider diversification"
            )

        return validation_result

    def _create_rebalanced_composite(
        self,
        base_strategies: dict[str, pd.Series],
        target_weights: dict[str, float],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        tolerance: float,
        cost_per_rebalance: float,
        cost_on_rebalanced_amount: bool = True,
    ) -> tuple[pd.Series, dict[str, Any]]:
        """Create a composite strategy with periodic rebalancing."""

        # Get aligned data for all strategies
        aligned_strategies = {}
        for strategy_name in target_weights.keys():
            strategy_data = base_strategies[strategy_name].dropna()
            aligned_data = strategy_data.loc[start_date:end_date]
            aligned_strategies[strategy_name] = aligned_data

        # Get common dates
        common_dates = aligned_strategies[list(target_weights.keys())[0]].index
        for strategy_data in aligned_strategies.values():
            common_dates = common_dates.intersection(strategy_data.index)

        if len(common_dates) == 0:
            raise ValueError("No common dates found for rebalancing")

        common_dates = sorted(common_dates)

        # Initialize portfolio tracking
        portfolio_returns = pd.Series(index=common_dates, dtype=float)
        portfolio_weights = pd.DataFrame(
            index=common_dates, columns=list(target_weights.keys()), dtype=float
        )

        # Track rebalancing events
        rebalancing_events = []
        total_rebalancing_cost = 0.0
        drift_values = []

        # Initialize with target weights
        current_weights = target_weights.copy()
        portfolio_weights.iloc[0] = pd.Series(current_weights)

        # Track strategy values (starting with equal allocation)
        strategy_values = pd.DataFrame(
            index=common_dates, columns=list(target_weights.keys()), dtype=float
        )

        # Initialize strategy values based on target weights (start with $1 portfolio)
        for strategy_name in target_weights.keys():
            strategy_values.loc[common_dates[0], strategy_name] = target_weights[
                strategy_name
            ]

        # First day portfolio return
        first_day_return = sum(
            target_weights[strategy] * aligned_strategies[strategy].iloc[0]
            for strategy in target_weights.keys()
        )
        portfolio_returns.iloc[0] = first_day_return

        for i in range(1, len(common_dates)):
            date = common_dates[i]

            # Get today's strategy returns
            strategy_returns = {}
            for strategy_name in target_weights.keys():
                strategy_returns[strategy_name] = aligned_strategies[strategy_name].loc[
                    date
                ]

            # Update strategy values based on their individual performance
            # This simulates how each strategy component grows independently
            for strategy_name in target_weights.keys():
                prev_value = strategy_values.loc[common_dates[i - 1], strategy_name]
                strategy_return = strategy_returns[strategy_name]
                new_value = prev_value * (1 + strategy_return)
                strategy_values.loc[date, strategy_name] = new_value

            # Calculate total portfolio value and current weights
            total_portfolio_value = strategy_values.loc[date].sum()
            new_weights = {}
            for strategy_name in target_weights.keys():
                new_weights[strategy_name] = (
                    strategy_values.loc[date, strategy_name] / total_portfolio_value
                )

            # Check if rebalancing is needed
            max_drift = max(
                abs(new_weights[strategy] - target_weights[strategy])
                for strategy in target_weights.keys()
            )

            drift_values.append(max_drift)

            if max_drift > tolerance:
                # Rebalancing needed - calculate cost based on method
                rebalanced_amount = 0.0

                if cost_on_rebalanced_amount:
                    # Calculate cost based on amount being rebalanced
                    for strategy_name in target_weights.keys():
                        current_value = strategy_values.loc[date, strategy_name]
                        target_value = (
                            target_weights[strategy_name] * total_portfolio_value
                        )
                        rebalanced_amount += abs(current_value - target_value)

                    # Cost applies to the total amount being moved (half of total rebalanced amount)
                    rebalancing_cost_amount = cost_per_rebalance * (
                        rebalanced_amount / 2
                    )
                else:
                    # Apply cost to entire portfolio value (original method)
                    rebalanced_amount = total_portfolio_value
                    rebalancing_cost_amount = cost_per_rebalance * total_portfolio_value

                total_rebalancing_cost += rebalancing_cost_amount

                # Apply rebalancing cost - reduce total portfolio value
                total_portfolio_value -= rebalancing_cost_amount

                # Reset strategy values to target proportions after cost
                for strategy_name in target_weights.keys():
                    strategy_values.loc[date, strategy_name] = (
                        target_weights[strategy_name] * total_portfolio_value
                    )

                # Reset weights to targets
                current_weights = target_weights.copy()

                rebalancing_events.append(
                    {
                        "date": date,
                        "drift": max_drift,
                        "cost": rebalancing_cost_amount,
                        "cost_method": "rebalanced_amount"
                        if cost_on_rebalanced_amount
                        else "total_portfolio",
                        "rebalanced_amount": rebalanced_amount,
                        "weights_before_rebalancing": new_weights.copy(),
                        "weights_after_rebalancing": current_weights.copy(),
                    }
                )
            else:
                # No rebalancing, keep drifted weights
                current_weights = new_weights

            # Calculate portfolio return for this period
            prev_total_value = strategy_values.loc[common_dates[i - 1]].sum()
            current_total_value = strategy_values.loc[date].sum()
            portfolio_returns.iloc[i] = (
                current_total_value - prev_total_value
            ) / prev_total_value

            # Store the current weights
            portfolio_weights.iloc[i] = pd.Series(current_weights)

        # Calculate final portfolio value
        final_portfolio_value = strategy_values.loc[common_dates[-1]].sum()

        # Prepare rebalancing info
        rebalancing_info = {
            "strategy_contributions": {
                strategy: {
                    "weight": target_weights[strategy],
                    "contribution": target_weights[strategy]
                    * aligned_strategies[strategy].mean(),
                }
                for strategy in target_weights.keys()
            },
            "total_rebalancing_events": len(rebalancing_events),
            "total_rebalancing_cost": total_rebalancing_cost,
            "average_drift": np.mean(drift_values) if drift_values else 0.0,
            "max_drift": max(drift_values) if drift_values else 0.0,
            "rebalancing_events": rebalancing_events,
            "portfolio_weights_over_time": portfolio_weights,
            "strategy_values_over_time": strategy_values,
            "final_portfolio_value": final_portfolio_value,
        }

        return portfolio_returns, rebalancing_info

    def _create_static_composite(
        self,
        base_strategies: dict[str, pd.Series],
        target_weights: dict[str, float],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> tuple[pd.Series, dict[str, Any]]:
        """Create a static composite strategy where weights drift over time (no rebalancing).

        This simulates a true 'buy and hold' portfolio where you invest initial amounts
        and never rebalance, allowing weights to drift based on relative performance.
        """

        # Get aligned data for all strategies
        aligned_strategies = {}
        for strategy_name in target_weights.keys():
            strategy_data = base_strategies[strategy_name].dropna()
            aligned_data = strategy_data.loc[start_date:end_date]
            aligned_strategies[strategy_name] = aligned_data

        # Get common dates
        common_dates = aligned_strategies[list(target_weights.keys())[0]].index
        for strategy_data in aligned_strategies.values():
            common_dates = common_dates.intersection(strategy_data.index)

        if len(common_dates) == 0:
            raise ValueError("No common dates found for static portfolio")

        common_dates = sorted(common_dates)

        # Track strategy values over time (buy and hold simulation)
        strategy_values = pd.DataFrame(
            index=common_dates, columns=list(target_weights.keys()), dtype=float
        )

        # Initialize strategy values based on target weights (start with $1 portfolio)
        for strategy_name in target_weights.keys():
            strategy_values.loc[common_dates[0], strategy_name] = target_weights[
                strategy_name
            ]

        # Simulate buy and hold - each strategy grows independently
        for i in range(1, len(common_dates)):
            date = common_dates[i]

            for strategy_name in target_weights.keys():
                prev_value = strategy_values.loc[common_dates[i - 1], strategy_name]
                strategy_return = aligned_strategies[strategy_name].loc[date]
                new_value = prev_value * (1 + strategy_return)
                strategy_values.loc[date, strategy_name] = new_value

        # Calculate portfolio returns from total value changes
        total_values = strategy_values.sum(axis=1)
        portfolio_returns = total_values.pct_change().fillna(0)

        # Calculate final strategy contributions
        final_total_value = total_values.iloc[-1]
        strategy_contributions = {}
        for strategy_name in target_weights.keys():
            final_value = strategy_values.loc[common_dates[-1], strategy_name]
            final_weight = final_value / final_total_value
            strategy_contributions[strategy_name] = {
                "initial_weight": target_weights[strategy_name],
                "final_weight": final_weight,
                "weight_drift": final_weight - target_weights[strategy_name],
                "contribution": target_weights[strategy_name]
                * aligned_strategies[strategy_name].mean(),
            }

        static_info = {
            "strategy_contributions": strategy_contributions,
            "strategy_values_over_time": strategy_values,
            "final_weights": {
                name: strategy_values.loc[common_dates[-1], name] / final_total_value
                for name in target_weights.keys()
            },
        }

        return portfolio_returns, static_info
