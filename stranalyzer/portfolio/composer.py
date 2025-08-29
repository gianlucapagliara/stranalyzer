"""Portfolio composition functionality for strategy analysis."""

from typing import Any

import pandas as pd
import numpy as np


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
                result["errors"].append(
                    "Rebalancing cost must be between 0% and 10%"
                )
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
                    base_strategies, weights, common_start, common_end, 
                    rebalancing_tolerance, rebalancing_cost
                )
                strategy_contributions = rebalancing_info["strategy_contributions"]
            else:
                composite_data = None
                strategy_contributions = {}

                for strategy_name, weight in weights.items():
                    strategy_data = base_strategies[strategy_name].dropna()

                    # Align to common period
                    aligned_data = strategy_data.loc[common_start:common_end]

                    if composite_data is None:
                        composite_data = aligned_data * weight
                    else:
                        composite_data = composite_data.add(
                            aligned_data * weight, fill_value=0
                        )

                    strategy_contributions[strategy_name] = {
                        "weight": weight,
                        "contribution": (
                            weight * aligned_data.mean() if len(aligned_data) > 0 else 0
                        ),
                    }
                
                rebalancing_info = None

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
                strategy_info.update({
                    "rebalancing_tolerance": rebalancing_tolerance,
                    "rebalancing_cost": rebalancing_cost,
                    "rebalancing_info": rebalancing_info,
                })
            
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
                    "total_rebalancing_events": rebalancing_info.get("total_rebalancing_events", 0),
                    "total_rebalancing_cost": rebalancing_info.get("total_rebalancing_cost", 0.0),
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
        
        # Initialize portfolio
        portfolio_value = pd.Series(index=common_dates, dtype=float)
        portfolio_weights = pd.DataFrame(
            index=common_dates, 
            columns=list(target_weights.keys()), 
            dtype=float
        )
        
        # Track rebalancing events
        rebalancing_events = []
        total_rebalancing_cost = 0.0
        drift_values = []
        
        # Initialize with target weights
        current_weights = target_weights.copy()
        portfolio_weights.iloc[0] = pd.Series(current_weights)
        portfolio_value.iloc[0] = 1.0  # Start with unit value
        
        for i in range(1, len(common_dates)):
            date = common_dates[i]
            prev_date = common_dates[i-1]
            
            # Calculate returns for each strategy
            strategy_returns = {}
            for strategy_name in target_weights.keys():
                if prev_date in aligned_strategies[strategy_name].index and date in aligned_strategies[strategy_name].index:
                    prev_val = aligned_strategies[strategy_name].loc[prev_date]
                    curr_val = aligned_strategies[strategy_name].loc[date]
                    if prev_val != 0:
                        strategy_returns[strategy_name] = (curr_val - prev_val) / prev_val
                    else:
                        strategy_returns[strategy_name] = 0.0
                else:
                    strategy_returns[strategy_name] = 0.0
            
            # Update portfolio value and weights based on returns
            portfolio_return = sum(
                current_weights[strategy] * strategy_returns[strategy]
                for strategy in target_weights.keys()
            )
            
            new_portfolio_value = portfolio_value.iloc[i-1] * (1 + portfolio_return)
            
            # Update weights based on individual strategy performance
            new_weights = {}
            for strategy_name in target_weights.keys():
                strategy_return = strategy_returns[strategy_name]
                old_weight = current_weights[strategy_name]
                # Weight changes based on relative performance
                new_weights[strategy_name] = (
                    old_weight * (1 + strategy_return) / (1 + portfolio_return)
                    if portfolio_return != -1 else old_weight
                )
            
            # Check if rebalancing is needed
            max_drift = max(
                abs(new_weights[strategy] - target_weights[strategy])
                for strategy in target_weights.keys()
            )
            
            drift_values.append(max_drift)
            
            if max_drift > tolerance:
                # Rebalancing needed
                rebalancing_cost = cost_per_rebalance * new_portfolio_value
                total_rebalancing_cost += rebalancing_cost
                new_portfolio_value -= rebalancing_cost
                
                # Reset to target weights
                current_weights = target_weights.copy()
                
                rebalancing_events.append({
                    "date": date,
                    "drift": max_drift,
                    "cost": rebalancing_cost,
                    "portfolio_value_before": new_portfolio_value + rebalancing_cost,
                    "portfolio_value_after": new_portfolio_value,
                })
            else:
                # No rebalancing, keep drifted weights
                current_weights = new_weights
            
            portfolio_value.iloc[i] = new_portfolio_value
            portfolio_weights.iloc[i] = pd.Series(current_weights)
        
        # Calculate portfolio returns
        portfolio_returns = portfolio_value.pct_change().fillna(0)
        
        # Prepare rebalancing info
        rebalancing_info = {
            "strategy_contributions": {
                strategy: {
                    "weight": target_weights[strategy],
                    "contribution": target_weights[strategy] * aligned_strategies[strategy].mean(),
                }
                for strategy in target_weights.keys()
            },
            "total_rebalancing_events": len(rebalancing_events),
            "total_rebalancing_cost": total_rebalancing_cost,
            "average_drift": np.mean(drift_values) if drift_values else 0.0,
            "max_drift": max(drift_values) if drift_values else 0.0,
            "rebalancing_events": rebalancing_events,
            "final_portfolio_value": portfolio_value.iloc[-1],
            "portfolio_weights_over_time": portfolio_weights,
        }
        
        return portfolio_returns, rebalancing_info
