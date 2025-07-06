"""Portfolio composition functionality for strategy analysis."""

from typing import Any

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

            # Create composite strategy
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

            # Store composite strategy
            self.composite_strategies[name] = {
                "data": composite_data,
                "weights": weights,
                "contributions": strategy_contributions,
                "created_at": pd.Timestamp.now(),
                "period": {"start": common_start, "end": common_end},
            }

            result["data"] = composite_data
            result["info"] = {
                "weights": weights,
                "contributions": strategy_contributions,
                "period": f"{common_start.strftime('%Y-%m-%d')} to {common_end.strftime('%Y-%m-%d')}",
                "data_points": len(composite_data),
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
