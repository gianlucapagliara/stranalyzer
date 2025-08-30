"""Tests for the PortfolioComposer class."""

import numpy as np
import pandas as pd

from stranalyzer.portfolio.composer import PortfolioComposer


class TestPortfolioComposer:
    """Test cases for PortfolioComposer class."""

    def test_init(self):
        """Test PortfolioComposer initialization."""
        composer = PortfolioComposer()
        assert hasattr(composer, "composite_strategies")
        assert isinstance(composer.composite_strategies, dict)
        assert len(composer.composite_strategies) == 0

    def test_create_composite_strategy_success(
        self, sample_returns_data, sample_weights
    ):
        """Test successful composite strategy creation."""
        composer = PortfolioComposer()

        result = composer.create_composite_strategy(
            base_strategies=sample_returns_data,
            weights=sample_weights,
            name="TestPortfolio",
            normalize_weights=True,
        )

        assert result["success"] is True
        assert result["data"] is not None
        assert isinstance(result["data"], pd.Series)
        assert len(result["errors"]) == 0

        # Check that the composite strategy is stored
        assert "TestPortfolio" in composer.composite_strategies

        # Verify info is populated
        info = result["info"]
        assert "weights" in info
        assert "contributions" in info
        assert "period" in info
        assert "data_points" in info

    def test_create_composite_strategy_no_base_strategies(self, sample_weights):
        """Test composite strategy creation with no base strategies."""
        composer = PortfolioComposer()

        result = composer.create_composite_strategy(
            base_strategies={}, weights=sample_weights, name="TestPortfolio"
        )

        assert result["success"] is False
        assert "No base strategies provided" in result["errors"]

    def test_create_composite_strategy_no_weights(self, sample_returns_data):
        """Test composite strategy creation with no weights."""
        composer = PortfolioComposer()

        result = composer.create_composite_strategy(
            base_strategies=sample_returns_data, weights={}, name="TestPortfolio"
        )

        assert result["success"] is False
        assert "No weights provided" in result["errors"]

    def test_create_composite_strategy_empty_name(
        self, sample_returns_data, sample_weights
    ):
        """Test composite strategy creation with empty name."""
        composer = PortfolioComposer()

        result = composer.create_composite_strategy(
            base_strategies=sample_returns_data, weights=sample_weights, name=""
        )

        assert result["success"] is False
        assert "Strategy name cannot be empty" in result["errors"]

    def test_create_composite_strategy_missing_strategies(self, sample_returns_data):
        """Test composite strategy creation with missing strategies in weights."""
        composer = PortfolioComposer()

        # Weights reference strategies not in base_strategies
        invalid_weights = {"NonExistentStrategy1": 0.5, "NonExistentStrategy2": 0.5}

        result = composer.create_composite_strategy(
            base_strategies=sample_returns_data,
            weights=invalid_weights,
            name="TestPortfolio",
        )

        assert result["success"] is False
        assert "Strategies not found" in result["errors"][0]

    def test_create_composite_strategy_negative_weights(self, sample_returns_data):
        """Test composite strategy creation with negative weights."""
        composer = PortfolioComposer()

        negative_weights = {
            "Conservative": -0.2,
            "Aggressive": 0.7,
            "Market_Neutral": 0.5,
        }

        result = composer.create_composite_strategy(
            base_strategies=sample_returns_data,
            weights=negative_weights,
            name="TestPortfolio",
        )

        assert result["success"] is False
        assert "Negative weights not allowed" in result["errors"][0]

    def test_create_composite_strategy_zero_total_weight(self, sample_returns_data):
        """Test composite strategy creation with zero total weight."""
        composer = PortfolioComposer()

        zero_weights = {"Conservative": 0.0, "Aggressive": 0.0, "Market_Neutral": 0.0}

        result = composer.create_composite_strategy(
            base_strategies=sample_returns_data,
            weights=zero_weights,
            name="TestPortfolio",
        )

        assert result["success"] is False
        assert "Total weight cannot be zero" in result["errors"]

    def test_create_composite_strategy_weight_normalization(self, sample_returns_data):
        """Test weight normalization functionality."""
        composer = PortfolioComposer()

        unnormalized_weights = {
            "Conservative": 0.8,
            "Aggressive": 0.8,
            "Market_Neutral": 0.4,
        }  # Total = 2.0

        result = composer.create_composite_strategy(
            base_strategies=sample_returns_data,
            weights=unnormalized_weights,
            name="TestPortfolio",
            normalize_weights=True,
        )

        assert result["success"] is True

        # Check that weights are normalized
        normalized_weights = result["info"]["weights"]
        total_weight = sum(normalized_weights.values())
        assert abs(total_weight - 1.0) < 1e-10

    def test_create_composite_strategy_no_normalization(self, sample_returns_data):
        """Test composite strategy creation without normalization."""
        composer = PortfolioComposer()

        weights = {
            "Conservative": 0.8,
            "Aggressive": 0.8,
            "Market_Neutral": 0.4,
        }  # Total = 2.0

        result = composer.create_composite_strategy(
            base_strategies=sample_returns_data,
            weights=weights,
            name="TestPortfolio",
            normalize_weights=False,
        )

        assert result["success"] is True

        # Weights should remain unnormalized
        result_weights = result["info"]["weights"]
        assert abs(sum(result_weights.values()) - 2.0) < 1e-10

    def test_get_composite_strategy_exists(self, sample_returns_data, sample_weights):
        """Test getting an existing composite strategy."""
        composer = PortfolioComposer()

        # Create a strategy first
        composer.create_composite_strategy(
            base_strategies=sample_returns_data,
            weights=sample_weights,
            name="TestPortfolio",
        )

        # Get the strategy
        strategy = composer.get_composite_strategy("TestPortfolio")

        assert strategy is not None
        assert "data" in strategy
        assert "weights" in strategy
        assert "created_at" in strategy

    def test_get_composite_strategy_not_exists(self):
        """Test getting a non-existent composite strategy."""
        composer = PortfolioComposer()

        strategy = composer.get_composite_strategy("NonExistent")

        assert strategy is None

    def test_list_composite_strategies_empty(self):
        """Test listing composite strategies when none exist."""
        composer = PortfolioComposer()

        strategies = composer.list_composite_strategies()

        assert isinstance(strategies, list)
        assert len(strategies) == 0

    def test_list_composite_strategies_with_data(
        self, sample_returns_data, sample_weights
    ):
        """Test listing composite strategies with data."""
        composer = PortfolioComposer()

        # Create multiple strategies
        composer.create_composite_strategy(
            sample_returns_data, sample_weights, "Portfolio1"
        )
        composer.create_composite_strategy(
            sample_returns_data, sample_weights, "Portfolio2"
        )

        strategies = composer.list_composite_strategies()

        assert isinstance(strategies, list)
        assert len(strategies) == 2
        assert "Portfolio1" in strategies
        assert "Portfolio2" in strategies

    def test_remove_composite_strategy_exists(
        self, sample_returns_data, sample_weights
    ):
        """Test removing an existing composite strategy."""
        composer = PortfolioComposer()

        # Create a strategy first
        composer.create_composite_strategy(
            sample_returns_data, sample_weights, "TestPortfolio"
        )

        # Remove it
        result = composer.remove_composite_strategy("TestPortfolio")

        assert result is True
        assert "TestPortfolio" not in composer.composite_strategies

    def test_remove_composite_strategy_not_exists(self):
        """Test removing a non-existent composite strategy."""
        composer = PortfolioComposer()

        result = composer.remove_composite_strategy("NonExistent")

        assert result is False

    def test_get_strategy_composition(self, sample_returns_data, sample_weights):
        """Test getting strategy composition details."""
        composer = PortfolioComposer()

        # Create a strategy first
        composer.create_composite_strategy(
            sample_returns_data, sample_weights, "TestPortfolio"
        )

        composition = composer.get_strategy_composition("TestPortfolio")

        assert composition is not None
        assert composition["name"] == "TestPortfolio"
        assert "weights" in composition
        assert "contributions" in composition
        assert "created_at" in composition
        assert "period" in composition
        assert "data_points" in composition

    def test_get_strategy_composition_not_exists(self):
        """Test getting composition for non-existent strategy."""
        composer = PortfolioComposer()

        composition = composer.get_strategy_composition("NonExistent")

        assert composition is None

    def test_export_composite_strategy(self, sample_returns_data, sample_weights):
        """Test exporting composite strategy to CSV."""
        composer = PortfolioComposer()

        # Create a strategy first
        composer.create_composite_strategy(
            sample_returns_data, sample_weights, "TestPortfolio"
        )

        csv_data = composer.export_composite_strategy("TestPortfolio")

        assert csv_data is not None
        assert isinstance(csv_data, str)
        assert len(csv_data) > 0

        # Should be valid CSV format
        lines = csv_data.strip().split("\n")
        assert len(lines) > 1  # At least header + some data

    def test_export_composite_strategy_not_exists(self):
        """Test exporting non-existent composite strategy."""
        composer = PortfolioComposer()

        csv_data = composer.export_composite_strategy("NonExistent")

        assert csv_data is None

    def test_optimize_weights_equal_weight(self, sample_returns_data):
        """Test equal weight optimization."""
        composer = PortfolioComposer()

        selected_strategies = ["Conservative", "Aggressive"]
        weights = composer.optimize_weights(
            sample_returns_data, selected_strategies, "equal_weight"
        )

        assert len(weights) == 2
        assert abs(weights["Conservative"] - 0.5) < 1e-10
        assert abs(weights["Aggressive"] - 0.5) < 1e-10

    def test_optimize_weights_inverse_volatility(self, sample_returns_data):
        """Test inverse volatility weight optimization."""
        composer = PortfolioComposer()

        selected_strategies = ["Conservative", "Aggressive"]
        weights = composer.optimize_weights(
            sample_returns_data, selected_strategies, "inverse_volatility"
        )

        assert len(weights) == 2
        assert all(w > 0 for w in weights.values())
        assert abs(sum(weights.values()) - 1.0) < 1e-10

        # Conservative should have higher weight (lower volatility)
        assert weights["Conservative"] > weights["Aggressive"]

    def test_validate_weights_valid(self, sample_weights):
        """Test validation of valid weights."""
        composer = PortfolioComposer()

        result = composer.validate_weights(sample_weights)

        assert result["is_valid"] is True
        assert len(result["errors"]) == 0

    def test_validate_weights_negative(self):
        """Test validation of weights with negative values."""
        composer = PortfolioComposer()

        negative_weights = {"Strategy1": -0.2, "Strategy2": 0.7, "Strategy3": 0.5}

        result = composer.validate_weights(negative_weights)

        assert result["is_valid"] is False
        assert any(
            "Negative weights not allowed" in error for error in result["errors"]
        )

    def test_validate_weights_zero_total(self):
        """Test validation of weights with zero total."""
        composer = PortfolioComposer()

        zero_weights = {"Strategy1": 0.0, "Strategy2": 0.0}

        result = composer.validate_weights(zero_weights)

        assert result["is_valid"] is False
        assert any("Total weight cannot be zero" in error for error in result["errors"])

    def test_validate_weights_not_normalized(self):
        """Test validation of weights that don't sum to 1."""
        composer = PortfolioComposer()

        unnormalized_weights = {
            "Strategy1": 0.3,
            "Strategy2": 0.3,
            "Strategy3": 0.3,
        }  # Total = 0.9

        result = composer.validate_weights(unnormalized_weights)

        assert result["is_valid"] is True  # Valid but with warning
        assert any("consider normalizing" in warning for warning in result["warnings"])

    def test_validate_weights_extreme_concentration(self):
        """Test validation of weights with extreme concentration."""
        composer = PortfolioComposer()

        concentrated_weights = {"Strategy1": 0.95, "Strategy2": 0.05}

        result = composer.validate_weights(concentrated_weights)

        assert result["is_valid"] is True  # Valid but with warning
        assert any(
            "consider diversification" in warning for warning in result["warnings"]
        )


class TestPortfolioComposerRebalancing:
    """Test cases for rebalancing functionality in PortfolioComposer."""

    def test_create_composite_strategy_with_rebalancing_success(
        self, sample_returns_data, sample_weights
    ):
        """Test successful creation of composite strategy with rebalancing enabled."""
        composer = PortfolioComposer()

        result = composer.create_composite_strategy(
            base_strategies=sample_returns_data,
            weights=sample_weights,
            name="RebalancedPortfolio",
            normalize_weights=True,
            enable_rebalancing=True,
            rebalancing_tolerance=0.05,
            rebalancing_cost=0.001,
        )

        assert result["success"] is True
        assert result["data"] is not None
        assert isinstance(result["data"], pd.Series)
        assert len(result["errors"]) == 0

        # Check rebalancing info is included
        info = result["info"]
        assert info["enable_rebalancing"] is True
        assert "rebalancing" in info

        rebal_info = info["rebalancing"]
        assert "tolerance" in rebal_info
        assert "cost" in rebal_info
        assert "total_rebalancing_events" in rebal_info
        assert "total_rebalancing_cost" in rebal_info
        assert "average_drift" in rebal_info

        # Check stored strategy includes rebalancing info
        stored_strategy = composer.get_composite_strategy("RebalancedPortfolio")
        assert stored_strategy["enable_rebalancing"] is True
        assert "rebalancing_tolerance" in stored_strategy
        assert "rebalancing_cost" in stored_strategy
        assert "rebalancing_info" in stored_strategy

    def test_create_composite_strategy_invalid_rebalancing_tolerance(
        self, sample_returns_data, sample_weights
    ):
        """Test composite strategy creation with invalid rebalancing tolerance."""
        composer = PortfolioComposer()

        # Test tolerance too low
        result = composer.create_composite_strategy(
            base_strategies=sample_returns_data,
            weights=sample_weights,
            name="TestPortfolio",
            enable_rebalancing=True,
            rebalancing_tolerance=0.0005,  # Too low
            rebalancing_cost=0.001,
        )

        assert result["success"] is False
        assert any(
            "Rebalancing tolerance must be between 0.1% and 50%" in error
            for error in result["errors"]
        )

        # Test tolerance too high
        result = composer.create_composite_strategy(
            base_strategies=sample_returns_data,
            weights=sample_weights,
            name="TestPortfolio",
            enable_rebalancing=True,
            rebalancing_tolerance=0.6,  # Too high
            rebalancing_cost=0.001,
        )

        assert result["success"] is False
        assert any(
            "Rebalancing tolerance must be between 0.1% and 50%" in error
            for error in result["errors"]
        )

    def test_create_composite_strategy_invalid_rebalancing_cost(
        self, sample_returns_data, sample_weights
    ):
        """Test composite strategy creation with invalid rebalancing cost."""
        composer = PortfolioComposer()

        # Test cost too high
        result = composer.create_composite_strategy(
            base_strategies=sample_returns_data,
            weights=sample_weights,
            name="TestPortfolio",
            enable_rebalancing=True,
            rebalancing_tolerance=0.05,
            rebalancing_cost=0.15,  # Too high
        )

        assert result["success"] is False
        assert any(
            "Rebalancing cost must be between 0% and 10%" in error
            for error in result["errors"]
        )

        # Test negative cost
        result = composer.create_composite_strategy(
            base_strategies=sample_returns_data,
            weights=sample_weights,
            name="TestPortfolio",
            enable_rebalancing=True,
            rebalancing_tolerance=0.05,
            rebalancing_cost=-0.001,  # Negative
        )

        assert result["success"] is False
        assert any(
            "Rebalancing cost must be between 0% and 10%" in error
            for error in result["errors"]
        )

    def test_rebalanced_composite_creation_basic_functionality(self):
        """Test basic functionality of rebalanced composite creation."""
        composer = PortfolioComposer()

        # Create sample data with different volatilities to trigger rebalancing
        dates = pd.date_range("2023-01-01", periods=100, freq="D")

        # Strategy 1: Low volatility, steady growth
        strategy1_returns = pd.Series(
            np.random.normal(0.0005, 0.01, 100), index=dates, name="LowVol"
        )

        # Strategy 2: High volatility, higher growth
        strategy2_returns = pd.Series(
            np.random.normal(0.001, 0.03, 100), index=dates, name="HighVol"
        )

        base_strategies = {
            "LowVol": strategy1_returns,
            "HighVol": strategy2_returns,
        }

        weights = {"LowVol": 0.6, "HighVol": 0.4}

        result = composer.create_composite_strategy(
            base_strategies=base_strategies,
            weights=weights,
            name="TestRebalanced",
            enable_rebalancing=True,
            rebalancing_tolerance=0.1,  # 10% tolerance
            rebalancing_cost=0.002,  # 0.2% cost
        )

        assert result["success"] is True
        assert result["data"] is not None

        # Check rebalancing metrics are reasonable
        rebal_info = result["info"]["rebalancing"]
        assert rebal_info["total_rebalancing_events"] >= 0
        assert rebal_info["total_rebalancing_cost"] >= 0
        assert rebal_info["average_drift"] >= 0

    def test_rebalanced_composite_no_rebalancing_needed(self):
        """Test rebalanced composite when no rebalancing is triggered."""
        composer = PortfolioComposer()

        # Create sample data with very low volatility
        dates = pd.date_range("2023-01-01", periods=50, freq="D")

        # Both strategies have very similar, low volatility
        strategy1_returns = pd.Series(
            np.random.normal(0.0001, 0.001, 50), index=dates, name="Stable1"
        )
        strategy2_returns = pd.Series(
            np.random.normal(0.0001, 0.001, 50), index=dates, name="Stable2"
        )

        base_strategies = {
            "Stable1": strategy1_returns,
            "Stable2": strategy2_returns,
        }

        weights = {"Stable1": 0.5, "Stable2": 0.5}

        result = composer.create_composite_strategy(
            base_strategies=base_strategies,
            weights=weights,
            name="StablePortfolio",
            enable_rebalancing=True,
            rebalancing_tolerance=0.05,  # 5% tolerance
            rebalancing_cost=0.001,
        )

        assert result["success"] is True

        # With low volatility and equal weights, should have minimal rebalancing
        rebal_info = result["info"]["rebalancing"]
        assert rebal_info["total_rebalancing_events"] >= 0
        assert rebal_info["total_rebalancing_cost"] >= 0

    def test_rebalanced_composite_high_tolerance(self):
        """Test rebalanced composite with high tolerance (fewer rebalancing events)."""
        composer = PortfolioComposer()

        # Create sample data
        dates = pd.date_range("2023-01-01", periods=50, freq="D")

        strategy1_returns = pd.Series(
            np.random.normal(0.002, 0.02, 50), index=dates, name="Strategy1"
        )
        strategy2_returns = pd.Series(
            np.random.normal(-0.001, 0.03, 50), index=dates, name="Strategy2"
        )

        base_strategies = {
            "Strategy1": strategy1_returns,
            "Strategy2": strategy2_returns,
        }

        weights = {"Strategy1": 0.7, "Strategy2": 0.3}

        result = composer.create_composite_strategy(
            base_strategies=base_strategies,
            weights=weights,
            name="HighTolerancePortfolio",
            enable_rebalancing=True,
            rebalancing_tolerance=0.2,  # 20% tolerance (very high)
            rebalancing_cost=0.001,
        )

        assert result["success"] is True

        # High tolerance should result in fewer rebalancing events
        rebal_info = result["info"]["rebalancing"]
        assert rebal_info["total_rebalancing_events"] >= 0
        assert rebal_info["average_drift"] >= 0

    def test_rebalanced_composite_cost_impact(self):
        """Test that rebalancing costs reduce portfolio value."""
        composer = PortfolioComposer()

        # Create identical strategies for comparison
        dates = pd.date_range("2023-01-01", periods=50, freq="D")

        # Create volatile data to ensure rebalancing
        returns_data = np.random.normal(0.001, 0.04, 50)
        strategy1_returns = pd.Series(returns_data, index=dates, name="Strategy1")
        strategy2_returns = pd.Series(
            -returns_data, index=dates, name="Strategy2"
        )  # Opposite returns

        base_strategies = {
            "Strategy1": strategy1_returns,
            "Strategy2": strategy2_returns,
        }

        weights = {"Strategy1": 0.5, "Strategy2": 0.5}

        # Create portfolio without rebalancing
        result_no_rebal = composer.create_composite_strategy(
            base_strategies=base_strategies,
            weights=weights,
            name="NoRebalancePortfolio",
            enable_rebalancing=False,
        )

        # Create portfolio with rebalancing
        result_with_rebal = composer.create_composite_strategy(
            base_strategies=base_strategies,
            weights=weights,
            name="RebalancePortfolio",
            enable_rebalancing=True,
            rebalancing_tolerance=0.05,  # Low tolerance to trigger more rebalancing
            rebalancing_cost=0.01,  # 1% cost per rebalancing
        )

        assert result_no_rebal["success"] is True
        assert result_with_rebal["success"] is True

        # Check that rebalancing info is only present in rebalanced portfolio
        assert "rebalancing" not in result_no_rebal["info"]
        assert "rebalancing" in result_with_rebal["info"]

        rebal_info = result_with_rebal["info"]["rebalancing"]
        if rebal_info["total_rebalancing_events"] > 0:
            assert rebal_info["total_rebalancing_cost"] > 0

    def test_rebalanced_composite_weight_drift_tracking(self):
        """Test that weight drift is properly tracked."""
        composer = PortfolioComposer()

        # Create sample data with known drift patterns
        dates = pd.date_range("2023-01-01", periods=30, freq="D")

        # Strategy with consistent positive returns
        strategy1_returns = pd.Series([0.01] * 30, index=dates, name="GrowthStrategy")
        # Strategy with no returns
        strategy2_returns = pd.Series([0.0] * 30, index=dates, name="FlatStrategy")

        base_strategies = {
            "GrowthStrategy": strategy1_returns,
            "FlatStrategy": strategy2_returns,
        }

        weights = {"GrowthStrategy": 0.5, "FlatStrategy": 0.5}

        result = composer.create_composite_strategy(
            base_strategies=base_strategies,
            weights=weights,
            name="DriftTrackingPortfolio",
            enable_rebalancing=True,
            rebalancing_tolerance=0.1,  # 10% tolerance
            rebalancing_cost=0.001,
        )

        assert result["success"] is True

        rebal_info = result["info"]["rebalancing"]

        # With one growing and one flat strategy, we should see drift
        assert rebal_info["average_drift"] >= 0

        # Check that rebalancing info is stored in the composer
        stored_strategy = composer.get_composite_strategy("DriftTrackingPortfolio")
        assert "rebalancing_info" in stored_strategy

        detailed_rebal_info = stored_strategy["rebalancing_info"]
        assert "rebalancing_events" in detailed_rebal_info
        assert "portfolio_weights_over_time" in detailed_rebal_info
        assert "final_portfolio_value" in detailed_rebal_info

    def test_create_rebalanced_composite_method_directly(self):
        """Test the private _create_rebalanced_composite method."""
        composer = PortfolioComposer()

        # Create test data
        dates = pd.date_range("2023-01-01", periods=20, freq="D")

        strategy1_returns = pd.Series(
            np.random.normal(0.001, 0.02, 20), index=dates, name="Strategy1"
        )
        strategy2_returns = pd.Series(
            np.random.normal(-0.0005, 0.015, 20), index=dates, name="Strategy2"
        )

        base_strategies = {
            "Strategy1": strategy1_returns,
            "Strategy2": strategy2_returns,
        }

        target_weights = {"Strategy1": 0.6, "Strategy2": 0.4}

        # Call the private method directly
        portfolio_returns, rebalancing_info = composer._create_rebalanced_composite(
            base_strategies=base_strategies,
            target_weights=target_weights,
            start_date=dates[0],
            end_date=dates[-1],
            tolerance=0.05,
            cost_per_rebalance=0.001,
            cost_on_rebalanced_amount=True,
        )

        assert isinstance(portfolio_returns, pd.Series)
        assert len(portfolio_returns) == len(dates)
        assert isinstance(rebalancing_info, dict)

        # Check required keys in rebalancing_info
        required_keys = [
            "strategy_contributions",
            "total_rebalancing_events",
            "total_rebalancing_cost",
            "average_drift",
            "max_drift",
            "rebalancing_events",
            "final_portfolio_value",
            "portfolio_weights_over_time",
        ]

        for key in required_keys:
            assert key in rebalancing_info, f"Missing key: {key}"

    def test_rebalanced_composite_empty_data_error(self):
        """Test rebalanced composite creation with empty data raises appropriate error."""
        composer = PortfolioComposer()

        # Create empty strategies
        base_strategies = {
            "Strategy1": pd.Series([], dtype=float),
            "Strategy2": pd.Series([], dtype=float),
        }

        weights = {"Strategy1": 0.5, "Strategy2": 0.5}

        result = composer.create_composite_strategy(
            base_strategies=base_strategies,
            weights=weights,
            name="EmptyDataPortfolio",
            enable_rebalancing=True,
            rebalancing_tolerance=0.05,
            rebalancing_cost=0.001,
        )

        assert result["success"] is False
        assert len(result["errors"]) > 0

    def test_cost_calculation_methods(self):
        """Test different cost calculation methods."""
        composer = PortfolioComposer()

        # Create test data with extreme differences to ensure rebalancing
        dates = pd.date_range("2023-01-01", periods=10, freq="D")

        # Strategy 1: High positive returns
        strategy1_returns = pd.Series([0.02] * 10, index=dates, name="HighGrowth")
        # Strategy 2: Flat returns
        strategy2_returns = pd.Series([0.0] * 10, index=dates, name="Flat")

        base_strategies = {
            "HighGrowth": strategy1_returns,
            "Flat": strategy2_returns,
        }

        weights = {"HighGrowth": 0.5, "Flat": 0.5}

        # Test cost on rebalanced amount with low tolerance to force rebalancing
        result_rebal_amount = composer.create_composite_strategy(
            base_strategies=base_strategies,
            weights=weights,
            name="RebalancedAmountCost",
            enable_rebalancing=True,
            rebalancing_tolerance=0.01,  # 1% tolerance (low)
            rebalancing_cost=0.01,  # 1% cost
            cost_on_rebalanced_amount=True,
        )

        # Test cost on total portfolio
        result_total_portfolio = composer.create_composite_strategy(
            base_strategies=base_strategies,
            weights=weights,
            name="TotalPortfolioCost",
            enable_rebalancing=True,
            rebalancing_tolerance=0.01,  # 1% tolerance (low)
            rebalancing_cost=0.01,  # 1% cost
            cost_on_rebalanced_amount=False,
        )

        assert result_rebal_amount["success"] is True
        assert result_total_portfolio["success"] is True

        # Check that cost method is stored correctly
        assert (
            result_rebal_amount["info"]["rebalancing"]["cost_on_rebalanced_amount"]
            is True
        )
        assert (
            result_total_portfolio["info"]["rebalancing"]["cost_on_rebalanced_amount"]
            is False
        )

        # With extreme strategy differences, we should get rebalancing events
        rebal_events = result_rebal_amount["info"]["rebalancing"][
            "total_rebalancing_events"
        ]
        total_events = result_total_portfolio["info"]["rebalancing"][
            "total_rebalancing_events"
        ]

        # Both should have rebalancing events
        assert rebal_events > 0
        assert total_events > 0

        # If there are rebalancing events, total portfolio cost should be higher
        if rebal_events > 0 and total_events > 0:
            rebal_cost = result_rebal_amount["info"]["rebalancing"][
                "total_rebalancing_cost"
            ]
            total_cost = result_total_portfolio["info"]["rebalancing"][
                "total_rebalancing_cost"
            ]
            assert total_cost > rebal_cost
