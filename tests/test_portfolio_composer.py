"""Tests for the PortfolioComposer class."""

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
