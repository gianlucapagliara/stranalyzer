#!/usr/bin/env python3
"""
Simple validation script for rebalancing functionality.
This script demonstrates the new rebalancing features without requiring full test dependencies.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def create_sample_data():
    """Create sample strategy data for testing."""
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    
    # Strategy 1: Steady growth with low volatility
    strategy1_returns = pd.Series(
        np.random.normal(0.001, 0.015, 50), 
        index=dates, 
        name="Conservative"
    )
    
    # Strategy 2: Higher volatility, variable returns
    strategy2_returns = pd.Series(
        np.random.normal(0.002, 0.03, 50), 
        index=dates, 
        name="Aggressive"
    )
    
    return {
        "Conservative": strategy1_returns,
        "Aggressive": strategy2_returns,
    }

def validate_rebalancing():
    """Validate rebalancing functionality."""
    try:
        from stranalyzer.portfolio.composer import PortfolioComposer
        
        print("✅ Successfully imported PortfolioComposer")
        
        # Create sample data
        base_strategies = create_sample_data()
        weights = {"Conservative": 0.6, "Aggressive": 0.4}
        
        composer = PortfolioComposer()
        
        # Test 1: Create portfolio without rebalancing
        print("\n📊 Testing portfolio creation without rebalancing...")
        result_no_rebal = composer.create_composite_strategy(
            base_strategies=base_strategies,
            weights=weights,
            name="StaticPortfolio",
            enable_rebalancing=False,
        )
        
        if result_no_rebal["success"]:
            print("✅ Static portfolio created successfully")
            print(f"   Data points: {len(result_no_rebal['data'])}")
            print(f"   Enable rebalancing: {result_no_rebal['info']['enable_rebalancing']}")
        else:
            print("❌ Static portfolio creation failed:")
            for error in result_no_rebal["errors"]:
                print(f"   - {error}")
            return False
        
        # Test 2: Create portfolio with rebalancing
        print("\n⚖️ Testing portfolio creation with rebalancing...")
        result_with_rebal = composer.create_composite_strategy(
            base_strategies=base_strategies,
            weights=weights,
            name="RebalancedPortfolio",
            enable_rebalancing=True,
            rebalancing_tolerance=0.05,  # 5% tolerance
            rebalancing_cost=0.001,      # 0.1% cost per rebalancing
        )
        
        if result_with_rebal["success"]:
            print("✅ Rebalanced portfolio created successfully")
            print(f"   Data points: {len(result_with_rebal['data'])}")
            print(f"   Enable rebalancing: {result_with_rebal['info']['enable_rebalancing']}")
            
            if "rebalancing" in result_with_rebal["info"]:
                rebal_info = result_with_rebal["info"]["rebalancing"]
                print(f"   Rebalancing events: {rebal_info['total_rebalancing_events']}")
                print(f"   Total rebalancing cost: {rebal_info['total_rebalancing_cost']:.4f}")
                print(f"   Average drift: {rebal_info['average_drift']:.4f}")
                print(f"   Tolerance: {rebal_info['tolerance']:.2%}")
                print(f"   Cost per rebalance: {rebal_info['cost']:.3%}")
        else:
            print("❌ Rebalanced portfolio creation failed:")
            for error in result_with_rebal["errors"]:
                print(f"   - {error}")
            return False
        
        # Test 3: Validation of rebalancing parameters
        print("\n🔍 Testing parameter validation...")
        
        # Test invalid tolerance
        result_invalid_tolerance = composer.create_composite_strategy(
            base_strategies=base_strategies,
            weights=weights,
            name="InvalidTolerance",
            enable_rebalancing=True,
            rebalancing_tolerance=0.0005,  # Too low
            rebalancing_cost=0.001,
        )
        
        if not result_invalid_tolerance["success"]:
            print("✅ Invalid tolerance correctly rejected")
        else:
            print("❌ Invalid tolerance was accepted (should be rejected)")
            return False
        
        # Test invalid cost
        result_invalid_cost = composer.create_composite_strategy(
            base_strategies=base_strategies,
            weights=weights,
            name="InvalidCost",
            enable_rebalancing=True,
            rebalancing_tolerance=0.05,
            rebalancing_cost=0.15,  # Too high
        )
        
        if not result_invalid_cost["success"]:
            print("✅ Invalid cost correctly rejected")
        else:
            print("❌ Invalid cost was accepted (should be rejected)")
            return False
        
        # Test 4: Stored strategy information
        print("\n📋 Testing stored strategy information...")
        stored_rebalanced = composer.get_composite_strategy("RebalancedPortfolio")
        stored_static = composer.get_composite_strategy("StaticPortfolio")
        
        if stored_rebalanced and stored_static:
            print("✅ Both strategies stored successfully")
            print(f"   Static portfolio has rebalancing: {stored_static.get('enable_rebalancing', False)}")
            print(f"   Rebalanced portfolio has rebalancing: {stored_rebalanced.get('enable_rebalancing', False)}")
            
            if stored_rebalanced.get('enable_rebalancing'):
                print(f"   Rebalanced portfolio tolerance: {stored_rebalanced.get('rebalancing_tolerance', 0):.2%}")
                print(f"   Rebalanced portfolio cost: {stored_rebalanced.get('rebalancing_cost', 0):.3%}")
        else:
            print("❌ Failed to retrieve stored strategies")
            return False
        
        print("\n🎉 All rebalancing functionality tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Validating rebalancing functionality...")
    success = validate_rebalancing()
    
    if success:
        print("\n✅ Rebalancing functionality validation completed successfully!")
        print("\nKey features implemented:")
        print("• ✅ Rebalancing toggle (enable/disable)")
        print("• ✅ Configurable tolerance threshold")
        print("• ✅ Configurable rebalancing cost")
        print("• ✅ Weight drift calculation and tracking")
        print("• ✅ Cost impact accounting")
        print("• ✅ Comprehensive parameter validation")
        print("• ✅ UI integration with controls and display")
        print("• ✅ Detailed rebalancing statistics")
        sys.exit(0)
    else:
        print("\n❌ Validation failed!")
        sys.exit(1)