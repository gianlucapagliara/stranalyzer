"""Data loading functionality for strategy analysis."""

from typing import Any

import numpy as np
import pandas as pd


class DataLoader:
    """Handles data loading operations including file uploads and sample data generation."""

    def __init__(self) -> None:
        self.data_storage: dict[str, pd.Series] = {}

    def generate_sample_data(self) -> dict[str, pd.Series]:
        """Generate sample data for testing purposes."""
        np.random.seed(42)  # For reproducible results

        # Generate date range
        date_range = pd.date_range(start="2023-01-01", end="2024-06-30", freq="D")

        # Generate sample strategies with different characteristics
        sample_strategies: dict[str, pd.Series] = {}

        # Strategy 1: Low volatility, steady growth
        returns_1 = np.random.normal(
            0.0005, 0.015, len(date_range)
        )  # 0.05% daily mean, 1.5% volatility
        sample_strategies["Conservative_Strategy"] = pd.Series(
            returns_1, index=date_range
        )

        # Strategy 2: Medium volatility, higher growth
        returns_2 = np.random.normal(
            0.001, 0.025, len(date_range)
        )  # 0.1% daily mean, 2.5% volatility
        sample_strategies["Balanced_Strategy"] = pd.Series(returns_2, index=date_range)

        # Strategy 3: High volatility, highest growth potential
        returns_3 = np.random.normal(
            0.0015, 0.035, len(date_range)
        )  # 0.15% daily mean, 3.5% volatility
        sample_strategies["Aggressive_Strategy"] = pd.Series(
            returns_3, index=date_range
        )

        # Strategy 4: Market-neutral strategy
        returns_4 = np.random.normal(
            0.0002, 0.008, len(date_range)
        )  # 0.02% daily mean, 0.8% volatility
        sample_strategies["Market_Neutral"] = pd.Series(returns_4, index=date_range)

        # Strategy 5: Momentum strategy (more volatile)
        returns_5 = []
        momentum: float = 0.0
        for i in range(len(date_range)):
            momentum = 0.95 * momentum + 0.05 * np.random.normal(0, 0.02)
            daily_return = 0.0008 + momentum + np.random.normal(0, 0.03)
            returns_5.append(daily_return)
        sample_strategies["Momentum_Strategy"] = pd.Series(returns_5, index=date_range)

        return sample_strategies

    def process_uploaded_file(self, uploaded_file: str) -> dict[str, Any]:
        """Process a single uploaded CSV file."""
        try:
            # Read the CSV file
            df: pd.DataFrame = pd.read_csv(uploaded_file)

            result = {
                "success": True,
                "data": {},
                "info": {"shape": df.shape, "columns": list(df.columns)},
                "errors": [],
            }

            # Validate and process the file
            if len(df.columns) < 2:
                result["success"] = False
                result["errors"].append(
                    "File must have at least 2 columns (date and one data column)"
                )
                return result

            # Assume first column is date
            date_column = df.columns[0]

            # Try to parse dates
            try:
                df[date_column] = pd.to_datetime(df[date_column])
            except Exception as e:
                result["success"] = False
                result["errors"].append(
                    f"Could not parse dates in column '{date_column}': {str(e)}"
                )
                return result

            # Set date as index
            df = df.set_index(date_column)

            # Process data columns (all columns except the date column)
            data_columns = [col for col in df.columns if col != date_column]

            # Convert data columns to numeric
            for col in data_columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                except Exception as e:
                    result["errors"].append(
                        f"Could not convert column '{col}' to numeric: {str(e)}"
                    )

            # Store processed data
            file_key = uploaded_file.name.replace(".csv", "")

            if len(data_columns) == 1:
                # Single column - store as series
                result["data"][file_key] = df[data_columns[0]].dropna()
            else:
                # Multiple columns - store each as separate series
                for col in data_columns:
                    series_key = f"{file_key}_{col}"
                    result["data"][series_key] = df[col].dropna()

            return result

        except Exception as e:
            return {
                "success": False,
                "data": {},
                "info": {},
                "errors": [f"Error processing file: {str(e)}"],
            }

    def get_sample_csv_format(self) -> str:
        """Get sample CSV format string."""
        return """date,strategy_1,strategy_2,strategy_3
2023-01-01,0.012,0.008,0.015
2023-01-02,-0.005,0.002,-0.003
2023-01-03,0.008,0.012,0.006
2023-01-04,0.015,-0.001,0.009
2023-01-05,-0.002,0.007,0.004"""

    def get_data_summary(self, data_dict: dict[str, pd.Series]) -> pd.DataFrame:
        """Generate summary information for loaded data."""
        summary_data = []

        sample_strategy_names = [
            "Conservative_Strategy",
            "Balanced_Strategy",
            "Aggressive_Strategy",
            "Market_Neutral",
            "Momentum_Strategy",
        ]

        for name, data in data_dict.items():
            data_type = "Sample" if name in sample_strategy_names else "Uploaded"
            summary_data.append(
                {
                    "Strategy": name,
                    "Type": data_type,
                    "Data Points": len(data),
                    "Start Date": str(data.index[0])[:10] if len(data) > 0 else "N/A",
                    "End Date": str(data.index[-1])[:10] if len(data) > 0 else "N/A",
                    "Has Data": "Yes" if len(data) > 0 else "No",
                }
            )

        return pd.DataFrame(summary_data)
