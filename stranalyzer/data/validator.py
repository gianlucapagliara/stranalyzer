"""Data validation functionality for strategy analysis."""

from typing import Any

import pandas as pd


class DataValidator:
    """Validates data integrity and provides data quality checks."""

    @staticmethod
    def validate_series(series: pd.Series) -> dict[str, Any]:
        """Validate a time series for analysis."""
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "statistics": {},
        }

        # Check for empty series
        if len(series) == 0:
            validation_result["is_valid"] = False
            validation_result["errors"].append("Series is empty")
            return validation_result

        # Check for sufficient data points
        if len(series) < 30:
            validation_result["warnings"].append(
                f"Series has only {len(series)} data points. Minimum 30 recommended for reliable analysis."
            )

        # Check for missing values
        missing_count = series.isna().sum()
        if missing_count > 0:
            missing_pct = (missing_count / len(series)) * 100
            validation_result["warnings"].append(
                f"Series contains {missing_count} missing values ({missing_pct:.1f}%)"
            )

        # Check for extreme values
        clean_series = series.dropna()
        if len(clean_series) > 0:
            q1 = clean_series.quantile(0.25)
            q3 = clean_series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = clean_series[
                (clean_series < lower_bound) | (clean_series > upper_bound)
            ]
            if len(outliers) > 0:
                outlier_pct = (len(outliers) / len(clean_series)) * 100
                validation_result["warnings"].append(
                    f"Series contains {len(outliers)} outliers ({outlier_pct:.1f}%)"
                )

        # Check for constant values
        if clean_series.nunique() == 1:
            validation_result["warnings"].append("Series contains only constant values")

        # Calculate basic statistics
        validation_result["statistics"] = {
            "count": len(series),
            "missing": missing_count,
            "mean": float(clean_series.mean()) if len(clean_series) > 0 else None,
            "std": float(clean_series.std()) if len(clean_series) > 0 else None,
            "min": float(clean_series.min()) if len(clean_series) > 0 else None,
            "max": float(clean_series.max()) if len(clean_series) > 0 else None,
            "skewness": float(clean_series.skew()) if len(clean_series) > 0 else None,
            "kurtosis": (
                float(clean_series.kurtosis()) if len(clean_series) > 0 else None
            ),
        }

        return validation_result

    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> dict[str, Any]:
        """Validate a DataFrame containing multiple strategies."""
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "column_validations": {},
        }

        # Check if DataFrame is empty
        if df.empty:
            validation_result["is_valid"] = False
            validation_result["errors"].append("DataFrame is empty")
            return validation_result

        # Validate each column
        for column in df.columns:
            col_validation = DataValidator.validate_series(df[column])
            validation_result["column_validations"][column] = col_validation

            if not col_validation["is_valid"]:
                validation_result["is_valid"] = False
                validation_result["errors"].extend(
                    [f"{column}: {error}" for error in col_validation["errors"]]
                )

        # Check for date index
        if not isinstance(df.index, pd.DatetimeIndex):
            validation_result["warnings"].append("Index is not a DatetimeIndex")

        # Check for index consistency
        if df.index.duplicated().any():
            validation_result["warnings"].append("Index contains duplicate values")

        return validation_result

    @staticmethod
    def check_data_alignment(data_dict: dict[str, pd.Series]) -> dict[str, Any]:
        """Check alignment of multiple time series."""
        alignment_result = {
            "is_aligned": True,
            "warnings": [],
            "date_ranges": {},
            "common_period": None,
        }

        if len(data_dict) == 0:
            return alignment_result

        # Get date ranges for each series
        for name, series in data_dict.items():
            if len(series) > 0:
                alignment_result["date_ranges"][name] = {
                    "start": series.index[0],
                    "end": series.index[-1],
                    "count": len(series),
                }

        # Find common period
        if len(alignment_result["date_ranges"]) > 1:
            start_dates = [
                info["start"] for info in alignment_result["date_ranges"].values()
            ]
            end_dates = [
                info["end"] for info in alignment_result["date_ranges"].values()
            ]

            common_start = max(start_dates)
            common_end = min(end_dates)

            if common_start <= common_end:
                alignment_result["common_period"] = {
                    "start": common_start,
                    "end": common_end,
                }
            else:
                alignment_result["is_aligned"] = False
                alignment_result["warnings"].append(
                    "No overlapping period found between series"
                )

        return alignment_result

    @staticmethod
    def suggest_data_improvements(validation_result: dict[str, Any]) -> list[str]:
        """Suggest improvements based on validation results."""
        suggestions = []

        if not validation_result["is_valid"]:
            suggestions.append("Fix data errors before proceeding with analysis")

        for warning in validation_result.get("warnings", []):
            if "missing values" in warning:
                suggestions.append(
                    "Consider filling missing values using forward fill or interpolation"
                )
            elif "outliers" in warning:
                suggestions.append(
                    "Review outliers - consider winsorization or removal if appropriate"
                )
            elif "only constant values" in warning:
                suggestions.append(
                    "Constant values provide no information - consider removing this series"
                )
            elif "data points" in warning:
                suggestions.append(
                    "Consider using a longer time period for more reliable analysis"
                )

        return suggestions
