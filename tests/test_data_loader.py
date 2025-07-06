"""Tests for the DataLoader class."""

from unittest.mock import Mock

import pandas as pd

from stranalyzer.data.loader import DataLoader


class TestDataLoader:
    """Test cases for DataLoader class."""

    def test_init(self):
        """Test DataLoader initialization."""
        loader = DataLoader()
        assert hasattr(loader, "data_storage")
        assert isinstance(loader.data_storage, dict)
        assert len(loader.data_storage) == 0

    def test_generate_sample_data(self):
        """Test sample data generation."""
        loader = DataLoader()
        sample_data = loader.generate_sample_data()

        # Check that we get the expected strategies
        expected_strategies = [
            "Conservative_Strategy",
            "Balanced_Strategy",
            "Aggressive_Strategy",
            "Market_Neutral",
            "Momentum_Strategy",
        ]

        assert isinstance(sample_data, dict)
        assert len(sample_data) == 5

        for strategy in expected_strategies:
            assert strategy in sample_data
            assert isinstance(sample_data[strategy], pd.Series)
            assert len(sample_data[strategy]) > 0
            assert isinstance(sample_data[strategy].index, pd.DatetimeIndex)

    def test_sample_data_reproducibility(self):
        """Test that sample data generation is reproducible."""
        loader1 = DataLoader()
        loader2 = DataLoader()

        data1 = loader1.generate_sample_data()
        data2 = loader2.generate_sample_data()

        # Should be identical due to fixed random seed
        for strategy in data1.keys():
            pd.testing.assert_series_equal(data1[strategy], data2[strategy])

    def test_process_uploaded_file_valid_single_column(self, sample_csv_content):
        """Test processing a valid CSV file with single data column."""
        # Create a single column CSV
        csv_content = """date,returns
2023-01-01,0.012
2023-01-02,-0.005
2023-01-03,0.008"""

        # Mock uploaded file
        mock_file = Mock()
        mock_file.name = "test_strategy.csv"

        # Mock pandas read_csv to return our test data
        from unittest.mock import patch

        import pandas as pd

        test_df = pd.DataFrame(
            {
                "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "returns": [0.012, -0.005, 0.008],
            }
        )

        with patch("pandas.read_csv", return_value=test_df):
            loader = DataLoader()
            result = loader.process_uploaded_file(mock_file)

        assert result["success"] is True
        assert "test_strategy" in result["data"]
        assert isinstance(result["data"]["test_strategy"], pd.Series)
        assert len(result["data"]["test_strategy"]) == 3
        assert len(result["errors"]) == 0

    def test_process_uploaded_file_valid_multiple_columns(self):
        """Test processing a valid CSV file with multiple data columns."""
        # Mock uploaded file
        mock_file = Mock()
        mock_file.name = "multi_strategy.csv"

        test_df = pd.DataFrame(
            {
                "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "strategy_1": [0.012, -0.005, 0.008],
                "strategy_2": [0.008, 0.002, 0.012],
            }
        )

        from unittest.mock import patch

        with patch("pandas.read_csv", return_value=test_df):
            loader = DataLoader()
            result = loader.process_uploaded_file(mock_file)

        assert result["success"] is True
        assert "multi_strategy_strategy_1" in result["data"]
        assert "multi_strategy_strategy_2" in result["data"]
        assert len(result["errors"]) == 0

    def test_process_uploaded_file_invalid_single_column(self):
        """Test processing an invalid CSV file with only one column."""
        mock_file = Mock()
        mock_file.name = "invalid.csv"

        test_df = pd.DataFrame({"date": ["2023-01-01", "2023-01-02", "2023-01-03"]})

        from unittest.mock import patch

        with patch("pandas.read_csv", return_value=test_df):
            loader = DataLoader()
            result = loader.process_uploaded_file(mock_file)

        assert result["success"] is False
        assert len(result["errors"]) > 0
        assert "at least 2 columns" in result["errors"][0]

    def test_process_uploaded_file_invalid_dates(self):
        """Test processing a CSV file with invalid dates."""
        mock_file = Mock()
        mock_file.name = "invalid_dates.csv"

        test_df = pd.DataFrame(
            {
                "date": ["invalid-date", "2023-01-02", "2023-01-03"],
                "returns": [0.012, -0.005, 0.008],
            }
        )

        from unittest.mock import patch

        with patch("pandas.read_csv", return_value=test_df):
            # Mock pd.to_datetime to raise an exception
            with patch("pandas.to_datetime", side_effect=ValueError("Invalid date")):
                loader = DataLoader()
                result = loader.process_uploaded_file(mock_file)

        assert result["success"] is False
        assert len(result["errors"]) > 0
        assert "Could not parse dates" in result["errors"][0]

    def test_process_uploaded_file_exception_handling(self):
        """Test exception handling in file processing."""
        mock_file = Mock()
        mock_file.name = "error.csv"

        from unittest.mock import patch

        with patch("pandas.read_csv", side_effect=Exception("File read error")):
            loader = DataLoader()
            result = loader.process_uploaded_file(mock_file)

        assert result["success"] is False
        assert len(result["errors"]) > 0
        assert "Error processing file" in result["errors"][0]

    def test_get_sample_csv_format(self):
        """Test getting sample CSV format."""
        loader = DataLoader()
        csv_format = loader.get_sample_csv_format()

        assert isinstance(csv_format, str)
        assert "date,strategy_1,strategy_2,strategy_3" in csv_format
        assert "2023-01-01" in csv_format
        assert len(csv_format.split("\n")) == 6  # Header + 5 data rows

    def test_get_data_summary_empty(self):
        """Test data summary with empty data."""
        loader = DataLoader()
        summary = loader.get_data_summary({})

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 0

    def test_get_data_summary_with_data(self, sample_returns_data):
        """Test data summary with actual data."""
        loader = DataLoader()
        summary = loader.get_data_summary(sample_returns_data)

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == len(sample_returns_data)

        expected_columns = [
            "Strategy",
            "Type",
            "Data Points",
            "Start Date",
            "End Date",
            "Has Data",
        ]
        for col in expected_columns:
            assert col in summary.columns

        # Check that all strategies have data (type depends on how they're identified)
        for _, row in summary.iterrows():
            assert row["Has Data"] == "Yes"
            assert row["Data Points"] > 0

    def test_get_data_summary_sample_strategies(self):
        """Test data summary correctly identifies sample strategies."""
        loader = DataLoader()
        sample_data = loader.generate_sample_data()
        summary = loader.get_data_summary(sample_data)

        # All should be marked as "Sample"
        for _, row in summary.iterrows():
            assert row["Type"] == "Sample"

    def test_get_data_summary_mixed_strategies(self, sample_returns_data):
        """Test data summary with mixed sample and uploaded strategies."""
        loader = DataLoader()
        sample_data = loader.generate_sample_data()

        # Mix sample and uploaded data (rename uploaded to avoid conflicts with sample names)
        mixed_data = sample_data.copy()
        for i, (key, value) in enumerate(sample_returns_data.items()):
            mixed_data[f"Uploaded_{key}"] = value

        summary = loader.get_data_summary(mixed_data)

        # Check that types are correctly identified
        sample_count = sum(
            1 for _, row in summary.iterrows() if row["Type"] == "Sample"
        )
        uploaded_count = sum(
            1 for _, row in summary.iterrows() if row["Type"] == "Uploaded"
        )

        assert sample_count == len(sample_data)
        assert uploaded_count == len(sample_returns_data)
