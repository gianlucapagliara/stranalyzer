# Strategies Analyzer Dashboard

A comprehensive Streamlit dashboard for analyzing financial strategies with support for custom data uploads and portfolio composition.

## Features

- 📁 **File Upload**: Upload your own CSV files with custom strategies
- 🧪 **Sample Data**: Generate sample data for testing and experimentation
- 🏗️ **Portfolio Composition**: Create weighted combinations of strategies
- 📊 **Tearsheet Reports**: Generate comprehensive QuantStats reports
- 📈 **Interactive Analysis**: Visualize performance metrics and comparisons
- 🔄 **Data Management**: Easy data loading, clearing, and format validation

## Quick Start

### Getting Started with Data

1. **Upload Your Data**: Use the file upload section to upload CSV files
2. **Or Generate Sample Data**: Click "Generate Sample Data" to create test data
3. **Start Analyzing**: Select strategies and explore the analysis tools

## Data Format

The dashboard accepts CSV files with the following format:

### Single Strategy File
```csv
date,strategy_name
2023-01-01,0.012
2023-01-02,-0.005
2023-01-03,0.008
```

### Multiple Strategies File
```csv
date,strategy_1,strategy_2,strategy_3
2023-01-01,0.012,0.008,0.015
2023-01-02,-0.005,0.002,-0.003
2023-01-03,0.008,0.012,0.006
```

## Sample Data

The dashboard includes 5 sample strategies for testing:

- **Conservative Strategy**: Low volatility, steady growth
- **Balanced Strategy**: Medium volatility, higher growth
- **Aggressive Strategy**: High volatility, highest growth potential
- **Market Neutral**: Ultra-low volatility, minimal growth
- **Momentum Strategy**: Variable volatility with momentum patterns

## Features Guide

### 1. File Upload & Data Management
- **Multi-file upload**: Upload multiple CSV files simultaneously
- **Real-time validation**: Instant feedback on file format and data quality
- **Data preview**: View file contents before processing
- **Format detection**: Automatic handling of different CSV structures
- **Sample data generation**: Create realistic test data instantly

### 2. Portfolio Composition
- **Strategy selection**: Choose from uploaded/sample strategies
- **Custom weights**: Assign precise weights to each strategy
- **Auto-normalization**: Weights automatically normalized to sum to 1.0
- **Portfolio management**: Save, edit, and delete composite strategies
- **Performance tracking**: Full analysis available for composite strategies

### 3. Analysis Tools
- **Cumulative returns**: Interactive charts showing strategy performance
- **Drawdown analysis**: Visualize maximum drawdown periods
- **Rolling metrics**: Sharpe ratio and volatility over time
- **Return distributions**: Histogram and box plot analysis
- **Correlation matrix**: Strategy correlation heatmap
- **Monthly heatmaps**: Calendar-based performance visualization

### 4. Tearsheet Reports
- **QuantStats integration**: Professional-grade performance reports
- **Benchmark comparison**: Compare strategies against benchmarks
- **HTML export**: Download complete analysis reports
- **Customizable**: Choose between basic and comprehensive reports

### 5. Data Export
- **CSV export**: Download processed data for external analysis
- **Report export**: Save HTML tearsheet reports
- **Sample format**: Download properly formatted CSV templates

## User Interface

- **Expandable sections**: Clean, organized interface with collapsible panels
- **Real-time updates**: Instant feedback and data processing
- **Error handling**: Clear error messages and troubleshooting guidance
- **Responsive design**: Works on desktop and tablet devices
- **Progress indicators**: Visual feedback for long-running operations

## Data Requirements

- **Date column**: First column should contain dates (various formats supported)
- **Numeric columns**: Subsequent columns should contain returns or performance data
- **File format**: CSV files with proper headers
- **Missing data**: NaN values are automatically handled and excluded
- **File size**: Optimized for files up to 200MB

## Configuration

The dashboard can be configured via `.streamlit/config.toml`:

```toml
[server]
maxUploadSize = 200  # Maximum file size in MB
maxMessageSize = 200

[theme]
base = "light"
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "monospace"

[browser]
gatherUsageStats = false
```

## Performance Metrics

The dashboard calculates and displays:

- **Total Return**: Cumulative strategy performance
- **CAGR**: Compound Annual Growth Rate
- **Volatility**: Standard deviation of returns
- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Calmar Ratio**: CAGR divided by maximum drawdown
- **VaR/CVaR**: Value at Risk and Conditional Value at Risk
- **Skewness & Kurtosis**: Distribution characteristics

## Troubleshooting

### Common Issues

1. **File Upload Errors**
   - Ensure CSV format with date in first column
   - Check that data columns contain only numbers
   - Verify file size is under 200MB

2. **Date Parsing Issues**
   - Use standard date formats (YYYY-MM-DD, MM/DD/YYYY, etc.)
   - Ensure date column is properly formatted in source file
   - Check for missing or invalid date entries

3. **Performance Issues**
   - Limit analysis to essential strategies
   - Use sample data for testing complex features
   - Clear unused data regularly

4. **Memory Issues**
   - Upload smaller files or reduce date ranges
   - Clear all data and restart if needed
   - Generate sample data instead of using large files

### Getting Help

- Use the sample data feature to test functionality
- Download the sample CSV format for reference
- Check error messages for specific guidance
- Ensure all required packages are installed

## Usage

### Quick Start

```python
from stranalyzer import run_dashboard

# Launch the Streamlit dashboard
run_dashboard()
```

### Dashboard Launch

```
streamlit run stranalyzer/dashboard.py
```

### Programmatic Usage

```python
from stranalyzer import DataLoader, PortfolioComposer, StrategyAnalyzer

# Load data
loader = DataLoader()
data = loader.generate_sample_data()

# Create portfolio
composer = PortfolioComposer()
weights = {"Strategy1": 0.6, "Strategy2": 0.4}
result = composer.create_composite_strategy(data, weights, "MyPortfolio")

# Analyze performance
analyzer = StrategyAnalyzer()
analysis = analyzer.analyze_single_strategy(result["data"], "MyPortfolio")
```

## Development (uv)

- Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Create env and install deps: `uv sync`  (installs default groups incl. `dev`)
- Run tests: `uv run pytest -q`
- Lint/format: `uv run ruff check . && uv run black --check . && uv run isort --check-only .`
- Run dashboard: `uv run streamlit run stranalyzer/dashboard.py`

## License

This project is licensed under the MIT License. 