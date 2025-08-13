# Forecasting Electricity Congestion in NYISO Using Machine Learning

## Project Overview
This project provides tools for analyzing and forecasting electricity congestion and demand in the New York Independent System Operator (NYISO) region. The main analysis is implemented in the `NYCEnergyAnalyzer` class and can be executed as a Python module.

## Installation
1. Create a Python virtual environment (recommended).
2. Install the project in editable mode:
   ```bash
   pip install -e .
   ```

## Usage
Run the full analysis with:
```bash
python -m nyiso_ml.analyzer
```
This command loads the provided data files, trains forecasting models and prints a summary report.

## Repository Structure
- `src/nyiso_ml/analyzer.py`: Core analysis module containing the `NYCEnergyAnalyzer` class.
- `requirements.txt`: Python dependencies.
- `pyproject.toml`: Project metadata for packaging.
- `README.md`: Project documentation.
- Data CSV files: historical NYISO load and price data (already included).

## License
This project is provided as-is for educational purposes.
