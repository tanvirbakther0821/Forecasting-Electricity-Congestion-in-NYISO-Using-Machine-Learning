# Forecasting Electricity Congestion in NYISO Using Machine Learning

## Overview
Power grids must deliver electricity efficiently while avoiding costly congestion. This project applies machine learning to predict regional electricity demand and identify areas of likely transmission congestion across New York State using publicly available data from the New York Independent System Operator (NYISO).

## Data
Public datasets from NYISO covering 2021–2024:
- **Locational Based Marginal Pricing (LBMP)** – energy price, congestion and losses (5‑minute intervals)
- **Real-Time Integrated Actual Load** – electricity demand in MW across NYISO zones
- **Weather data** – hourly temperature and humidity observations

## Installation
```bash
pip install -r requirements.txt
```

## Usage
The main workflow resides in `Forecasting-Electricity-Congestion-in-NYISO-Using-Machine-Learning.py`.

```bash
python Forecasting-Electricity-Congestion-in-NYISO-Using-Machine-Learning.py
```

The script expects the NYISO CSV files to be present in the working directory.

## Repository Structure
- `Forecasting-Electricity-Congestion-in-NYISO-Using-Machine-Learning.py` – primary analysis script
- `congestion_model.ipynb` – exploratory notebook
- `*.csv` – sample NYISO datasets (large files)
- `requirements.txt` – Python dependencies

## Results
Initial experiments achieved high accuracy in predicting demand (R² ≈ 0.97 for NYC and Long Island) and congestion (accuracy up to 98% in some zones).

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for improvements.
