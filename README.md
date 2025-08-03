# Forecasting-Electricity-Congestion-in-NYISO-Using-Machine-Learning
#Project Overview
Power grids must deliver electricity efficiently while avoiding costly congestion. In this project, we developed machine learning models to forecast regional electricity demand and identify areas of likely transmission congestion across New York State using publicly available data from the New York Independent System Operator (NYISO).

We used XGBoost regression to predict hourly electricity demand and XGBoost classification to predict congestion events. Our models captured complex nonlinear patterns in the data, including seasonal variation, population distribution, and geographic features affecting grid performance.

Data Sources
We used three publicly available datasets from NYISO:

Locational-Based Marginal Pricing (LBMP) – energy price, congestion, and losses (5-minute intervals)

Real-Time Integrated Actual Load – electricity demand in MW across NYISO zones

Weather data – hourly temperature and humidity from NYC weather stations

Time range: 2021–2024
Regions modeled: West, Genesee, Central, North, Capital, Hudson Valley, Millwood, NYC, Long Island

Machine Learning Approach: Demand Forecasting (Regression)
Model: XGBoost Regressor

Targets: Hourly electricity demand by region

Evaluation: R², MAPE, RMSE

Performance:

NYC: R² = 0.97, MAPE = 2%

Long Island: R² = 0.97

Other zones: R² between 0.88–0.92

Congestion Classification (Binary Classification)
Model: XGBoost Classifier

Target: Congestion (Yes/No) based on transmission bottlenecks

Evaluation: Accuracy, Confusion Matrix

Performance:

Accuracy ranged from 71% (NYC) to 98% (North)

Top predictors: Month, LBMP, Demand, Weather

