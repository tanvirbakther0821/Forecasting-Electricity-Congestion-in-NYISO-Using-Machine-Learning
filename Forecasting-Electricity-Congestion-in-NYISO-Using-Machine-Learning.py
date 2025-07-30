#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NYC Energy Forecast - Complete Analysis Suite

Analyzes energy congestion patterns across:
- Central NY (central)
- Genese NY (genese)
- Long Island (longil)

Uses combined datasets with location identifiers for unified analysis.
Includes three separate models as per procedure:
1. Congestion Model (RTD Zonal Congestion as y)
2. Demand Model (TWI Actual Load as y) - NO pricing/congestion features
3. LBMP Pricing Model (RTD Zonal LBMP as y) - NO congestion features
"""


import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# Machine Learning imports
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost loaded successfully")
except Exception as e:
    print(f"⚠️ XGBoost error: {e}")
    print("Using RandomForest instead")
    XGBOOST_AVAILABLE = False

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from pandas.tseries.holiday import USFederalHolidayCalendar

class NYCEnergyAnalyzer:
    """
    Comprehensive energy analysis using merged NYC datasets.

    Features:
    - Loads and combines all merged NYC files
    - Creates unified dataset with location-specific weather
    - Builds three separate models per procedure:
      1. Congestion forecasting
      2. Demand forecasting (no pricing/congestion)
      3. Price forecasting (no congestion)
    - Provides comparative analysis across regions
    - Generates comprehensive visualizations
    """

    def __init__(self, data_directory="."):
        """
        Initialize the analyzer.

        Args:
            data_directory (str): Directory containing merged NYC CSV files
        """
        self.data_directory = data_directory
        self.lbmp_data = pd.DataFrame()
        self.load_data = pd.DataFrame()
        self.weather_data = pd.DataFrame()
        self.merged_dataset = pd.DataFrame()
        self.congestion_models = {}
        self.demand_models = {}
        self.price_models = {}
        self.location_data = {}

    def load_all_merged_files(self):
        """
        Load all merged NYC data files.
        Data processing/clean up:
        Turn all data into per hour
        Next then fill out table accordingly
        Using LMBP for the table and it label
        Each location has own table
        Need data set organized by dollars - congestion times actual load
        """
        print("📊 Loading All Merged NYC Data Files...")

        # Load LBMP files (all years)
        lbmp_files = [
            'OASIS_Real_Time_Dispatch_Zonal_LBMP nyc 2021.csv',
            'OASIS_Real_Time_Dispatch_Zonal_LBMP nyc 2022.csv',
            'OASIS_Real_Time_Dispatch_Zonal_LBMP nyc 2023.csv',
            'OASIS_Real_Time_Dispatch_Zonal_LBMP nyc 2024.csv'
        ]

        lbmp_dataframes = []
        for file in lbmp_files:
            file_path = os.path.join(self.data_directory, file)
            try:
                df = pd.read_csv(file_path)
                lbmp_dataframes.append(df)
                print(f"  ✅ Loaded {file}: {len(df)} records")
            except FileNotFoundError:
                print(f"  ⚠️ File not found: {file}")
            except Exception as e:
                print(f"  ❌ Error loading {file}: {e}")

        # Combine LBMP data
        if lbmp_dataframes:
            self.lbmp_data = pd.concat(lbmp_dataframes, ignore_index=True)
            print(f"  📊 Total LBMP records: {len(self.lbmp_data):,}")

        # Load Load files (all years)
        load_files = [
            'OASIS_Real_Time_Weighted_Integrated_Actual_Load nyc 2021.csv',
            'OASIS_Real_Time_Weighted_Integrated_Actual_Load nyc 2022.csv',
            'OASIS_Real_Time_Weighted_Integrated_Actual_Load nyc 2023.csv',
            'OASIS_Real_Time_Weighted_Integrated_Actual_Load nyc 2024.csv'
        ]

        load_dataframes = []
        for file in load_files:
            file_path = os.path.join(self.data_directory, file)
            try:
                df = pd.read_csv(file_path)
                load_dataframes.append(df)
                print(f"  ✅ Loaded {file}: {len(df)} records")
            except FileNotFoundError:
                print(f"  ⚠️ File not found: {file}")
            except Exception as e:
                print(f"  ❌ Error loading {file}: {e}")

        # Combine Load data
        if load_dataframes:
            self.load_data = pd.concat(load_dataframes, ignore_index=True)
            print(f"  ⚡ Total Load records: {len(self.load_data):,}")

        # Load Weather data
        weather_file = os.path.join(self.data_directory, 'NYC.csv')
        try:
            self.weather_data = pd.read_csv(weather_file)
            print(f"  ✅ Loaded NYC.csv: {len(self.weather_data)} weather records")
        except FileNotFoundError:
            print("  ⚠️ NYC.csv weather file not found!")
        except Exception as e:
            print(f"  ❌ Error loading NYC.csv: {e}")

    def show_data_summary(self):
        """Display summary of loaded data."""
        print("\n📋 DATA SUMMARY")
        print("=" * 40)

        if not self.lbmp_data.empty:
            print("📊 LBMP Data:")
            print(f"   Total records: {len(self.lbmp_data):,}")
            if 'location' in self.lbmp_data.columns:
                location_counts = self.lbmp_data['location'].value_counts()
                for loc, count in location_counts.items():
                    print(f"   {loc}: {count:,} records")

            # Date range
            if 'RTD End Time Stamp' in self.lbmp_data.columns:
                self.lbmp_data['RTD End Time Stamp'] = pd.to_datetime(self.lbmp_data['RTD End Time Stamp'])
                date_range = f"{self.lbmp_data['RTD End Time Stamp'].min().date()} to {self.lbmp_data['RTD End Time Stamp'].max().date()}"
                print(f"   Date range: {date_range}")

        if not self.load_data.empty:
            print("\n⚡ Load Data:")
            print(f"   Total records: {len(self.load_data):,}")
            if 'location' in self.load_data.columns:
                location_counts = self.load_data['location'].value_counts()
                for loc, count in location_counts.items():
                    print(f"   {loc}: {count:,} records")

        if not self.weather_data.empty:
            print("\n🌤️ Weather Data:")
            print(f"   Total records: {len(self.weather_data):,}")
            if 'location' in self.weather_data.columns:
                location_counts = self.weather_data['location'].value_counts()
                for loc, count in location_counts.items():
                    print(f"   {loc}: {count:,} records")

    def create_time_features(self, df, timestamp_col):
        """Create comprehensive time-based features."""
        df['date'] = df[timestamp_col].dt.date
        df['year'] = df[timestamp_col].dt.year
        df['month'] = df[timestamp_col].dt.month
        df['day'] = df[timestamp_col].dt.day
        df['hour'] = df[timestamp_col].dt.hour
        df['day_of_week'] = df[timestamp_col].dt.day_name()

        # Create binary day-of-week columns
        for day in ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']:
            df[day] = (df['day_of_week'] == day).astype(int)

        return df

    def add_holiday_features(self, df):
        """Add US holiday indicators."""
        try:
            cal = USFederalHolidayCalendar()
            start_date = df['date'].min()
            end_date = df['date'].max()
            holidays = cal.holidays(start=start_date, end=end_date)

            df['date_dt'] = pd.to_datetime(df['date'])
            df['holiday'] = df['date_dt'].isin(holidays).astype(int)
            df = df.drop('date_dt', axis=1)
        except Exception as e:
            print(f"⚠️ Could not create holiday features: {e}")
            df['holiday'] = 0

        return df

    def create_lagged_features_by_location(self, df):
        """Create lagged features separately for each location."""
        print("🔄 Creating location-specific lagged features...")

        location_dataframes = []

        for location in df['location'].unique():
            location_df = df[df['location'] == location].copy()
            location_df = location_df.sort_values('RTD End Time Stamp')

            # Create lagged features for this location
            # LBMP features
            location_df['LBMP_yesterday'] = location_df['RTD Zonal LBMP'].shift(24)
            location_df['LBMP_7_days_ago'] = location_df['RTD Zonal LBMP'].shift(168)
            location_df['LBMP_1_hour_ago'] = location_df['RTD Zonal LBMP'].shift(1)

            # Price rolling statistics
            location_df['price_24h_avg'] = location_df['RTD Zonal LBMP'].rolling(window=24, min_periods=1).mean()
            location_df['price_7d_avg'] = location_df['RTD Zonal LBMP'].rolling(window=168, min_periods=1).mean()
            location_df['price_24h_std'] = location_df['RTD Zonal LBMP'].rolling(window=24, min_periods=1).std()
            location_df['price_volatility_24h'] = location_df['RTD Zonal LBMP'].rolling(window=24, min_periods=1).std()
            location_df['price_range_24h'] = (location_df['RTD Zonal LBMP'].rolling(window=24, min_periods=1).max() -
                                            location_df['RTD Zonal LBMP'].rolling(window=24, min_periods=1).min())

            # Price momentum
            location_df['price_momentum_1h'] = location_df['RTD Zonal LBMP'] - location_df['LBMP_1_hour_ago']
            location_df['price_momentum_24h'] = location_df['RTD Zonal LBMP'] - location_df['LBMP_yesterday']

            # Congestion features
            location_df['congestion_yesterday'] = location_df['RTD Zonal Congestion'].shift(24)
            location_df['congestion_7_days_ago'] = location_df['RTD Zonal Congestion'].shift(168)

            location_dataframes.append(location_df)

        # Combine all locations back together
        combined_df = pd.concat(location_dataframes, ignore_index=True)
        combined_df = combined_df.sort_values(['location', 'RTD End Time Stamp'])

        print("✅ Lagged features created for all locations")
        return combined_df

    def create_demand_lagged_features_by_location(self, df):
        """Create lagged demand features separately for each location."""
        print("🔄 Creating location-specific lagged demand features...")

        location_dataframes = []

        for location in df['location'].unique():
            location_df = df[df['location'] == location].copy()
            location_df = location_df.sort_values('Eastern Date Hour')

            # Create lagged demand features for this location
            location_df['demand_yesterday'] = location_df['TWI Actual Load'].shift(24)
            location_df['demand_7_days_ago'] = location_df['TWI Actual Load'].shift(168)
            location_df['demand_1_hour_ago'] = location_df['TWI Actual Load'].shift(1)

            # Create rolling averages
            location_df['demand_24h_avg'] = location_df['TWI Actual Load'].rolling(window=24, min_periods=1).mean()
            location_df['demand_7d_avg'] = location_df['TWI Actual Load'].rolling(window=168, min_periods=1).mean()

            location_dataframes.append(location_df)

        # Combine all locations back together
        combined_df = pd.concat(location_dataframes, ignore_index=True)
        combined_df = combined_df.sort_values(['location', 'Eastern Date Hour'])

        print("✅ Lagged demand features created for all locations")
        return combined_df

    def merge_all_data(self):
        """Merge LBMP, Load, and Weather data with location-specific weather mapping."""
        print("\n🔗 Merging All NYC Data...")

        if self.lbmp_data.empty:
            print("❌ No LBMP data available!")
            return

        # Start with LBMP data
        print("  Processing LBMP data...")
        merged = self.lbmp_data.copy()
        merged['RTD End Time Stamp'] = pd.to_datetime(merged['RTD End Time Stamp'])
        merged = self.create_time_features(merged, 'RTD End Time Stamp')

        # Create lagged features by location
        merged = self.create_lagged_features_by_location(merged)

        # Group by location, year, month, day, hour for hourly averages
        print("  Creating hourly averages...")
        hourly_lbmp = merged.groupby(['location', 'year', 'month', 'day', 'hour']).agg({
            'RTD End Time Stamp': 'first',
            'RTD Zonal LBMP': 'mean',
            'RTD Zonal Losses': 'mean',
            'RTD Zonal Congestion': 'mean',
            'LBMP_yesterday': 'mean',
            'LBMP_7_days_ago': 'mean',
            'LBMP_1_hour_ago': 'mean',
            'price_24h_avg': 'mean',
            'price_7d_avg': 'mean',
            'price_24h_std': 'mean',
            'price_volatility_24h': 'mean',
            'price_range_24h': 'mean',
            'price_momentum_1h': 'mean',
            'price_momentum_24h': 'mean',
            'congestion_yesterday': 'mean',
            'congestion_7_days_ago': 'mean',
            'date': 'first',
            'day_of_week': 'first',
            'Sunday': 'first', 'Monday': 'first', 'Tuesday': 'first', 'Wednesday': 'first',
            'Thursday': 'first', 'Friday': 'first', 'Saturday': 'first'
        }).reset_index()

        # Add holiday features
        hourly_lbmp = self.add_holiday_features(hourly_lbmp)

        # Merge with Load data
        if not self.load_data.empty:
            print("  Merging Load data...")
            load_data = self.load_data.copy()
            load_data['Eastern Date Hour'] = pd.to_datetime(load_data['Eastern Date Hour'])
            load_data = self.create_time_features(load_data, 'Eastern Date Hour')

            # Create demand lagged features
            load_data = self.create_demand_lagged_features_by_location(load_data)

            # Group load data by location and hour
            hourly_load = load_data.groupby(['location', 'year', 'month', 'day', 'hour']).agg({
                'TWI Actual Load': 'mean',
                'demand_yesterday': 'mean',
                'demand_7_days_ago': 'mean',
                'demand_1_hour_ago': 'mean',
                'demand_24h_avg': 'mean',
                'demand_7d_avg': 'mean'
            }).reset_index()

            # Merge with LBMP data
            hourly_lbmp = pd.merge(hourly_lbmp, hourly_load,
                                 on=['location', 'year', 'month', 'day', 'hour'], how='left')
            print("    ✅ Load data merged")

        # Merge with Weather data (location-specific)
        if not self.weather_data.empty:
            print("  Merging Weather data...")
            weather_data = self.weather_data.copy()
            weather_data['valid'] = pd.to_datetime(weather_data['valid'])
            weather_data = self.create_time_features(weather_data, 'valid')

            # Group weather data by location and hour
            hourly_weather = weather_data.groupby(['location', 'year', 'month', 'day', 'hour']).agg({
                'tmpf': 'mean',
                'relh': 'mean'
            }).reset_index()

            # Rename weather columns for clarity
            hourly_weather = hourly_weather.rename(columns={'tmpf': 'temperature_F', 'relh': 'humidity_pct'})

            # Merge with main data
            hourly_lbmp = pd.merge(hourly_lbmp, hourly_weather,
                                 on=['location', 'year', 'month', 'day', 'hour'], how='left')
            print("    ✅ Weather data merged")

        # Create additional features
        if 'RTD Zonal LBMP' in hourly_lbmp.columns and 'TWI Actual Load' in hourly_lbmp.columns:
            hourly_lbmp['price_demand'] = hourly_lbmp['RTD Zonal LBMP'] * hourly_lbmp['TWI Actual Load']
            hourly_lbmp['price_per_MW'] = hourly_lbmp['RTD Zonal LBMP'] / (hourly_lbmp['TWI Actual Load'] + 1)
            hourly_lbmp['demand_price_ratio'] = hourly_lbmp['TWI Actual Load'] / (hourly_lbmp['RTD Zonal LBMP'] + 1)
            print("    ✅ Additional features created")

        self.merged_dataset = hourly_lbmp
        print(f"✅ Final merged dataset: {len(self.merged_dataset)} records")

        # Split data by location for individual analysis
        for location in self.merged_dataset['location'].unique():
            self.location_data[location] = self.merged_dataset[self.merged_dataset['location'] == location].copy()
            print(f"   {location}: {len(self.location_data[location]):,} records")

    def prepare_congestion_model_data(self, data, include_location_features=False):
        """Prepare data for congestion forecasting."""

        # Base features
        feature_columns = [
            'month', 'day', 'hour',
            'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday',
            'holiday'
        ]

        # Add location dummy variables if requested
        if include_location_features and 'location' in data.columns:
            location_dummies = pd.get_dummies(data['location'], prefix='location')
            data = pd.concat([data, location_dummies], axis=1)
            feature_columns.extend(location_dummies.columns.tolist())

        # Add lagged congestion features
        congestion_features = ['congestion_yesterday', 'congestion_7_days_ago']
        for feature in congestion_features:
            if feature in data.columns:
                feature_columns.append(feature)

        # Add demand features
        if 'TWI Actual Load' in data.columns:
            feature_columns.append('TWI Actual Load')

        # Add weather features
        weather_features = ['temperature_F', 'humidity_pct']
        for feature in weather_features:
            if feature in data.columns:
                feature_columns.append(feature)

        # Add price features
        price_features = ['RTD Zonal LBMP', 'RTD Zonal Losses']
        for feature in price_features:
            if feature in data.columns:
                feature_columns.append(feature)

        # Create model dataset
        model_data = data[feature_columns + ['RTD Zonal Congestion']].copy()
        model_data = model_data.dropna(subset=['RTD Zonal Congestion'])

        # Fill missing values
        for col in feature_columns:
            if col in model_data.columns and model_data[col].isnull().sum() > 0:
                model_data[col] = model_data[col].fillna(model_data[col].median())

        X = model_data[feature_columns]
        y = model_data['RTD Zonal Congestion']

        return X, y, feature_columns

    def prepare_demand_model_data(self, data, include_location_features=False):
        """
        Prepare data for demand forecasting models.
        PROCEDURE COMPLIANT: No pricing or congestion features allowed.
        """

        # Base features for demand forecasting
        feature_columns = [
            'month', 'day', 'hour',
            'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday',
            'holiday'
        ]

        # Add location dummy variables if requested
        if include_location_features and 'location' in data.columns:
            location_dummies = pd.get_dummies(data['location'], prefix='location')
            data = pd.concat([data, location_dummies], axis=1)
            feature_columns.extend(location_dummies.columns.tolist())

        # Add lagged demand features (key for demand forecasting)
        lagged_features = ['demand_yesterday', 'demand_7_days_ago', 'demand_1_hour_ago',
                          'demand_24h_avg', 'demand_7d_avg']

        for feature in lagged_features:
            if feature in data.columns:
                feature_columns.append(feature)

        # Add weather features ONLY (temperature strongly affects electricity demand)
        weather_features = ['temperature_F', 'humidity_pct']

        for feature in weather_features:
            if feature in data.columns:
                feature_columns.append(feature)

        # ❌ REMOVED: Price and congestion features per procedure
        # price_features = ['RTD Zonal LBMP', 'RTD Zonal Losses', 'RTD Zonal Congestion']

        # Create model dataset
        model_data = data[feature_columns + ['TWI Actual Load']].copy()
        model_data = model_data.dropna(subset=['TWI Actual Load'])

        # Fill missing values
        for col in feature_columns:
            if col in model_data.columns and model_data[col].isnull().sum() > 0:
                model_data[col] = model_data[col].fillna(model_data[col].median())

        X = model_data[feature_columns]
        y = model_data['TWI Actual Load']

        return X, y, feature_columns

    def prepare_price_model_data(self, data, include_location_features=False):
        """
        Prepare data for price forecasting models.
        PROCEDURE COMPLIANT: No congestion features allowed.
        """

        # Base features for price forecasting
        feature_columns = [
            'month', 'day', 'hour',
            'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday',
            'holiday'
        ]

        # Add location dummy variables if requested
        if include_location_features and 'location' in data.columns:
            location_dummies = pd.get_dummies(data['location'], prefix='location')
            data = pd.concat([data, location_dummies], axis=1)
            feature_columns.extend(location_dummies.columns.tolist())

        # Add lagged price features (key for price forecasting)
        price_features = ['LBMP_yesterday', 'LBMP_7_days_ago', 'LBMP_1_hour_ago',
                         'price_24h_avg', 'price_7d_avg', 'price_24h_std',
                         'price_volatility_24h', 'price_range_24h',
                         'price_momentum_1h', 'price_momentum_24h']

        for feature in price_features:
            if feature in data.columns:
                feature_columns.append(feature)

        # Add demand features (demand strongly affects price)
        demand_features = ['TWI Actual Load']

        for feature in demand_features:
            if feature in data.columns:
                feature_columns.append(feature)

        # Add ONLY losses component (not congestion per procedure)
        lbmp_components = ['RTD Zonal Losses']  # ✅ REMOVED CONGESTION

        for feature in lbmp_components:
            if feature in data.columns:
                feature_columns.append(feature)

        # Add weather features (affects demand and hence price)
        weather_features = ['temperature_F', 'humidity_pct']

        for feature in weather_features:
            if feature in data.columns:
                feature_columns.append(feature)

        # Add interaction features
        interaction_features = ['price_per_MW', 'demand_price_ratio']

        for feature in interaction_features:
            if feature in data.columns:
                feature_columns.append(feature)

        # Create model dataset
        model_data = data[feature_columns + ['RTD Zonal LBMP']].copy()
        model_data = model_data.dropna(subset=['RTD Zonal LBMP'])

        # Fill missing values
        for col in feature_columns:
            if col in model_data.columns and model_data[col].isnull().sum() > 0:
                model_data[col] = model_data[col].fillna(model_data[col].median())

        X = model_data[feature_columns]
        y = model_data['RTD Zonal LBMP']

        return X, y, feature_columns

    def train_all_models(self):
        """Train all three models as per procedure."""
        print("\n🎯 Training All Models Per Procedure...")

        # Train congestion models
        self.train_congestion_models()

        # Train demand models
        self.train_demand_models()

        # Train price models
        self.train_price_models()

    def train_congestion_models(self):
        """Train congestion forecasting models for each location."""
        print("\n📊 Training Congestion Models...")

        for location, data in self.location_data.items():
            if len(data) < 100:
                print(f"  ⚠️ Skipping {location}: insufficient data ({len(data)} records)")
                continue

            print(f"  Training congestion model for {location}...")

            # Prepare data
            X, y, feature_names = self.prepare_congestion_model_data(data, include_location_features=False)

            if len(X) == 0:
                print(f"    ❌ No valid data for {location}")
                continue

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            if XGBOOST_AVAILABLE:
                try:
                    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
                    model_type = "XGBoost"
                except:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model_type = "RandomForest"
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model_type = "RandomForest"

            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # Store results
            self.congestion_models[location] = {
                'model': model,
                'model_type': model_type,
                'X_test': X_test,
                'y_test': y_test,
                'y_pred': y_pred,
                'feature_names': feature_names,
                'metrics': {'r2': r2, 'mape': mape, 'rmse': rmse}
            }

            print(f"    📊 {location} {model_type} Congestion Results:")
            print(f"       R²: {r2:.4f}")
            print(f"       MAPE: {mape:.4f}")
            print(f"       RMSE: {rmse:.4f}")

    def train_demand_models(self):
        """Train demand forecasting models for each location."""
        print("\n⚡ Training Demand Models (No Pricing/Congestion)...")

        for location, data in self.location_data.items():
            if len(data) < 100:
                print(f"  ⚠️ Skipping {location}: insufficient data ({len(data)} records)")
                continue

            print(f"  Training demand model for {location}...")

            # Prepare data
            X, y, feature_names = self.prepare_demand_model_data(data, include_location_features=False)

            if len(X) == 0:
                print(f"    ❌ No valid data for {location}")
                continue

            # Split data (temporal split for demand forecasting)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            # Train model
            if XGBOOST_AVAILABLE:
                try:
                    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
                    model_type = "XGBoost"
                except:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model_type = "RandomForest"
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model_type = "RandomForest"

            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # Store results
            self.demand_models[location] = {
                'model': model,
                'model_type': model_type,
                'X_test': X_test,
                'y_test': y_test,
                'y_pred': y_pred,
                'feature_names': feature_names,
                'metrics': {'r2': r2, 'mape': mape, 'rmse': rmse}
            }

            print(f"    ⚡ {location} {model_type} Demand Results:")
            print(f"       R²: {r2:.4f}")
            print(f"       MAPE: {mape:.4f}")
            print(f"       RMSE: {rmse:.2f} MW")

    def train_price_models(self):
        """Train price forecasting models for each location."""
        print("\n💰 Training Price Models (No Congestion)...")

        for location, data in self.location_data.items():
            if len(data) < 100:
                print(f"  ⚠️ Skipping {location}: insufficient data ({len(data)} records)")
                continue

            print(f"  Training price model for {location}...")

            # Prepare data
            X, y, feature_names = self.prepare_price_model_data(data, include_location_features=False)

            if len(X) == 0:
                print(f"    ❌ No valid data for {location}")
                continue

            # Split data (temporal split for price forecasting)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            # Train model
            if XGBOOST_AVAILABLE:
                try:
                    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
                    model_type = "XGBoost"
                except:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model_type = "RandomForest"
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model_type = "RandomForest"

            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # Store results
            self.price_models[location] = {
                'model': model,
                'model_type': model_type,
                'X_test': X_test,
                'y_test': y_test,
                'y_pred': y_pred,
                'feature_names': feature_names,
                'metrics': {'r2': r2, 'mape': mape, 'rmse': rmse}
            }

            print(f"    💰 {location} {model_type} Price Results:")
            print(f"       R²: {r2:.4f}")
            print(f"       MAPE: {mape:.4f}")
            print(f"       RMSE: ${rmse:.2f}/MWh")

    def create_comprehensive_visualizations(self):
        """
        Create comprehensive visualizations for all analyses.
        Outputs of models:
        * X y graph for pricing, demand, and congestion
        * X is original price, demand, congestion
        * Y is predict price, demand, congestion
        """
        print("\n📈 Creating Comprehensive Visualizations...")

        # 1. REGIONAL COMPARISON DASHBOARD
        self.create_regional_comparison_dashboard()

        # 2. ACTUAL VS PREDICTED PLOTS (as per procedure)
        self.create_actual_vs_predicted_plots()

        # 3. FEATURE IMPORTANCE COMPARISON
        self.create_feature_importance_comparison()

        # 4. TIME SERIES ANALYSIS
        self.create_time_series_analysis()

    def create_regional_comparison_dashboard(self):
        """Create regional comparison dashboard."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('NYC Energy Market Regional Comparison Dashboard', fontsize=16, fontweight='bold')

        if self.merged_dataset.empty:
            return

        # 1. Average Congestion by Location
        congestion_by_location = self.merged_dataset.groupby('location')['RTD Zonal Congestion'].mean()
        axes[0, 0].bar(congestion_by_location.index, congestion_by_location.values,
                      color=['lightblue', 'lightgreen', 'lightcoral'])
        axes[0, 0].set_title('Average Congestion by Region')
        axes[0, 0].set_ylabel('Average Congestion ($/MWh)')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Average Demand by Location
        if 'TWI Actual Load' in self.merged_dataset.columns:
            demand_by_location = self.merged_dataset.groupby('location')['TWI Actual Load'].mean()
            axes[0, 1].bar(demand_by_location.index, demand_by_location.values,
                          color=['lightblue', 'lightgreen', 'lightcoral'])
            axes[0, 1].set_title('Average Demand by Region')
            axes[0, 1].set_ylabel('Average Demand (MW)')
            axes[0, 1].grid(True, alpha=0.3)

        # 3. Average Price by Location
        price_by_location = self.merged_dataset.groupby('location')['RTD Zonal LBMP'].mean()
        axes[0, 2].bar(price_by_location.index, price_by_location.values,
                      color=['lightblue', 'lightgreen', 'lightcoral'])
        axes[0, 2].set_title('Average Price by Region')
        axes[0, 2].set_ylabel('Average Price ($/MWh)')
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Congestion Model Performance
        if self.congestion_models:
            locations = list(self.congestion_models.keys())
            r2_scores = [self.congestion_models[loc]['metrics']['r2'] for loc in locations]
            axes[1, 0].bar(locations, r2_scores, color=['lightblue', 'lightgreen', 'lightcoral'])
            axes[1, 0].set_title('Congestion Model Performance (R²)')
            axes[1, 0].set_ylabel('R² Score')
            axes[1, 0].grid(True, alpha=0.3)

        # 5. Demand Model Performance
        if self.demand_models:
            locations = list(self.demand_models.keys())
            r2_scores = [self.demand_models[loc]['metrics']['r2'] for loc in locations]
            axes[1, 1].bar(locations, r2_scores, color=['lightblue', 'lightgreen', 'lightcoral'])
            axes[1, 1].set_title('Demand Model Performance (R²)')
            axes[1, 1].set_ylabel('R² Score')
            axes[1, 1].grid(True, alpha=0.3)

        # 6. Price Model Performance
        if self.price_models:
            locations = list(self.price_models.keys())
            r2_scores = [self.price_models[loc]['metrics']['r2'] for loc in locations]
            axes[1, 2].bar(locations, r2_scores, color=['lightblue', 'lightgreen', 'lightcoral'])
            axes[1, 2].set_title('Price Model Performance (R²)')
            axes[1, 2].set_ylabel('R² Score')
            axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def create_actual_vs_predicted_plots(self):
        """
        Create actual vs predicted plots as per procedure.
        X y graph for pricing, demand, and congestion
        X is original price, demand, congestion
        Y is predict price, demand, congestion
        """
        print("📊 Creating Actual vs Predicted Plots (Per Procedure)...")

        # Determine number of locations
        all_locations = set()
        if self.congestion_models:
            all_locations.update(self.congestion_models.keys())
        if self.demand_models:
            all_locations.update(self.demand_models.keys())
        if self.price_models:
            all_locations.update(self.price_models.keys())

        all_locations = list(all_locations)
        n_locations = len(all_locations)

        if n_locations == 0:
            print("No models available for plotting.")
            return

        # Create plots for each model type
        fig, axes = plt.subplots(3, n_locations, figsize=(5*n_locations, 15))
        if n_locations == 1:
            axes = axes.reshape(3, 1)

        fig.suptitle('Actual vs Predicted: Congestion, Demand, and Price Models', fontsize=16, fontweight='bold')

        for col, location in enumerate(all_locations):
            # 1. Congestion Model Plot
            if location in self.congestion_models:
                model_info = self.congestion_models[location]
                y_test = model_info['y_test']
                y_pred = model_info['y_pred']
                r2 = model_info['metrics']['r2']

                axes[0, col].scatter(y_test, y_pred, alpha=0.6, s=20, color='red')
                min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
                axes[0, col].plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')
                axes[0, col].set_xlabel('Actual Congestion ($/MWh)')
                axes[0, col].set_ylabel('Predicted Congestion ($/MWh)')
                axes[0, col].set_title(f'{location} Congestion\nR² = {r2:.3f}')
                axes[0, col].grid(True, alpha=0.3)
                axes[0, col].legend()

            # 2. Demand Model Plot
            if location in self.demand_models:
                model_info = self.demand_models[location]
                y_test = model_info['y_test']
                y_pred = model_info['y_pred']
                r2 = model_info['metrics']['r2']

                axes[1, col].scatter(y_test, y_pred, alpha=0.6, s=20, color='blue')
                min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
                axes[1, col].plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')
                axes[1, col].set_xlabel('Actual Demand (MW)')
                axes[1, col].set_ylabel('Predicted Demand (MW)')
                axes[1, col].set_title(f'{location} Demand\nR² = {r2:.3f}')
                axes[1, col].grid(True, alpha=0.3)
                axes[1, col].legend()

            # 3. Price Model Plot
            if location in self.price_models:
                model_info = self.price_models[location]
                y_test = model_info['y_test']
                y_pred = model_info['y_pred']
                r2 = model_info['metrics']['r2']

                axes[2, col].scatter(y_test, y_pred, alpha=0.6, s=20, color='green')
                min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
                axes[2, col].plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')
                axes[2, col].set_xlabel('Actual Price ($/MWh)')
                axes[2, col].set_ylabel('Predicted Price ($/MWh)')
                axes[2, col].set_title(f'{location} Price\nR² = {r2:.3f}')
                axes[2, col].grid(True, alpha=0.3)
                axes[2, col].legend()

        plt.tight_layout()
        plt.show()

    def create_feature_importance_comparison(self):
        """Create feature importance comparison across all models."""
        print("📊 Creating Feature Importance Analysis...")

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Feature Importance Comparison Across All Models', fontsize=16, fontweight='bold')

        # 1. Congestion Model Feature Importance
        if self.congestion_models:
            self._plot_feature_importance(self.congestion_models, axes[0], "Congestion Models")

        # 2. Demand Model Feature Importance
        if self.demand_models:
            self._plot_feature_importance(self.demand_models, axes[1], "Demand Models")

        # 3. Price Model Feature Importance
        if self.price_models:
            self._plot_feature_importance(self.price_models, axes[2], "Price Models")

        plt.tight_layout()
        plt.show()

    def _plot_feature_importance(self, models, ax, title):
        """Helper function to plot feature importance."""
        feature_importance_data = {}

        for location, model_info in models.items():
            if not hasattr(model_info['model'], 'feature_importances_'):
                continue

            importances = model_info['model'].feature_importances_
            feature_names = model_info['feature_names']
            feature_importance_data[location] = dict(zip(feature_names, importances))

        if not feature_importance_data:
            ax.text(0.5, 0.5, 'No feature importance data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return

        # Create DataFrame for easier plotting
        importance_df = pd.DataFrame(feature_importance_data).fillna(0)

        # Plot top 10 most important features
        feature_means = importance_df.mean(axis=1).sort_values(ascending=False)
        top_features = feature_means.head(10).index

        importance_subset = importance_df.loc[top_features]

        # Create grouped bar chart
        x = np.arange(len(top_features))
        width = 0.25
        colors = ['lightblue', 'lightgreen', 'lightcoral']

        for i, location in enumerate(importance_subset.columns):
            ax.bar(x + i*width, importance_subset[location], width,
                   label=location, color=colors[i % len(colors)])

        ax.set_xlabel('Features')
        ax.set_ylabel('Importance Score')
        ax.set_title(title)
        ax.set_xticks(x + width)
        ax.set_xticklabels(top_features, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def create_time_series_analysis(self):
        """Create time series analysis of all variables."""
        if self.merged_dataset.empty:
            return

        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle('NYC Energy Market Time Series Analysis', fontsize=16, fontweight='bold')

        # 1. Congestion trends
        monthly_congestion = self.merged_dataset.groupby(['location', 'year', 'month'])['RTD Zonal Congestion'].mean().reset_index()
        monthly_congestion['date'] = pd.to_datetime(monthly_congestion[['year', 'month']].assign(day=1))

        for location in monthly_congestion['location'].unique():
            location_monthly = monthly_congestion[monthly_congestion['location'] == location]
            axes[0].plot(location_monthly['date'], location_monthly['RTD Zonal Congestion'],
                        label=location, linewidth=2, marker='o', markersize=4)

        axes[0].set_title('Monthly Average Congestion Trends')
        axes[0].set_ylabel('Congestion ($/MWh)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. Demand trends
        if 'TWI Actual Load' in self.merged_dataset.columns:
            monthly_demand = self.merged_dataset.groupby(['location', 'year', 'month'])['TWI Actual Load'].mean().reset_index()
            monthly_demand['date'] = pd.to_datetime(monthly_demand[['year', 'month']].assign(day=1))

            for location in monthly_demand['location'].unique():
                location_monthly = monthly_demand[monthly_demand['location'] == location]
                axes[1].plot(location_monthly['date'], location_monthly['TWI Actual Load'],
                            label=location, linewidth=2, marker='s', markersize=4)

            axes[1].set_title('Monthly Average Demand Trends')
            axes[1].set_ylabel('Demand (MW)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        # 3. Price trends
        monthly_price = self.merged_dataset.groupby(['location', 'year', 'month'])['RTD Zonal LBMP'].mean().reset_index()
        monthly_price['date'] = pd.to_datetime(monthly_price[['year', 'month']].assign(day=1))

        for location in monthly_price['location'].unique():
            location_monthly = monthly_price[monthly_price['location'] == location]
            axes[2].plot(location_monthly['date'], location_monthly['RTD Zonal LBMP'],
                        label=location, linewidth=2, marker='^', markersize=4)

        axes[2].set_title('Monthly Average Price Trends')
        axes[2].set_xlabel('Date')
        axes[2].set_ylabel('Price ($/MWh)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def generate_analysis_report(self):
        """Generate comprehensive analysis report."""
        print("\n📋 NYC ENERGY MARKET ANALYSIS REPORT")
        print("=" * 60)

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"Report Generated: {current_time}")

        # Data Summary
        print(f"\n📊 DATA OVERVIEW:")
        print(f"   Total Records: {len(self.merged_dataset):,}")
        if 'location' in self.merged_dataset.columns:
            for location in self.merged_dataset['location'].unique():
                count = len(self.merged_dataset[self.merged_dataset['location'] == location])
                print(f"   {location}: {count:,} records")

        # Date Range
        if 'RTD End Time Stamp' in self.merged_dataset.columns:
            min_date = self.merged_dataset['RTD End Time Stamp'].min()
            max_date = self.merged_dataset['RTD End Time Stamp'].max()
            print(f"   Date Range: {min_date.date()} to {max_date.date()}")

        # Model Performance Summary
        print(f"\n🎯 MODEL PERFORMANCE SUMMARY:")

        # Congestion Models
        print(f"\n   📊 CONGESTION MODELS:")
        for location, model_info in self.congestion_models.items():
            metrics = model_info['metrics']
            model_type = model_info['model_type']
            print(f"     {location} ({model_type}): R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")

        # Demand Models
        print(f"\n   ⚡ DEMAND MODELS (No Pricing/Congestion):")
        for location, model_info in self.demand_models.items():
            metrics = model_info['metrics']
            model_type = model_info['model_type']
            print(f"     {location} ({model_type}): R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.2f} MW")

        # Price Models
        print(f"\n   💰 PRICE MODELS (No Congestion):")
        for location, model_info in self.price_models.items():
            metrics = model_info['metrics']
            model_type = model_info['model_type']
            print(f"     {location} ({model_type}): R²={metrics['r2']:.4f}, RMSE=${metrics['rmse']:.2f}/MWh")

        # Key Insights
        print(f"\n🔍 KEY INSIGHTS:")

        if not self.merged_dataset.empty:
            # Best performing models
            if self.congestion_models:
                congestion_r2 = {loc: info['metrics']['r2'] for loc, info in self.congestion_models.items()}
                best_congestion = max(congestion_r2, key=congestion_r2.get)
                print(f"   Best Congestion Model: {best_congestion} (R² = {congestion_r2[best_congestion]:.4f})")

            if self.demand_models:
                demand_r2 = {loc: info['metrics']['r2'] for loc, info in self.demand_models.items()}
                best_demand = max(demand_r2, key=demand_r2.get)
                print(f"   Best Demand Model: {best_demand} (R² = {demand_r2[best_demand]:.4f})")

            if self.price_models:
                price_r2 = {loc: info['metrics']['r2'] for loc, info in self.price_models.items()}
                best_price = max(price_r2, key=price_r2.get)
                print(f"   Best Price Model: {best_price} (R² = {price_r2[best_price]:.4f})")

        print(f"\n✅ Analysis Complete - Procedure Compliant!")
        print(f"   ✅ Congestion Model: Uses RTD Zonal Congestion as y")
        print(f"   ✅ Demand Model: Uses TWI Actual Load as y (No pricing/congestion features)")
        print(f"   ✅ Price Model: Uses RTD Zonal LBMP as y (No congestion features)")

    def run_complete_analysis(self):
        """Run the complete NYC energy analysis."""
        print("🗽 NYC ENERGY DATA ANALYSIS - PROCEDURE COMPLIANT")
        print("=" * 65)

        try:
            # Load all data
            self.load_all_merged_files()

            # Show data summary
            self.show_data_summary()

            # Merge and prepare data
            self.merge_all_data()

            if self.merged_dataset.empty:
                print("❌ No merged data available for analysis!")
                return

            # Train all models
            self.train_all_models()

            # Create visualizations
            self.create_comprehensive_visualizations()

            # Generate report
            self.generate_analysis_report()

            print(f"\n🎉 Complete NYC Energy Analysis Finished!")

            return {
                'congestion_models': self.congestion_models,
                'demand_models': self.demand_models,
                'price_models': self.price_models,
                'merged_dataset': self.merged_dataset
            }

        except Exception as e:
            print(f"❌ Error in analysis: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main function to run the NYC energy analysis."""

    # Initialize analyzer
    analyzer = NYCEnergyAnalyzer(data_directory=".")

    # Run complete analysis
    results = analyzer.run_complete_analysis()

    return results

if __name__ == "__main__":
    results = main()
