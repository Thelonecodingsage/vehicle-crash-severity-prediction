import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

kaggle_df = pd.read_csv("data/US_Accidents_clean.csv")

# Handling Missing Data
missing_values = kaggle_df.isnull().sum()

kaggle_df = kaggle_df.dropna(subset=['End_Lat', 'End_Lng'])

median_columns = ['Precipitation(in)', 'Wind_Chill(F)', 'Wind_Speed(mph)', 'Visibility(mi)', 'Humidity(%)', 'Temperature(F)', 'Pressure(in)']

for column in median_columns:
   kaggle_df[column] = kaggle_df[column].fillna(kaggle_df[column].median())

#missing = kaggle_df.isnull().mean().sort_values(ascending=False) * 100
#print(missing.head(20))

# Encoding and Transforming Categorical Data

# Data Scaling 
std_scaler = StandardScaler()

scaler_cols = ['Temperature(F)', 'Wind_Speed(mph)']
kaggle_df[scaler_cols] = std_scaler.fit_transform(kaggle_df[scaler_cols])

min_max_scaler = MinMaxScaler()

min_max_cols = ['Precipitation(in)']
kaggle_df[min_max_cols] = min_max_scaler.fit_transform(kaggle_df[min_max_cols])

robust_scaler = RobustScaler()

robust_cols = ['Humidity(%)', 'Pressure(in)', 'Visibility(mi)']
kaggle_df[robust_cols] = robust_scaler.fit_transform(kaggle_df[robust_cols])

# Outlier Detection

# Handling Redundant or Correlated Features

# Feature Engineering

# Save and Export Processed Dataset for Modeling

