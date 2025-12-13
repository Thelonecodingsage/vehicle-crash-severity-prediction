import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split


def preprocessed_data(file_path="data/US_Accidents_clean.csv", save_path=None):
    kaggle_df = pd.read_csv(file_path)


    # Handling Missing Data
    missing_values = kaggle_df.isnull().sum()
        #print(missing_values.sample(25))

    kaggle_df = kaggle_df.dropna(subset=['End_Lat', 'End_Lng'])

    nonnegative_cols = ['Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)','Humidity(%)', 'Temperature(F)', 'Pressure(in)']

    for col in nonnegative_cols:
        kaggle_df = kaggle_df[kaggle_df[col] >= 0]

    kaggle_df = kaggle_df[(kaggle_df['Humidity(%)'] >= 0) & (kaggle_df['Humidity(%)'] <= 100)]

    median_columns = ['Precipitation(in)', 'Wind_Chill(F)', 'Wind_Speed(mph)', 'Visibility(mi)', 'Humidity(%)', 'Temperature(F)', 'Pressure(in)']

    for column in median_columns:
        kaggle_df[column] = kaggle_df[column].fillna(kaggle_df[column].median())


    # Encoding and Transforming Categorical Data
    encoded_columns = ['Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight']
    
    for col in encoded_columns:
        kaggle_df[col] = pd.factorize(kaggle_df[col])[0]


    # Handling Redundant or Correlated Features
    pd.set_option('display.max_columns', 30)
    pd.set_option('display.max_rows', 30)

    correlations= kaggle_df.corr(numeric_only=True)


    # Feature Engineering
    kaggle_df['Start_Time'] = pd.to_datetime(
        kaggle_df['Start_Time'], format= 'mixed', errors='coerce'
        )

    kaggle_df['Hour_of_day'] = kaggle_df['Start_Time'].dt.hour
    kaggle_df['Day_of_week'] = kaggle_df['Start_Time'].dt.dayofweek

    kaggle_df['Date'] = pd.to_datetime({
        'year': kaggle_df['Start_Time'].dt.year,
        'month': kaggle_df['Start_Time'].dt.month,
        'day': kaggle_df['Start_Time'].dt.day
    })

    def get_seasons(date):
        year = date.year
    
        spring_start = pd.Timestamp(year=year, month=3, day=20)
        summer_start = pd.Timestamp(year=year, month=7, day=20)
        fall_start = pd.Timestamp(year=year, month=9, day=22)
        winter_start = pd.Timestamp(year=year, month=12, day=21)
    
        if spring_start <= date < summer_start:
            return 'Spring'
        elif summer_start <= date < fall_start:
            return 'Summer' 
        elif fall_start <= date < winter_start:
            return 'Fall'
        else:
            return 'Winter'
   
    kaggle_df['Season'] = kaggle_df['Start_Time'].apply(get_seasons)

    def categorize_visibility(v):
        if v == 0:
            return "Unknown"
        elif v < 1:
            return "Dangerous"
        elif v < 3:
            return "Obstruction"
        elif v < 6:
            return "Normal"
        else:
            return "Very Safe"

    kaggle_df['Categorized Visibility'] = kaggle_df['Visibility(mi)'].apply(categorize_visibility)

    season_dummies = pd.get_dummies(kaggle_df['Season'], prefix='Season')
    kaggle_df = pd.concat([kaggle_df, season_dummies], axis=1)

    visibility_dummies = pd.get_dummies(kaggle_df['Categorized Visibility'], prefix='Visibility')
    kaggle_df = pd.concat([kaggle_df, visibility_dummies], axis=1)

    kaggle_df['Rush Hour'] = kaggle_df['Hour_of_day'].isin([6,7,8,16,17,18])
    kaggle_df['Is_Weekend'] = (kaggle_df['Day_of_week'] >= 5).astype(int)

    kaggle_df['Is_Foggy'] = (kaggle_df['Visibility(mi)'] < 2).astype(int)
    
    kaggle_df['Crosswind Risk'] = kaggle_df['Wind_Speed(mph)'] * (1 / (kaggle_df['Visibility(mi)'] + 1))
    kaggle_df['Icy Risk'] = kaggle_df['Precipitation(in)'] * (32 - kaggle_df['Temperature(F)']).clip(lower=0)
    kaggle_df['Temp_Wind_Interaction'] = kaggle_df['Temperature(F)'] * kaggle_df['Wind_Chill(F)']
    

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

    # Train and Test Split for Severity
    X = kaggle_df.drop("Severity", axis=1)
    X = X.select_dtypes(include=[np.number])
    y = kaggle_df["Severity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    #Save and Export Processed Dataset for Modeling
    if save_path:
        kaggle_df.to_csv(save_path, index=False)
    
    return X_train, X_test, y_train, y_test, kaggle_df