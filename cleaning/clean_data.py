"""
AeroTwin - Data Cleaning & Preprocessing Module
===============================================
Author: AeroTwin Team
Target Audience: B.Tech Engineering / Student Level

Purpose:
    Inspects, validates, cleans, and pre-processes raw air quality data from `data/raw/`
    and exports high-quality, analysis-ready datasets to `data/processed/`.

Key Cleaning Rules Applied:
    1. Zero Data Loss for Valid Observations:
       Only rows where ALL 12 pollutant measurements are completely missing (sensors offline)
       are removed. No arbitrary mass deletion of rows.
    2. Continuous Time-Series Imputation:
       Missing pollutant concentrations are imputed using chronological forward/backward fill
       within each city/station group, followed by city-level median imputation.
       Values are NOT blindly filled with zero (which would falsely imply 100% clean air).
    3. Extreme Value Preservation:
       Real-world severe smog episodes (AQI > 400, PM2.5 > 500) are preserved as authentic
       health-hazard events rather than deleted as outliers. An explicit indicator flag
       (`is_extreme_pollution`) is added for downstream modeling.
    4. Categorical Consistency:
       Missing AQI buckets are derived strictly using Central Pollution Control Board (CPCB)
       official AQI category cutoffs.
"""

import os
import pandas as pd
import numpy as np


# Setup Paths relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

POLLUTANT_COLUMNS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"
]


def get_cpcb_aqi_bucket(aqi_value):
    """
    Categorizes an Air Quality Index (AQI) value into the official CPCB categories.
    
    Parameters:
        aqi_value (float): Numerical AQI value
        
    Returns:
        str: AQI category ('Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe')
    """
    if pd.isnull(aqi_value):
        return "Unknown"
    elif aqi_value <= 50:
        return "Good"
    elif aqi_value <= 100:
        return "Satisfactory"
    elif aqi_value <= 200:
        return "Moderate"
    elif aqi_value <= 300:
        return "Poor"
    elif aqi_value <= 400:
        return "Very Poor"
    else:
        return "Severe"


def clean_stations_data():
    """
    Cleans the stations reference dataset.
    
    Returns:
        pd.DataFrame: Cleaned stations dataframe.
    """
    raw_path = os.path.join(RAW_DIR, "stations.csv")
    output_path = os.path.join(PROCESSED_DIR, "cleaned_stations.csv")
    
    print("\n" + "=" * 50)
    print(">>> 1. Processing stations.csv...")
    print("=" * 50)
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing file: {raw_path}")
        
    df = pd.read_csv(raw_path)
    initial_rows = len(df)
    
    # Standardize string columns (strip accidental leading/trailing spaces)
    for col in ["StationId", "StationName", "City", "State", "Status"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            
    # Handle missing status: 'nan' strings converted to 'Unknown'
    df["Status"] = df["Status"].replace({"nan": "Unknown", "": "Unknown"})
    df["Status"] = df["Status"].fillna("Unknown")
    
    # Save cleaned version
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Stations: {initial_rows} rows processed, saved to -> {output_path}")
    return df


def clean_city_day_data():
    """
    Cleans the city-level daily air quality dataset.
    
    Returns:
        pd.DataFrame: Cleaned city_day dataframe.
    """
    raw_path = os.path.join(RAW_DIR, "city_day.csv")
    output_path = os.path.join(PROCESSED_DIR, "cleaned_city_day.csv")
    
    print("\n" + "=" * 50)
    print(">>> 2. Processing city_day.csv...")
    print("=" * 50)
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing file: {raw_path}")
        
    df = pd.read_csv(raw_path)
    initial_rows = len(df)
    initial_nulls = df.isnull().sum()
    
    # 1. Parse Date and sort chronologically by City and Date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["City"] = df["City"].astype(str).str.strip()
    df = df.sort_values(by=["City", "Date"]).reset_index(drop=True)
    
    # 2. Drop rows where ALL 12 pollutants are missing (offline sensors)
    all_pollutants_missing = df[POLLUTANT_COLUMNS].isnull().all(axis=1)
    dropped_blank_rows = all_pollutants_missing.sum()
    df = df[~all_pollutants_missing].copy().reset_index(drop=True)
    
    # 3. Time-series interpolation: chronological ffill & bfill within each city
    df[POLLUTANT_COLUMNS] = df.groupby("City")[POLLUTANT_COLUMNS].ffill().bfill()
    
    # 4. Fill remaining sparse pollutant gaps using City median, then global median
    for col in POLLUTANT_COLUMNS:
        city_median = df.groupby("City")[col].transform("median")
        df[col] = df[col].fillna(city_median).fillna(df[col].median())
        
    # 5. Handle AQI: fill via chronological continuity within city, then city median
    df["AQI"] = df.groupby("City")["AQI"].ffill().bfill()
    df["AQI"] = df["AQI"].fillna(df.groupby("City")["AQI"].transform("median")).fillna(df["AQI"].median())
    
    # 6. Synchronize AQI_Bucket using CPCB official classification
    df["AQI_Bucket"] = df["AQI"].apply(get_cpcb_aqi_bucket)
    
    # 7. Add calendar feature columns useful for time-series / ML modeling
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.day_name()
    
    # 8. Flag extreme pollution events without deleting them
    df["is_extreme_pollution"] = (df["AQI"] > 400) | (df["PM2.5"] > 500)
    
    # Save cleaned version
    df.to_csv(output_path, index=False)
    
    final_rows = len(df)
    final_nulls = df.isnull().sum()
    
    print(f"city_day: Initial Rows = {initial_rows} | Blank Rows Dropped = {dropped_blank_rows} | Final Rows = {final_rows}")
    print(f"city_day saved to -> {output_path}")
    
    return df, initial_rows, final_rows, initial_nulls, final_nulls


def clean_station_day_data():
    """
    Cleans the station-level daily air quality dataset.
    
    Returns:
        pd.DataFrame: Cleaned station_day dataframe.
    """
    raw_path = os.path.join(RAW_DIR, "station_day.csv")
    output_path = os.path.join(PROCESSED_DIR, "cleaned_station_day.csv")
    
    print("\n" + "=" * 50)
    print(">>> 3. Processing station_day.csv...")
    print("=" * 50)
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing file: {raw_path}")
        
    df = pd.read_csv(raw_path)
    initial_rows = len(df)
    initial_nulls = df.isnull().sum()
    
    # 1. Parse Date and sort chronologically by StationId and Date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["StationId"] = df["StationId"].astype(str).str.strip()
    df = df.sort_values(by=["StationId", "Date"]).reset_index(drop=True)
    
    # 2. Drop rows where ALL 12 pollutants are missing
    all_pollutants_missing = df[POLLUTANT_COLUMNS].isnull().all(axis=1)
    dropped_blank_rows = all_pollutants_missing.sum()
    df = df[~all_pollutants_missing].copy().reset_index(drop=True)
    
    # 3. Time-series interpolation: chronological ffill & bfill within StationId
    df[POLLUTANT_COLUMNS] = df.groupby("StationId")[POLLUTANT_COLUMNS].ffill().bfill()
    
    # 4. Fill remaining sparse pollutant gaps using Station median, then global median
    for col in POLLUTANT_COLUMNS:
        st_median = df.groupby("StationId")[col].transform("median")
        df[col] = df[col].fillna(st_median).fillna(df[col].median())
        
    # 5. Handle AQI: fill via chronological continuity within station, then station median
    df["AQI"] = df.groupby("StationId")["AQI"].ffill().bfill()
    df["AQI"] = df["AQI"].fillna(df.groupby("StationId")["AQI"].transform("median")).fillna(df["AQI"].median())
    
    # 6. Synchronize AQI_Bucket
    df["AQI_Bucket"] = df["AQI"].apply(get_cpcb_aqi_bucket)
    
    # 7. Add calendar feature columns
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.day_name()
    
    # 8. Flag extreme pollution events
    df["is_extreme_pollution"] = (df["AQI"] > 400) | (df["PM2.5"] > 500)
    
    # Save cleaned version
    df.to_csv(output_path, index=False)
    
    final_rows = len(df)
    final_nulls = df.isnull().sum()
    
    print(f"station_day: Initial Rows = {initial_rows} | Blank Rows Dropped = {dropped_blank_rows} | Final Rows = {final_rows}")
    print(f"station_day saved to -> {output_path}")
    
    return df, initial_rows, final_rows, initial_nulls, final_nulls


def clean_city_hour_data():
    """
    Cleans the city-level hourly air quality dataset.
    
    Returns:
        pd.DataFrame: Cleaned city_hour dataframe.
    """
    raw_path = os.path.join(RAW_DIR, "city_hour.csv")
    output_path = os.path.join(PROCESSED_DIR, "cleaned_city_hour.csv")
    
    print("\n" + "=" * 50)
    print(">>> 4. Processing city_hour.csv...")
    print("=" * 50)
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing file: {raw_path}")
        
    df = pd.read_csv(raw_path)
    initial_rows = len(df)
    initial_nulls = df.isnull().sum()
    
    # 1. Parse Datetime and sort chronologically
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df["City"] = df["City"].astype(str).str.strip()
    df = df.sort_values(by=["City", "Datetime"]).reset_index(drop=True)
    
    # 2. Drop rows where ALL 12 pollutants are missing
    all_pollutants_missing = df[POLLUTANT_COLUMNS].isnull().all(axis=1)
    dropped_blank_rows = all_pollutants_missing.sum()
    df = df[~all_pollutants_missing].copy().reset_index(drop=True)
    
    # 3. Time-series interpolation: chronological ffill & bfill within each city
    df[POLLUTANT_COLUMNS] = df.groupby("City")[POLLUTANT_COLUMNS].ffill().bfill()
    
    # 4. Fill remaining gaps with City median, then global median
    for col in POLLUTANT_COLUMNS:
        city_median = df.groupby("City")[col].transform("median")
        df[col] = df[col].fillna(city_median).fillna(df[col].median())
        
    # 5. AQI fill
    df["AQI"] = df.groupby("City")["AQI"].ffill().bfill()
    df["AQI"] = df["AQI"].fillna(df.groupby("City")["AQI"].transform("median")).fillna(df["AQI"].median())
    
    # 6. AQI_Bucket
    df["AQI_Bucket"] = df["AQI"].apply(get_cpcb_aqi_bucket)
    
    # 7. Add temporal features
    df["Hour"] = df["Datetime"].dt.hour
    df["Date"] = df["Datetime"].dt.date
    df["DayOfWeek"] = df["Datetime"].dt.day_name()
    
    # 8. Flag extreme pollution
    df["is_extreme_pollution"] = (df["AQI"] > 400) | (df["PM2.5"] > 500)
    
    # Save cleaned version
    df.to_csv(output_path, index=False)
    
    final_rows = len(df)
    final_nulls = df.isnull().sum()
    
    print(f"city_hour: Initial Rows = {initial_rows} | Blank Rows Dropped = {dropped_blank_rows} | Final Rows = {final_rows}")
    print(f"city_hour saved to -> {output_path}")
    
    return df, initial_rows, final_rows, initial_nulls, final_nulls


def main():
    """
    Main execution pipeline for data cleaning in AeroTwin.
    Cleans stations.csv, city_day.csv, and station_day.csv (the primary datasets for modeling).
    """
    print("=" * 60)
    print("      AEROTWIN - DATA CLEANING & VALIDATION PIPELINE      ")
    print("=" * 60)
    
    # Clean stations metadata
    clean_stations_data()
    
    # Clean daily datasets
    clean_city_day_data()
    clean_station_day_data()
    
    print("\n" + "=" * 60)
    print("All primary datasets cleaned and saved successfully to data/processed/!")
    print("=" * 60)


if __name__ == "__main__":
    main()
