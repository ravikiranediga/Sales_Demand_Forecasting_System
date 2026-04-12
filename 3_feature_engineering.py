import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

print("=" * 60)
print("STEP 3: FEATURE ENGINEERING")
print("=" * 60)

df = pd.read_csv('data/sales_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
print(f"\nLoaded {len(df)} records")

print("\n[1] Creating time-based features...")
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
df['Quarter'] = df['Date'].dt.quarter
df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)
print("    - Basic time features added")

print("\n[2] Creating cyclical features...")
df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
df['DayOfYear'] = df['Date'].dt.dayofyear
df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
print("    - Cyclical features added")

print("\n[3] Creating lag features...")
df = df.sort_values(['Store', 'Product_Category', 'Date']).reset_index(drop=True)
for lag in [7, 14, 21, 30]:
    df[f'Sales_Lag_{lag}'] = df.groupby(['Store', 'Product_Category'])['Sales'].shift(lag)
print("    - Lag features: Sales_Lag_7, 14, 21, 30")

print("\n[4] Creating rolling statistics...")
for window in [7, 14, 30]:
    df[f'Sales_RollingMean_{window}'] = df.groupby(['Store', 'Product_Category'])['Sales'].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
    )
    df[f'Sales_RollingStd_{window}'] = df.groupby(['Store', 'Product_Category'])['Sales'].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
    )
print("    - Rolling features: Mean and Std for 7, 14, 30 day windows")

print("\n[5] Encoding categorical variables...")
le_store = LabelEncoder()
le_category = LabelEncoder()
le_region = LabelEncoder()
df['Store_Encoded'] = le_store.fit_transform(df['Store'])
df['Category_Encoded'] = le_category.fit_transform(df['Product_Category'])
df['Region_Encoded'] = le_region.fit_transform(df['Region'])
print(f"    - Store: {list(le_store.classes_)}")
print(f"    - Category: {list(le_category.classes_)}")
print(f"    - Region: {list(le_region.classes_)}")

feature_cols = [
    'Year', 'Month', 'Day', 'DayOfWeek', 'WeekOfYear', 'Quarter',
    'IsWeekend', 'IsMonthStart', 'IsMonthEnd',
    'Month_sin', 'Month_cos', 'DayOfWeek_sin', 'DayOfWeek_cos',
    'DayOfYear_sin', 'DayOfYear_cos',
    'Sales_Lag_7', 'Sales_Lag_14', 'Sales_Lag_21', 'Sales_Lag_30',
    'Sales_RollingMean_7', 'Sales_RollingMean_14', 'Sales_RollingMean_30',
    'Sales_RollingStd_7', 'Sales_RollingStd_14', 'Sales_RollingStd_30',
    'Store_Encoded', 'Category_Encoded', 'Region_Encoded'
]

print(f"\n[6] Total features created: {len(feature_cols)}")
print(f"    Features: {feature_cols}")

df_clean = df.dropna()
df_clean.to_csv('data/processed_sales.csv', index=False)
print(f"\n[7] Saved processed data: {len(df_clean)} records")
print("    Path: data/processed_sales.csv")

print("\nFeature Engineering Complete!")