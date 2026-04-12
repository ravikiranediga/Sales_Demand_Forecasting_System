import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

print("=" * 60)
print("STEP 5: FORECASTING & PREDICTIONS")
print("=" * 60)

df = pd.read_csv('data/sales_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
model = joblib.load('models/best_model.pkl')
print(f"\nLoaded trained model and sales data")

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

print("\n[1] Generating future dates (90 days)...")
last_date = df['Date'].max()
future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=90, freq='D')

stores = df['Store'].unique()
categories = df['Product_Category'].unique()
regions = df['Region'].unique()

future_data = []
for date in future_dates:
    for store in stores:
        for category in categories:
            future_data.append({
                'Date': date,
                'Store': store,
                'Product_Category': category,
                'Region': np.random.choice(regions)
            })

future_df = pd.DataFrame(future_data)
print(f"    Created {len(future_df)} future records")

print("\n[2] Creating features for future data...")
future_df['Year'] = future_df['Date'].dt.year
future_df['Month'] = future_df['Date'].dt.month
future_df['Day'] = future_df['Date'].dt.day
future_df['DayOfWeek'] = future_df['Date'].dt.dayofweek
future_df['WeekOfYear'] = future_df['Date'].dt.isocalendar().week.astype(int)
future_df['Quarter'] = future_df['Date'].dt.quarter
future_df['IsWeekend'] = (future_df['DayOfWeek'] >= 5).astype(int)
future_df['IsMonthStart'] = future_df['Date'].dt.is_month_start.astype(int)
future_df['IsMonthEnd'] = future_df['Date'].dt.is_month_end.astype(int)

future_df['Month_sin'] = np.sin(2 * np.pi * future_df['Month'] / 12)
future_df['Month_cos'] = np.cos(2 * np.pi * future_df['Month'] / 12)
future_df['DayOfWeek_sin'] = np.sin(2 * np.pi * future_df['DayOfWeek'] / 7)
future_df['DayOfWeek_cos'] = np.cos(2 * np.pi * future_df['DayOfWeek'] / 7)
future_df['DayOfYear'] = future_df['Date'].dt.dayofyear
future_df['DayOfYear_sin'] = np.sin(2 * np.pi * future_df['DayOfYear'] / 365)
future_df['DayOfYear_cos'] = np.cos(2 * np.pi * future_df['DayOfYear'] / 365)

store_map = {s: i for i, s in enumerate(stores)}
cat_map = {c: i for i, c in enumerate(categories)}
reg_map = {r: i for i, r in enumerate(regions)}

future_df['Store_Encoded'] = future_df['Store'].map(store_map)
future_df['Category_Encoded'] = future_df['Product_Category'].map(cat_map)
future_df['Region_Encoded'] = future_df['Region'].map(reg_map)

for lag in [7, 14, 21, 30]:
    future_df[f'Sales_Lag_{lag}'] = 0
for window in [7, 14, 30]:
    future_df[f'Sales_RollingMean_{window}'] = 0
    future_df[f'Sales_RollingStd_{window}'] = 0

X_future = future_df[feature_cols].fillna(0)

print("\n[3] Making predictions...")
future_df['Predicted_Sales'] = model.predict(X_future)

total_forecast = future_df['Predicted_Sales'].sum()
daily_avg = future_df.groupby('Date')['Predicted_Sales'].sum().mean()

print(f"\n[4] FORECAST RESULTS (Next 90 Days):")
print(f"    Total Predicted Sales: ${total_forecast:,.2f}")
print(f"    Average Daily Sales:  ${daily_avg:,.2f}")

store_forecast = future_df.groupby('Store')['Predicted_Sales'].sum()
print(f"\n    By Store:")
for store, sales in store_forecast.items():
    print(f"      {store}: ${sales:,.2f}")

cat_forecast = future_df.groupby('Product_Category')['Predicted_Sales'].sum().sort_values(ascending=False)
print(f"\n    By Category:")
for cat, sales in cat_forecast.items():
    print(f"      {cat}: ${sales:,.2f}")

future_df.to_csv('output/forecast_results.csv', index=False)
print(f"\n[5] Saved predictions to: output/forecast_results.csv")

print("\n[6] Generating visualizations...")
daily_forecast = future_df.groupby('Date')['Predicted_Sales'].sum()
historical_daily = df.groupby('Date')['Sales'].sum()

plt.figure(figsize=(14, 6))
plt.plot(historical_daily.index[-60:], historical_daily.values[-60:], 
         label='Historical (Last 60 days)', color='#3498db', linewidth=2)
plt.plot(daily_forecast.index, daily_forecast.values, 
         label='Forecast (Next 90 days)', color='#e74c3c', linewidth=2, linestyle='--')
plt.axvline(x=last_date, color='gray', linestyle=':', label='Today')
plt.title('Sales Forecast: Historical vs Future', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Total Sales ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/forecast_daily.png', dpi=150)
plt.close()
print("    Saved: visualizations/forecast_daily.png")

plt.figure(figsize=(10, 5))
future_df['Month'] = future_df['Date'].dt.to_period('M')
monthly_fc = future_df.groupby('Month')['Predicted_Sales'].sum()
plt.bar(range(len(monthly_fc)), monthly_fc.values, color=['#2ecc71', '#3498db', '#9b59b6'])
plt.xticks(range(len(monthly_fc)), [str(m) for m in monthly_fc.index])
plt.title('Monthly Sales Forecast', fontsize=14, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Predicted Sales ($)')
plt.tight_layout()
plt.savefig('visualizations/forecast_monthly.png', dpi=150)
plt.close()
print("    Saved: visualizations/forecast_monthly.png")

plt.figure(figsize=(10, 5))
plt.bar(store_forecast.index, store_forecast.values, color=['#2ecc71', '#3498db', '#e74c3c'])
plt.title('Forecast by Store', fontsize=14, fontweight='bold')
plt.xlabel('Store')
plt.ylabel('Predicted Sales ($)')
plt.tight_layout()
plt.savefig('visualizations/forecast_by_store.png', dpi=150)
plt.close()
print("    Saved: visualizations/forecast_by_store.png")

plt.figure(figsize=(10, 5))
plt.barh(cat_forecast.index, cat_forecast.values, color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12'])
plt.title('Forecast by Category', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Sales ($)')
plt.tight_layout()
plt.savefig('visualizations/forecast_by_category.png', dpi=150)
plt.close()
print("    Saved: visualizations/forecast_by_category.png")

print("\nForecasting Complete!")