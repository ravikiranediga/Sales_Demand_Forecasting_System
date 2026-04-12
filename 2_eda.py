import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)

df = pd.read_csv('data/sales_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

print(f"\nDataset Shape: {df.shape}")
print(f"Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}")

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Sales by Store ---")
store_sales = df.groupby('Store')['Sales'].sum()
print(store_sales)

print("\n--- Sales by Category ---")
cat_sales = df.groupby('Product_Category')['Sales'].sum().sort_values(ascending=False)
print(cat_sales)

print("\n--- Sales by Region ---")
region_sales = df.groupby('Region')['Sales'].sum()
print(region_sales)

print("\n--- Monthly Sales Trend ---")
df['YearMonth'] = df['Date'].dt.to_period('M')
monthly_sales = df.groupby('YearMonth')['Sales'].sum()
print(monthly_sales.head(10))

print("\n--- Seasonality (Avg Sales by Month) ---")
df['Month'] = df['Date'].dt.month
monthly_avg = df.groupby('Month')['Sales'].mean()
print(monthly_avg.round(2))

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
store_sales.plot(kind='bar', color=['#2ecc71', '#3498db', '#e74c3c'])
plt.title('Total Sales by Store')
plt.xlabel('Store')
plt.ylabel('Sales ($)')
plt.xticks(rotation=0)

plt.subplot(1, 2, 2)
cat_sales.plot(kind='bar', color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12'])
plt.title('Total Sales by Category')
plt.xlabel('Category')
plt.ylabel('Sales ($)')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('visualizations/eda_store_category.png', dpi=150)
plt.close()
print("\nSaved: visualizations/eda_store_category.png")

plt.figure(figsize=(12, 5))
monthly_sales.plot(marker='o', linewidth=2, color='#3498db')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Sales ($)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/eda_monthly_trend.png', dpi=150)
plt.close()
print("Saved: visualizations/eda_monthly_trend.png")

plt.figure(figsize=(10, 5))
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.bar(range(1, 13), monthly_avg.values, color=plt.cm.viridis(np.linspace(0.2, 0.8, 12)))
plt.xticks(range(1, 13), months)
plt.title('Average Sales by Month (Seasonality)')
plt.xlabel('Month')
plt.ylabel('Average Sales ($)')
plt.tight_layout()
plt.savefig('visualizations/eda_seasonality.png', dpi=150)
plt.close()
print("Saved: visualizations/eda_seasonality.png")

print("\nEDA Complete!")