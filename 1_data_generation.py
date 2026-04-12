import pandas as pd
import numpy as np

print("=" * 60)
print("STEP 1: DATA GENERATION")
print("=" * 60)

np.random.seed(42)

start_date = pd.Timestamp('2023-01-01')
end_date = pd.Timestamp('2025-03-31')
dates = pd.date_range(start_date, end_date, freq='D')

stores = ['Store_A', 'Store_B', 'Store_C']
categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Food & Beverages']
regions = ['North', 'South', 'East', 'West']

def generate_sales(date, store, category):
    base_sales = {
        'Electronics': 1500,
        'Clothing': 800,
        'Home & Garden': 600,
        'Sports': 400,
        'Food & Beverages': 1200
    }
    
    month = date.month
    day_of_week = date.dayofweek
    
    seasonal_factor = 1.0
    if month in [11, 12]:
        seasonal_factor = 1.4
    elif month in [6, 7, 8]:
        seasonal_factor = 1.2
    elif month in [1, 2]:
        seasonal_factor = 0.8
    
    if day_of_week >= 5:
        seasonal_factor *= 0.9
    else:
        seasonal_factor *= 1.1
    
    if store == 'Store_A':
        seasonal_factor *= 1.3
    elif store == 'Store_B':
        seasonal_factor *= 1.0
    else:
        seasonal_factor *= 0.8
    
    noise = np.random.normal(0, 0.15)
    sales = base_sales[category] * seasonal_factor * (1 + noise)
    quantity = int(sales / (base_sales[category] / 5))
    
    return round(max(100, sales), 2), max(1, quantity)

data = []
for date in dates:
    for store in stores:
        for category in categories:
            region = np.random.choice(regions)
            sales, quantity = generate_sales(date, store, category)
            data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Store': store,
                'Product_Category': category,
                'Region': region,
                'Sales': sales,
                'Quantity': quantity
            })

df = pd.DataFrame(data)
df.to_csv('data/sales_data.csv', index=False)
print(f"\nGenerated {len(df)} records")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Total sales: ${df['Sales'].sum():,.2f}")
print("\nData saved to data/sales_data.csv")