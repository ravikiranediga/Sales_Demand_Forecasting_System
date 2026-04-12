"""
Sales Demand Forecasting System
===============================
A complete ML pipeline for sales forecasting with Streamlit browser visualization.

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sales Demand Forecasting",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# DATA GENERATION FUNCTION
# ============================================================

def generate_data():
    """Generate synthetic sales data with seasonality."""
    np.random.seed(42)
    
    start_date = pd.Timestamp('2023-01-01')
    end_date = pd.Timestamp('2025-03-31')
    dates = pd.date_range(start_date, end_date, freq='D')
    
    stores = ['Store_A', 'Store_B', 'Store_C']
    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Food & Beverages']
    regions = ['North', 'South', 'East', 'West']
    
    def generate_sales(date, store, category):
        base_sales = {
            'Electronics': 1500, 'Clothing': 800, 'Home & Garden': 600,
            'Sports': 400, 'Food & Beverages': 1200
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
    return df

# ============================================================
# FEATURE ENGINEERING FUNCTION
# ============================================================

def engineer_features(df):
    """Create time-based, lag, and rolling features."""
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Time features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Quarter'] = df['Date'].dt.quarter
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
    df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)
    
    # Cyclical features
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
    df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
    
    # Lag features
    df = df.sort_values(['Store', 'Product_Category', 'Date']).reset_index(drop=True)
    for lag in [7, 14, 21, 30]:
        df[f'Sales_Lag_{lag}'] = df.groupby(['Store', 'Product_Category'])['Sales'].shift(lag)
    
    # Rolling features
    for window in [7, 14, 30]:
        df[f'Sales_RollingMean_{window}'] = df.groupby(['Store', 'Product_Category'])['Sales'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f'Sales_RollingStd_{window}'] = df.groupby(['Store', 'Product_Category'])['Sales'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
        )
    
    # Encode categoricals
    le_store = LabelEncoder()
    le_category = LabelEncoder()
    le_region = LabelEncoder()
    df['Store_Encoded'] = le_store.fit_transform(df['Store'])
    df['Category_Encoded'] = le_category.fit_transform(df['Product_Category'])
    df['Region_Encoded'] = le_region.fit_transform(df['Region'])
    
    return df

# ============================================================
# MODEL TRAINING FUNCTION
# ============================================================

def train_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple models."""
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        results[name] = {
            'model': model,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }
    
    return results

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_sales_data():
    """Load or generate sales data."""
    if os.path.exists('data/sales_data.csv'):
        df = pd.read_csv('data/sales_data.csv')
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        df = generate_data()
        df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_data
def load_processed_data():
    """Load or create processed data with features."""
    if os.path.exists('data/processed_sales.csv'):
        df = pd.read_csv('data/processed_sales.csv')
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        df = load_sales_data()
        df = engineer_features(df)
        df_clean = df.dropna()
        df_clean.to_csv('data/processed_sales.csv', index=False)
    return df

@st.cache_resource
def load_model():
    """Load trained model."""
    if os.path.exists('models/best_model.pkl'):
        return joblib.load('models/best_model.pkl')
    return None

# ============================================================
# STREAMLIT APP
# ============================================================

def main():
    """Main Streamlit application."""
    
    st.title("📊 Sales Demand Forecasting Dashboard")
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["🏠 Home", "📈 EDA", "🔧 Model Training", "🔮 Forecasting", "📋 Report"]
    )
    
    # Load data
    df = load_sales_data()
    df_processed = load_processed_data()
    model = load_model()
    
    # ========== HOME PAGE ==========
    if page == "🏠 Home":
        st.header("Welcome to Sales Demand Forecasting")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Total Sales", f"${df['Sales'].sum():,.0f}")
        with col3:
            st.metric("Date Range", f"{df['Date'].min().date()} to {df['Date'].max().date()}")
        
        st.subheader("Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Sales by Store:**")
            st.dataframe(df.groupby('Store')['Sales'].sum())
        with col2:
            st.write("**Sales by Category:**")
            st.dataframe(df.groupby('Product_Category')['Sales'].sum().sort_values(ascending=False))
        
        st.info("👈 Use the sidebar to navigate through different sections!")
    
    # ========== EDA PAGE ==========
    elif page == "📈 EDA":
        st.header("Exploratory Data Analysis")
        
        st.subheader("Dataset Overview")
        st.dataframe(df.head())
        
        st.subheader("Sales by Store")
        store_sales = df.groupby('Store')['Sales'].sum()
        fig, ax = plt.subplots(figsize=(10, 5))
        store_sales.plot(kind='bar', ax=ax, color=['#2ecc71', '#3498db', '#e74c3c'])
        ax.set_title('Total Sales by Store')
        ax.set_ylabel('Sales ($)')
        st.pyplot(fig)
        
        st.subheader("Sales by Category")
        cat_sales = df.groupby('Product_Category')['Sales'].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        cat_sales.plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12'])
        ax.set_title('Total Sales by Category')
        ax.set_ylabel('Sales ($)')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
        
        st.subheader("Monthly Trend")
        monthly_sales = df.groupby(df['Date'].dt.to_period('M'))['Sales'].sum()
        fig, ax = plt.subplots(figsize=(12, 5))
        monthly_sales.plot(ax=ax, marker='o', linewidth=2, color='#3498db')
        ax.set_title('Monthly Sales Trend')
        ax.set_ylabel('Sales ($)')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        st.subheader("Seasonality")
        df['Month'] = df['Date'].dt.month
        monthly_avg = df.groupby('Month')['Sales'].mean()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(1, 13), monthly_avg.values)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(months)
        ax.set_title('Average Sales by Month')
        ax.set_ylabel('Average Sales ($)')
        st.pyplot(fig)
    
    # ========== MODEL TRAINING PAGE ==========
    elif page == "🔧 Model Training":
        st.header("Model Training & Evaluation")
        
        # Define features
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
        
        X = df_processed[feature_cols]
        y = df_processed['Sales']
        
        # Train-test split
        df_sorted = df_processed.sort_values('Date')
        split_idx = int(len(df_sorted) * 0.8)
        train_idx = df_sorted.index[:split_idx]
        test_idx = df_sorted.index[split_idx:]
        
        X_train = X.loc[train_idx]
        X_test = X.loc[test_idx]
        y_train = y.loc[train_idx]
        y_test = y.loc[test_idx]
        
        st.subheader("Training Data")
        st.write(f"Training samples: {len(X_train)}")
        st.write(f"Testing samples: {len(X_test)}")
        
        # Train models
        if st.button("Train Models", type="primary"):
            with st.spinner("Training models..."):
                results = train_models(X_train, X_test, y_train, y_test)
                
                # Save best model
                best_name = min(results, key=lambda x: results[x]['MAE'])
                best_model = results[best_name]['model']
                joblib.dump(best_model, 'models/best_model.pkl')
                
                st.success("Models trained successfully!")
                
                st.subheader("Model Comparison")
                for name, res in results.items():
                    with st.expander(f"{name}"):
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("R² Score", f"{res['R2']:.4f}")
                        col2.metric("MAE", f"${res['MAE']:,.2f}")
                        col3.metric("RMSE", f"${res['RMSE']:,.2f}")
                        col4.metric("MAPE", f"{res['MAPE']:.2f}%")
                
                st.subheader("Best Model Performance")
                best = results[best_name]
                st.success(f"**{best_name}** is the best model with R² = {best['R2']:.4f}")
        
        # Show feature importance if model exists
        if model is not None and hasattr(model, 'feature_importances_'):
            st.subheader("Feature Importance")
            fi = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            fi.head(15).plot(kind='barh', x='Feature', y='Importance', ax=ax, color='#3498db')
            ax.set_title('Top 15 Feature Importance')
            ax.set_xlabel('Importance')
            st.pyplot(fig)
    
    # ========== FORECASTING PAGE ==========
    elif page == "🔮 Forecasting":
        st.header("Sales Forecasting")
        
        if model is None:
            st.warning("⚠️ Please train the model first in 'Model Training' section!")
        else:
            # Generate forecast
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
            
            # Create features
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
            
            for col in feature_cols:
                if col not in future_df.columns:
                    future_df[col] = 0
            
            X_future = future_df[feature_cols].fillna(0)
            future_df['Predicted_Sales'] = model.predict(X_future)
            
            # Display results
            total_forecast = future_df['Predicted_Sales'].sum()
            daily_avg = future_df.groupby('Date')['Predicted_Sales'].sum().mean()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Forecast (90 days)", f"${total_forecast:,.0f}")
            col2.metric("Average Daily Sales", f"${daily_avg:,.0f}")
            col3.metric("Forecast Period", f"{future_dates[0].date()} to {future_dates[-1].date()}")
            
            st.subheader("Forecast by Store")
            store_forecast = future_df.groupby('Store')['Predicted_Sales'].sum()
            fig, ax = plt.subplots(figsize=(10, 5))
            store_forecast.plot(kind='bar', ax=ax, color=['#2ecc71', '#3498db', '#e74c3c'])
            ax.set_title('Forecast by Store')
            ax.set_ylabel('Predicted Sales ($)')
            st.pyplot(fig)
            
            st.subheader("Forecast by Category")
            cat_forecast = future_df.groupby('Product_Category')['Predicted_Sales'].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10, 5))
            cat_forecast.plot(kind='barh', ax=ax, color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12'])
            ax.set_title('Forecast by Category')
            ax.set_xlabel('Predicted Sales ($)')
            st.pyplot(fig)
            
            st.subheader("Daily Forecast Trend")
            daily_forecast = future_df.groupby('Date')['Predicted_Sales'].sum()
            fig, ax = plt.subplots(figsize=(14, 5))
            daily_forecast.plot(ax=ax, color='#e74c3c', linewidth=2)
            ax.set_title('Daily Sales Forecast (Next 90 Days)')
            ax.set_ylabel('Predicted Sales ($)')
            st.pyplot(fig)
            
            # Save forecast
            future_df.to_csv('output/forecast_results.csv', index=False)
            st.success("Forecast saved to output/forecast_results.csv")
    
    # ========== REPORT PAGE ==========
    elif page == "📋 Report":
        st.header("Business Report")
        
        if os.path.exists('output/model_evaluation.csv'):
            model_eval = pd.read_csv('output/model_evaluation.csv', index_col=0)
            st.subheader("Model Performance Summary")
            st.dataframe(model_eval)
        
        if os.path.exists('output/forecast_results.csv'):
            forecast = pd.read_csv('output/forecast_results.csv')
            forecast['Date'] = pd.to_datetime(forecast['Date'])
            
            st.subheader("Forecast Summary")
            total = forecast['Predicted_Sales'].sum()
            st.metric("Total Predicted Sales (90 days)", f"${total:,.0f}")
            
            st.write("**By Store:**")
            st.dataframe(forecast.groupby('Store')['Predicted_Sales'].sum())
            
            st.write("**By Category:**")
            st.dataframe(forecast.groupby('Product_Category')['Predicted_Sales'].sum().sort_values(ascending=False))
        
        st.subheader("Key Insights")
        st.info("""
        📌 **Seasonality Patterns:**
        - Peak: November, December (+40% - Holiday season)
        - High: June, July, August (+20% - Summer)
        - Low: January, February (-20% - Post-holiday)
        
        📌 **Business Recommendations:**
        - Increase stock 40% for November-December
        - Reduce January inventory by 20%
        - Maintain steady stock for Food & Beverages
        """)

# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":
    main()