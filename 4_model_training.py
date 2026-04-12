import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

print("=" * 60)
print("STEP 4: MODEL TRAINING & EVALUATION")
print("=" * 60)

df = pd.read_csv('data/processed_sales.csv')
df['Date'] = pd.to_datetime(df['Date'])
print(f"\nLoaded processed data: {len(df)} records")

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

X = df[feature_cols]
y = df['Sales']

print(f"\n[1] Features: {X.shape[1]}, Target: {len(y)}")

df_sorted = df.sort_values('Date')
split_idx = int(len(df_sorted) * 0.8)
train_idx = df_sorted.index[:split_idx]
test_idx = df_sorted.index[split_idx:]

X_train = X.loc[train_idx]
X_test = X.loc[test_idx]
y_train = y.loc[train_idx]
y_test = y.loc[test_idx]

print(f"\n[2] Train/Test Split (Time-based):")
print(f"    Training: {len(X_train)} records")
print(f"    Testing:  {len(X_test)} records")

print("\n[3] Training Models...")
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
    
    results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}
    print(f"    {name}:")
    print(f"      R2   = {r2:.4f}")
    print(f"      MAE  = ${mae:,.2f}")
    print(f"      RMSE = ${rmse:,.2f}")
    print(f"      MAPE = {mape:.2f}%")

best_name = min(results, key=lambda x: results[x]['MAE'])
best_model = models[best_name]

print(f"\n[4] Best Model: {best_name}")
print(f"    MAE: ${results[best_name]['MAE']:,.2f}")
print(f"    R2:  {results[best_name]['R2']:.4f}")

if hasattr(best_model, 'feature_importances_'):
    print("\n[5] Top 5 Feature Importance:")
    fi = pd.DataFrame({'Feature': feature_cols, 'Importance': best_model.feature_importances_})
    fi = fi.sort_values('Importance', ascending=False)
    for i, row in fi.head(5).iterrows():
        print(f"    {row['Feature']}: {row['Importance']:.4f}")

joblib.dump(best_model, 'models/best_model.pkl')
print(f"\n[6] Model saved to: models/best_model.pkl")

results_df = pd.DataFrame(results).T
results_df.to_csv('output/model_evaluation.csv')
print(f"    Evaluation results: output/model_evaluation.csv")

print("\nModel Training Complete!")