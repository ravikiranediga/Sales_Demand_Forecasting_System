# Sales & Demand Forecasting System

A machine learning-based sales forecasting system with interactive browser dashboard. Built with Python, scikit-learn, and Streamlit.

## Features

- **Streamlit Dashboard** - Interactive browser-based visualizations
- **Data Generation** - Creates realistic sales data with seasonality patterns
- **Exploratory Data Analysis (EDA)** - Interactive charts and data exploration
- **Feature Engineering** - Creates time-based, lag, and rolling features
- **Model Training** - Compares 4 ML models (Linear, Ridge, Random Forest, Gradient Boosting)
- **Forecasting** - Generates 90-day sales predictions
- **Business Report** - Summary of key insights and recommendations

## Project Structure

```
├── app.py                 # Main Streamlit dashboard (ALL IN ONE)
├── requirements.txt       # Dependencies
├── README.md             # This file
├── data/                  # Data files
│   ├── sales_data.csv
│   └── processed_sales.csv
├── models/                # Trained models
│   └── best_model.pkl
├── output/                # Results
│   ├── forecast_results.csv
│   └── model_evaluation.csv
└── visualizations/        # Static charts (optional)
```

## Installation

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
pip install streamlit
```

## How to Run

```bash
# Run the Streamlit dashboard
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Dashboard Navigation:
- **🏠 Home** - Overview, quick stats, dataset summary
- **📈 EDA** - Interactive charts (Sales by Store, Category, Monthly Trend, Seasonality)
- **🔧 Model Training** - Train 4 models, compare metrics, feature importance
- **🔮 Forecasting** - 90-day predictions with visualizations
- **📋 Report** - Business insights and recommendations

## Results

| Model | R² Score | MAE ($) | MAPE (%) |
|-------|----------|---------|----------|
| Linear Regression | 0.7980 | 172.31 | 17.84 |
| Ridge Regression | 0.8442 | 165.19 | 17.14 |
| Random Forest | 0.8913 | 142.33 | 14.14 |
| **Gradient Boosting** | **0.8958** | **137.86** | **13.64** |

### Key Insights
- **Best Model:** Gradient Boosting (R² = 89.58%)
- **Top Feature:** 7-day rolling average (66.76% importance)
- **90-Day Forecast:** ~$417,543 total predicted sales

## Seasonality Patterns

- **Peak:** November, December (+40% - Holiday season)
- **High:** June, July, August (+20% - Summer)
- **Low:** January, February (-20% - Post-holiday)

## Business Applications

- Inventory planning and management
- Staffing optimization
- Cash flow planning
- Seasonal marketing campaigns

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- joblib
- streamlit

## License

MIT License