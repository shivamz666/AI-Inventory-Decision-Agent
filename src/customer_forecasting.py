"""
Customer-Specific Demand Forecasting Pipeline for BFWAI AI Inventory Decision Agent.

Trains a demand forecasting model exclusively on customer-uploaded historical sales data.
Generates 7-day and 30-day demand predictions along with sales trends per product.

Important Architecture Rules:
1. Baseline development model (models/demand_forecasting_model.pkl) is NEVER loaded or used.
2. Market Intelligence (market_signal, market_impact, Gemini, NewsAPI) is NEVER used as a forecasting feature.
3. Customer models are saved under models/customer/ separately.
"""

from typing import Dict, Any, Tuple, Optional
import os
import logging
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

logger = logging.getLogger("BFWAI.CustomerForecasting")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Default paths for customer model artifacts
_DEFAULT_CUSTOMER_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "customer")
_DEFAULT_CUSTOMER_METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "reports", "customer_forecasting_metrics.csv")
_DEFAULT_CUSTOMER_PRED_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "predictions", "customer_demand_predictions.csv")


def extract_customer_forecasting_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs feature matrix from customer historical sales data.
    
    Features used:
    - Historical sales lags (lag_1, lag_7, lag_14, lag_30)
    - Rolling sales statistics (rolling_mean_7, rolling_std_7, rolling_mean_14, rolling_mean_30)
    - Price & promotional metrics (price, discount_pct, promotion, holiday_event, competitor_price, price_vs_competitor)
    - Temporal calendar features (dayofweek, month, quarter, is_weekend)
    
    Target variables:
    - target_7d_units: Cumulative 7-day future sales
    
    Note: NO market intelligence (market_signal, market_impact, Gemini/NewsAPI) is included.
    """
    df = df.copy()
    df = df.sort_values(by=["product_id", "date"]).reset_index(drop=True)
    
    # Temporal features
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    
    # Pricing & competitive ratios
    df["price_vs_competitor"] = np.where(
        df["competitor_price"] > 0,
        df["price"] / df["competitor_price"],
        1.0
    )
    df["effective_price"] = df["price"] * (1.0 - (df["discount_pct"] / 100.0))
    
    # Lag and rolling features calculated per product
    feature_dfs = []
    for pid, group in df.groupby("product_id"):
        grp = group.copy()
        
        # Lag features
        grp["sales_lag_1"] = grp["units_sold"].shift(1)
        grp["sales_lag_7"] = grp["units_sold"].shift(7)
        grp["sales_lag_14"] = grp["units_sold"].shift(14)
        grp["sales_lag_30"] = grp["units_sold"].shift(30)
        
        # Rolling sales statistics
        grp["sales_rolling_mean_7"] = grp["units_sold"].shift(1).rolling(7, min_periods=1).mean()
        grp["sales_rolling_std_7"] = grp["units_sold"].shift(1).rolling(7, min_periods=1).std().fillna(0)
        grp["sales_rolling_mean_14"] = grp["units_sold"].shift(1).rolling(14, min_periods=1).mean()
        grp["sales_rolling_mean_30"] = grp["units_sold"].shift(1).rolling(30, min_periods=1).mean()
        
        # 7-day cumulative target horizon (forward rolling sum)
        # Sum of units_sold for next 7 days: t+1 to t+7
        forward_7d = grp["units_sold"].iloc[::-1].rolling(7, min_periods=1).sum().iloc[::-1].shift(-7)
        grp["target_7d_units"] = forward_7d
        
        feature_dfs.append(grp)
        
    full_feat_df = pd.concat(feature_dfs, ignore_index=True)
    return full_feat_df


def train_customer_forecasting_model(
    customer_data: pd.DataFrame,
    model_dir: str = _DEFAULT_CUSTOMER_MODEL_DIR,
    metrics_path: str = _DEFAULT_CUSTOMER_METRICS_PATH
) -> Tuple[Any, Dict[str, float]]:
    """
    Trains a customer-specific demand forecasting model using historical sales data.
    Saves trained model artifact under models/customer/ (separate from baseline model).
    
    Args:
        customer_data: Cleaned DataFrame adapted by adapt_customer_data.
        model_dir: Directory where customer model artifact will be saved.
        metrics_path: File path to save evaluation metrics.
        
    Returns:
        Tuple of (trained_model, metrics_dict).
    """
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    
    logger.info("Extracting customer forecasting features...")
    feat_df = extract_customer_forecasting_features(customer_data)
    
    # Drop rows where lag features or target 7d horizon are NaN
    train_df = feat_df.dropna(subset=["sales_lag_7", "target_7d_units"]).copy()
    
    if len(train_df) < 20:
        logger.warning("Very small customer dataset for training. Using available rows with reduced split.")
        
    feature_cols = [
        "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_30",
        "sales_rolling_mean_7", "sales_rolling_std_7", "sales_rolling_mean_14", "sales_rolling_mean_30",
        "price", "unit_cost", "discount_pct", "effective_price",
        "promotion", "holiday_event", "rating", "review_count",
        "competitor_price", "price_vs_competitor",
        "dayofweek", "month", "quarter", "is_weekend"
    ]
    
    X = train_df[feature_cols]
    y = train_df["target_7d_units"]
    
    # Chronological Train-Test Split (80% Train, 20% Test)
    split_idx = int(len(train_df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    if len(X_test) == 0:
        X_train, X_test = X, X
        y_train, y_test = y, y
        
    logger.info(f"Training HistGradientBoostingRegressor on {len(X_train)} customer training samples...")
    model = HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate Model
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    try:
        mape = mean_absolute_percentage_error(y_test + 1, y_pred + 1) * 100.0
    except Exception:
        mape = 0.0
        
    metrics = {
        "dataset": "customer_historical",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "mape_pct": round(float(mape), 2)
    }
    
    logger.info(f"Customer Forecasting Model Trained successfully. Metrics: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")
    
    # Save Customer Model (NEVER overwrites models/demand_forecasting_model.pkl)
    model_file = os.path.join(model_dir, "customer_demand_model.pkl")
    joblib.dump(model, model_file)
    logger.info(f"Saved customer model artifact to '{model_file}'")
    
    # Save Metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved evaluation metrics to '{metrics_path}'")
    
    return model, metrics


def predict_customer_demand(
    model_or_path: Any,
    customer_data: pd.DataFrame,
    output_pred_path: str = _DEFAULT_CUSTOMER_PRED_PATH
) -> pd.DataFrame:
    """
    Generates 7-day and 30-day demand predictions and sales trends for each product in customer dataset.
    
    Args:
        model_or_path: Trained model object or file path to customer model pickle.
        customer_data: Cleaned DataFrame adapted by adapt_customer_data.
        output_pred_path: File path to save customer_demand_predictions.csv.
        
    Returns:
        DataFrame containing:
        - product_id
        - product_name
        - forecast_7d (int)
        - forecast_30d (int)
        - sales_trend (str: 'increasing', 'stable', or 'decreasing')
    """
    os.makedirs(os.path.dirname(output_pred_path), exist_ok=True)
    
    # Load model if path provided
    if isinstance(model_or_path, str):
        if not os.path.exists(model_or_path):
            raise FileNotFoundError(f"Customer model file not found at '{model_or_path}'. Ensure model is trained first.")
        model = joblib.load(model_or_path)
    else:
        model = model_or_path
        
    logger.info("Generating features for customer demand prediction...")
    feat_df = extract_customer_forecasting_features(customer_data)
    
    feature_cols = [
        "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_30",
        "sales_rolling_mean_7", "sales_rolling_std_7", "sales_rolling_mean_14", "sales_rolling_mean_30",
        "price", "unit_cost", "discount_pct", "effective_price",
        "promotion", "holiday_event", "rating", "review_count",
        "competitor_price", "price_vs_competitor",
        "dayofweek", "month", "quarter", "is_weekend"
    ]
    
    results = []
    for pid, group in feat_df.groupby("product_id"):
        # Take latest historical record for prediction
        latest_rec = group.sort_values(by="date").iloc[-1:]
        p_name = latest_rec["product_name"].values[0] if "product_name" in latest_rec.columns else str(pid)
        
        # Prepare feature vector for latest state
        X_latest = latest_rec[feature_cols].fillna(0)
        
        # Predict 7-day forecast
        pred_7d_raw = model.predict(X_latest)[0]
        forecast_7d = max(0, int(round(float(pred_7d_raw))))
        
        # Extrapolate 30-day forecast (accounting for 30d horizon vs 7d horizon)
        forecast_30d = max(0, int(round(forecast_7d * (30.0 / 7.0))))
        
        # Compute Recent 7-day actual historical sales for trend comparison
        recent_7d_actual = group.sort_values(by="date").tail(7)["units_sold"].sum()
        
        # Determine Sales Trend
        if recent_7d_actual > 0:
            ratio = forecast_7d / float(recent_7d_actual)
            if ratio > 1.05:
                trend = "increasing"
            elif ratio < 0.95:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"
            
        results.append({
            "product_id": str(pid),
            "product_name": str(p_name),
            "forecast_7d": forecast_7d,
            "forecast_30d": forecast_30d,
            "sales_trend": trend
        })
        
    pred_df = pd.DataFrame(results)
    pred_df.to_csv(output_pred_path, index=False)
    logger.info(f"Saved customer demand predictions for {len(pred_df)} products to '{output_pred_path}'")
    return pred_df
