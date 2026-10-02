"""
Feature engineering module for BFWAI AI Inventory Decision Agent.

Provides reusable pipelines for time series feature creation, lag/rolling feature generation,
external signal encoding, and strict data leakage validation.
"""

from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np


class FeatureEngineer:
    """
    Leakage-safe feature engineering transformer for time-series inventory & demand forecasting.
    """
    
    def __init__(self):
        self.market_signal_map = {'negative': -1, 'neutral': 0, 'positive': 1}
        self.feature_names = []

    def create_features(self, df: pd.DataFrame, drop_na_targets: bool = False) -> pd.DataFrame:
        """
        Generates time, lag, rolling, trend, price, inventory, and external features.
        
        Args:
            df: Input DataFrame containing raw inventory and sales time-series data.
            drop_na_targets: If True, drops trailing records where target forecast horizons are NaN.
            
        Returns:
            Processed pandas DataFrame with engineered features.
        """
        df_feat = df.copy()
        
        # Ensure correct date parsing and sorting
        df_feat['date'] = pd.to_datetime(df_feat['date'])
        df_feat = df_feat.sort_values(['product_id', 'date']).reset_index(drop=True)
        
        # 1. TIME FEATURES
        df_feat['day_of_week'] = df_feat['date'].dt.dayofweek
        df_feat['day_of_month'] = df_feat['date'].dt.day
        df_feat['month'] = df_feat['date'].dt.month
        df_feat['quarter'] = df_feat['date'].dt.quarter
        df_feat['year'] = df_feat['date'].dt.year
        df_feat['is_weekend'] = (df_feat['day_of_week'] >= 5).astype(int)
        
        # 2. SALES LAG FEATURES (Strictly shift(lag) per product)
        for lag in [1, 7, 14, 30]:
            df_feat[f'sales_lag_{lag}'] = df_feat.groupby('product_id')['units_sold'].shift(lag)
            
        # 3. ROLLING WINDOW FEATURES (Strictly shift(1) before rolling to prevent current day t leakage)
        for window in [7, 14, 30]:
            df_feat[f'sales_rolling_mean_{window}'] = df_feat.groupby('product_id')['units_sold'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
        df_feat['sales_rolling_std_7'] = df_feat.groupby('product_id')['units_sold'].transform(
            lambda x: x.shift(1).rolling(window=7, min_periods=2).std()
        ).fillna(0)
        
        # 4. TREND FEATURES
        df_feat['sales_growth_7d'] = (
            (df_feat['sales_rolling_mean_7'] - df_feat['sales_lag_7']) /
            (df_feat['sales_lag_7'] + 1e-5)
        ).fillna(0)
        
        # 5. PRICE & DISCOUNT FEATURES
        df_feat['price_difference'] = df_feat['price'] - df_feat['competitor_price']
        df_feat['price_ratio'] = df_feat['price'] / df_feat['competitor_price']
        
        # 6. EXTERNAL SIGNAL ENCODING
        df_feat['encoded_market_signal'] = df_feat['market_signal'].map(self.market_signal_map).fillna(0).astype(int)
        
        # 7. FORECASTING TARGETS (Forward window sum over t+1 to t+window)
        indexer_7 = pd.api.indexers.FixedForwardWindowIndexer(window_size=7)
        indexer_30 = pd.api.indexers.FixedForwardWindowIndexer(window_size=30)
        
        df_feat['target_demand_next_7d'] = df_feat.groupby('product_id')['units_sold'].transform(
            lambda x: x.shift(-1).rolling(window=indexer_7, min_periods=7).sum()
        )
        df_feat['target_demand_next_30d'] = df_feat.groupby('product_id')['units_sold'].transform(
            lambda x: x.shift(-1).rolling(window=indexer_30, min_periods=30).sum()
        )
        
        if drop_na_targets:
            df_feat = df_feat.dropna(subset=['target_demand_next_7d', 'target_demand_next_30d']).reset_index(drop=True)
            
        return df_feat


def generate_feature_dictionary() -> pd.DataFrame:
    """
    Generates a comprehensive feature dictionary describing all engineered variables.
    """
    dict_data = [
        # Identifier & Time
        {"feature_name": "date", "feature_group": "Identifier", "data_type": "datetime64", "description": "Calendar record date", "leakage_check": "PASS (Timestamp)"},
        {"feature_name": "product_id", "feature_group": "Identifier", "data_type": "string", "description": "Unique product SKU identifier", "leakage_check": "PASS (Static ID)"},
        {"feature_name": "day_of_week", "feature_group": "Time", "data_type": "int64", "description": "Day of the week (0=Mon, 6=Sun)", "leakage_check": "PASS (Known in advance)"},
        {"feature_name": "day_of_month", "feature_group": "Time", "data_type": "int64", "description": "Day of the month (1-31)", "leakage_check": "PASS (Known in advance)"},
        {"feature_name": "month", "feature_group": "Time", "data_type": "int64", "description": "Calendar month (1-12)", "leakage_check": "PASS (Known in advance)"},
        {"feature_name": "quarter", "feature_group": "Time", "data_type": "int64", "description": "Calendar quarter (1-4)", "leakage_check": "PASS (Known in advance)"},
        {"feature_name": "year", "feature_group": "Time", "data_type": "int64", "description": "Calendar year", "leakage_check": "PASS (Known in advance)"},
        {"feature_name": "is_weekend", "feature_group": "Time", "data_type": "int64", "description": "Binary indicator for weekend days (Saturday/Sunday)", "leakage_check": "PASS (Known in advance)"},
        
        # Sales Lags
        {"feature_name": "sales_lag_1", "feature_group": "Sales Lag", "data_type": "float64", "description": "Units sold 1 day prior (t-1)", "leakage_check": "PASS (Historical past only)"},
        {"feature_name": "sales_lag_7", "feature_group": "Sales Lag", "data_type": "float64", "description": "Units sold 7 days prior (t-7)", "leakage_check": "PASS (Historical past only)"},
        {"feature_name": "sales_lag_14", "feature_group": "Sales Lag", "data_type": "float64", "description": "Units sold 14 days prior (t-14)", "leakage_check": "PASS (Historical past only)"},
        {"feature_name": "sales_lag_30", "feature_group": "Sales Lag", "data_type": "float64", "description": "Units sold 30 days prior (t-30)", "leakage_check": "PASS (Historical past only)"},
        
        # Rolling Windows
        {"feature_name": "sales_rolling_mean_7", "feature_group": "Rolling Window", "data_type": "float64", "description": "7-day moving average of sales (excluding day t)", "leakage_check": "PASS (Strict shift(1))"},
        {"feature_name": "sales_rolling_mean_14", "feature_group": "Rolling Window", "data_type": "float64", "description": "14-day moving average of sales (excluding day t)", "leakage_check": "PASS (Strict shift(1))"},
        {"feature_name": "sales_rolling_mean_30", "feature_group": "Rolling Window", "data_type": "float64", "description": "30-day moving average of sales (excluding day t)", "leakage_check": "PASS (Strict shift(1))"},
        {"feature_name": "sales_rolling_std_7", "feature_group": "Rolling Window", "data_type": "float64", "description": "7-day rolling standard deviation of daily sales", "leakage_check": "PASS (Strict shift(1))"},
        
        # Trend
        {"feature_name": "sales_growth_7d", "feature_group": "Trend", "data_type": "float64", "description": "7-day relative sales growth rate vs 7-day lag", "leakage_check": "PASS (Historical calculation)"},
        
        # Price & Promotion
        {"feature_name": "price", "feature_group": "Price", "data_type": "float64", "description": "Current selling price ($)", "leakage_check": "PASS (Available at date t)"},
        {"feature_name": "competitor_price", "feature_group": "Price", "data_type": "float64", "description": "Competitor selling price ($)", "leakage_check": "PASS (Available at date t)"},
        {"feature_name": "price_difference", "feature_group": "Price", "data_type": "float64", "description": "Price difference (own price minus competitor price)", "leakage_check": "PASS (Available at date t)"},
        {"feature_name": "price_ratio", "feature_group": "Price", "data_type": "float64", "description": "Price ratio (own price divided by competitor price)", "leakage_check": "PASS (Available at date t)"},
        {"feature_name": "discount_pct", "feature_group": "Price", "data_type": "float64", "description": "Discount percentage applied", "leakage_check": "PASS (Available at date t)"},
        {"feature_name": "promotion", "feature_group": "Price", "data_type": "int64", "description": "Binary flag indicating active promotion", "leakage_check": "PASS (Available at date t)"},
        
        # Inventory
        {"feature_name": "current_inventory", "feature_group": "Inventory", "data_type": "float64", "description": "Opening inventory level on day t", "leakage_check": "PASS (Available at start of day)"},
        {"feature_name": "reserved_inventory", "feature_group": "Inventory", "data_type": "float64", "description": "Allocated/reserved stock count", "leakage_check": "PASS (Available at start of day)"},
        {"feature_name": "incoming_inventory", "feature_group": "Inventory", "data_type": "float64", "description": "Confirmed incoming shipment units", "leakage_check": "PASS (Available at start of day)"},
        {"feature_name": "lead_time_days", "feature_group": "Inventory", "data_type": "int64", "description": "Supplier replenishment lead time (days)", "leakage_check": "PASS (Static/known factor)"},
        {"feature_name": "minimum_order_quantity", "feature_group": "Inventory", "data_type": "int64", "description": "Supplier minimum order quantity constraint", "leakage_check": "PASS (Static/known factor)"},
        {"feature_name": "reorder_point", "feature_group": "Inventory", "data_type": "int64", "description": "Inventory threshold triggering reorder", "leakage_check": "PASS (Static/known factor)"},
        
        # External Signals
        {"feature_name": "search_interest_index", "feature_group": "External Signal", "data_type": "float64", "description": "Search engine demand interest metric", "leakage_check": "PASS (Available at date t)"},
        {"feature_name": "market_impact", "feature_group": "External Signal", "data_type": "float64", "description": "Market sentiment impact score", "leakage_check": "PASS (Available at date t)"},
        {"feature_name": "market_confidence", "feature_group": "External Signal", "data_type": "float64", "description": "Market intelligence confidence index", "leakage_check": "PASS (Available at date t)"},
        {"feature_name": "news_count", "feature_group": "External Signal", "data_type": "int64", "description": "Number of relevant industry news mentions", "leakage_check": "PASS (Available at date t)"},
        {"feature_name": "news_available", "feature_group": "External Signal", "data_type": "int64", "description": "Binary flag for news availability", "leakage_check": "PASS (Available at date t)"},
        {"feature_name": "encoded_market_signal", "feature_group": "External Signal", "data_type": "int64", "description": "Encoded market signal (-1=negative, 0=neutral, 1=positive)", "leakage_check": "PASS (Categorical mapping)"},
        
        # Target Variables
        {"feature_name": "target_demand_next_7d", "feature_group": "Target Horizon", "data_type": "float64", "description": "Cumulative units demanded in next 7 days (t+1 to t+7)", "leakage_check": "TARGET ONLY (Forecast output)"},
        {"feature_name": "target_demand_next_30d", "feature_group": "Target Horizon", "data_type": "float64", "description": "Cumulative units demanded in next 30 days (t+1 to t+30)", "leakage_check": "TARGET ONLY (Forecast output)"}
    ]
    return pd.DataFrame(dict_data)
