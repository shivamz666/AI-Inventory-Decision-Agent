"""
Demand forecasting module for BFWAI AI Inventory Decision Agent.

Provides baseline models, ML forecasting models (HistGradientBoostingRegressor / RandomForestRegressor),
chronological time-series splitting, evaluation metric utilities, and model persistence functions.
"""

from typing import Tuple, Dict, Any, List
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """
    Computes MAE, RMSE, and MAPE safely for target predictions.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    
    # Safe MAPE calculation
    mask = y_true_arr != 0
    if not np.any(mask):
        mape = 0.0
    else:
        mape = float(np.mean(np.abs((y_true_arr[mask] - y_pred_arr[mask]) / y_true_arr[mask])) * 100)
        
    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4)
    }


class DemandForecaster:
    """
    Time-series Demand Forecasting System supporting chronological train/val/test splitting,
    baseline estimation, GBDT training, and artifact persistence.
    """
    
    def __init__(self, target_col: str = 'target_demand_next_7d', random_state: int = 42):
        self.target_col = target_col
        self.random_state = random_state
        self.model = None
        self.feature_cols = []
        
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Returns feature column names excluding identifiers, raw text, and target variables."""
        exclude_cols = [
            'date', 'product_id', 'product_name', 'category', 'brand', 'market_signal',
            'target_demand_next_7d', 'target_demand_next_30d', 'units_sold', 'revenue'
        ]
        return [c for c in df.columns if c not in exclude_cols]

    def chronological_split(self, df: pd.DataFrame, train_pct: float = 0.70, val_pct: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits dataset chronologically into train, validation, and test subsets based on date.
        """
        df_sorted = df.dropna(subset=[self.target_col]).sort_values(['date', 'product_id']).reset_index(drop=True)
        unique_dates = sorted(df_sorted['date'].unique())
        n_dates = len(unique_dates)
        
        train_end_idx = int(n_dates * train_pct)
        val_end_idx = int(n_dates * (train_pct + val_pct))
        
        train_dates = unique_dates[:train_end_idx]
        val_dates = unique_dates[train_end_idx:val_end_idx]
        test_dates = unique_dates[val_end_idx:]
        
        train_df = df_sorted[df_sorted['date'].isin(train_dates)].copy()
        val_df = df_sorted[df_sorted['date'].isin(val_dates)].copy()
        test_df = df_sorted[df_sorted['date'].isin(test_dates)].copy()
        
        return train_df, val_df, test_df

    def train_baseline(self, df: pd.DataFrame) -> pd.Series:
        """
        Baseline prediction: 7-day rolling sales mean * 7.
        """
        return (df['sales_rolling_mean_7'] * 7).fillna(0)

    def train_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> HistGradientBoostingRegressor:
        """
        Trains HistGradientBoostingRegressor on training set.
        """
        self.feature_cols = self.get_feature_columns(train_df)
        X_train, y_train = train_df[self.feature_cols], train_df[self.target_col]
        
        self.model = HistGradientBoostingRegressor(
            random_state=self.random_state,
            max_iter=250,
            learning_rate=0.05,
            min_samples_leaf=20,
            l2_regularization=0.1
        )
        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generates ML model predictions."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        X = df[self.feature_cols]
        return self.model.predict(X)

    def evaluate_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluates Baseline and ML model across train, validation, and test splits.
        """
        results = []
        splits = [('Train', train_df), ('Validation', val_df), ('Test', test_df)]
        
        for name, split_df in splits:
            y_true = split_df[self.target_col]
            
            # Baseline
            b_pred = self.train_baseline(split_df)
            b_metrics = calculate_metrics(y_true, b_pred)
            results.append({
                'model': 'Baseline (7D Moving Avg)',
                'split': name,
                **b_metrics
            })
            
            # ML Model
            ml_pred = self.predict(split_df)
            ml_metrics = calculate_metrics(y_true, ml_pred)
            results.append({
                'model': 'HistGradientBoosting ML Model',
                'split': name,
                **ml_metrics
            })
            
        return pd.DataFrame(results)

    def compute_product_performance(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes product-level evaluation metrics on test set.
        """
        prod_metrics = []
        test_df = test_df.copy()
        test_df['baseline_pred'] = self.train_baseline(test_df)
        test_df['ml_pred'] = self.predict(test_df)
        
        for pid, group in test_df.groupby('product_id'):
            y_true = group[self.target_col]
            b_m = calculate_metrics(y_true, group['baseline_pred'])
            ml_m = calculate_metrics(y_true, group['ml_pred'])
            
            prod_metrics.append({
                'product_id': pid,
                'product_name': group['product_name'].iloc[0] if 'product_name' in group.columns else pid,
                'baseline_mae': b_m['MAE'],
                'ml_mae': ml_m['MAE'],
                'baseline_rmse': b_m['RMSE'],
                'ml_rmse': ml_m['RMSE'],
                'baseline_mape': b_m['MAPE'],
                'ml_mape': ml_m['MAPE'],
                'mae_improvement_pct': round((b_m['MAE'] - ml_m['MAE']) / b_m['MAE'] * 100, 2)
            })
            
        return pd.DataFrame(prod_metrics).sort_values('ml_mae').reset_index(drop=True)

    def save_model(self, file_path: str):
        """Saves trained model and feature metadata using joblib."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'target_col': self.target_col,
            'feature_cols': self.feature_cols
        }, file_path)

    @classmethod
    def load_model(cls, file_path: str):
        """Loads trained model from file."""
        data = joblib.load(file_path)
        forecaster = cls(target_col=data['target_col'])
        forecaster.model = data['model']
        forecaster.feature_cols = data['feature_cols']
        return forecaster
