"""
Unit tests for Customer-Specific Demand Forecasting (src/customer_forecasting.py).
"""

import unittest
import os
import sys
import tempfile
import pandas as pd
from unittest.mock import patch

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.customer_data_adapter import adapt_customer_data
from src.customer_forecasting import (
    train_customer_forecasting_model,
    predict_customer_demand
)


class TestCustomerForecasting(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        # Generate 40 days of dummy customer sales history for P001 & P002
        dates = pd.date_range("2023-01-01", periods=40, freq="D").astype(str).tolist()
        data_p1 = {
            "date": dates,
            "product_id": ["P001"] * 40,
            "product_name": ["Wireless Gaming Mouse"] * 40,
            "category": ["Computer Accessories"] * 40,
            "brand": ["NovaTech"] * 40,
            "units_sold": [30 + (i % 10) for i in range(40)],
            "revenue": [(30 + (i % 10)) * 899.0 for i in range(40)],
            "price": [899.0] * 40,
            "unit_cost": [500.0] * 40,
            "discount_pct": [0.0] * 40,
            "promotion": [0] * 40,
            "holiday_event": [0] * 40,
            "rating": [4.4] * 40,
            "review_count": [500 + i for i in range(40)],
            "current_inventory": [1000 - i * 5 for i in range(40)],
            "reserved_inventory": [5] * 40,
            "incoming_inventory": [0] * 40,
            "lead_time_days": [7] * 40,
            "minimum_order_quantity": [50] * 40,
            "reorder_point": [250] * 40,
            "competitor_price": [900.0] * 40
        }
        data_p2 = {
            "date": dates,
            "product_id": ["P002"] * 40,
            "product_name": ["Ergonomic Keyboard"] * 40,
            "category": ["Computer Accessories"] * 40,
            "brand": ["NovaTech"] * 40,
            "units_sold": [15 + (i % 5) for i in range(40)],
            "revenue": [(15 + (i % 5)) * 1200.0 for i in range(40)],
            "price": [1200.0] * 40,
            "unit_cost": [700.0] * 40,
            "discount_pct": [0.0] * 40,
            "promotion": [0] * 40,
            "holiday_event": [0] * 40,
            "rating": [4.6] * 40,
            "review_count": [300 + i for i in range(40)],
            "current_inventory": [600 - i * 3 for i in range(40)],
            "reserved_inventory": [2] * 40,
            "incoming_inventory": [0] * 40,
            "lead_time_days": [10] * 40,
            "minimum_order_quantity": [30] * 40,
            "reorder_point": [150] * 40,
            "competitor_price": [1150.0] * 40
        }
        df_p1 = pd.DataFrame(data_p1)
        df_p2 = pd.DataFrame(data_p2)
        combined_df = pd.concat([df_p1, df_p2], ignore_index=True)
        
        self.csv_path = os.path.join(self.temp_dir.name, "customer_sample.csv")
        combined_df.to_csv(self.csv_path, index=False)
        self.customer_data = adapt_customer_data(self.csv_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_1_training_on_valid_customer_data(self):
        """1. Test training customer forecasting model on valid customer data."""
        model_dir = os.path.join(self.temp_dir.name, "models", "customer")
        metrics_path = os.path.join(self.temp_dir.name, "metrics.csv")
        
        model, metrics = train_customer_forecasting_model(
            self.customer_data,
            model_dir=model_dir,
            metrics_path=metrics_path
        )
        self.assertIsNotNone(model)
        self.assertIn("mae", metrics)
        self.assertIn("rmse", metrics)
        self.assertTrue(os.path.exists(metrics_path))

    def test_2_prediction_generation(self):
        """2. Test generating predictions (forecast_7d, forecast_30d, sales_trend)."""
        model_dir = os.path.join(self.temp_dir.name, "models", "customer")
        metrics_path = os.path.join(self.temp_dir.name, "metrics.csv")
        pred_path = os.path.join(self.temp_dir.name, "customer_demand_predictions.csv")
        
        model, _ = train_customer_forecasting_model(
            self.customer_data,
            model_dir=model_dir,
            metrics_path=metrics_path
        )
        
        pred_df = predict_customer_demand(model, self.customer_data, output_pred_path=pred_path)
        self.assertEqual(len(pred_df), 2)
        self.assertIn("forecast_7d", pred_df.columns)
        self.assertIn("forecast_30d", pred_df.columns)
        self.assertIn("sales_trend", pred_df.columns)
        self.assertTrue(all(pred_df["forecast_7d"] >= 0))
        self.assertTrue(all(pred_df["forecast_30d"] >= 0))
        self.assertTrue(all(t in ["increasing", "stable", "decreasing"] for t in pred_df["sales_trend"]))

    def test_3_model_saved_successfully(self):
        """3. Test that trained model is saved under customer model directory."""
        model_dir = os.path.join(self.temp_dir.name, "models", "customer")
        metrics_path = os.path.join(self.temp_dir.name, "metrics.csv")
        
        train_customer_forecasting_model(
            self.customer_data,
            model_dir=model_dir,
            metrics_path=metrics_path
        )
        model_file = os.path.join(model_dir, "customer_demand_model.pkl")
        self.assertTrue(os.path.exists(model_file))

    def test_4_forecast_output_generated(self):
        """4. Test that forecast output CSV is generated on disk."""
        model_dir = os.path.join(self.temp_dir.name, "models", "customer")
        metrics_path = os.path.join(self.temp_dir.name, "metrics.csv")
        pred_path = os.path.join(self.temp_dir.name, "customer_predictions.csv")
        
        model, _ = train_customer_forecasting_model(
            self.customer_data,
            model_dir=model_dir,
            metrics_path=metrics_path
        )
        predict_customer_demand(model, self.customer_data, output_pred_path=pred_path)
        self.assertTrue(os.path.exists(pred_path))

    @patch("joblib.load")
    def test_5_old_model_not_loaded(self, mock_joblib_load):
        """5. Test that old baseline model (models/demand_forecasting_model.pkl) is NOT loaded."""
        model_dir = os.path.join(self.temp_dir.name, "models", "customer")
        metrics_path = os.path.join(self.temp_dir.name, "metrics.csv")
        
        train_customer_forecasting_model(
            self.customer_data,
            model_dir=model_dir,
            metrics_path=metrics_path
        )
        
        # Verify joblib.load was never called for baseline 'models/demand_forecasting_model.pkl'
        for call_args in mock_joblib_load.call_args_list:
            arg = str(call_args[0][0])
            self.assertNotIn("demand_forecasting_model.pkl", arg)


if __name__ == "__main__":
    unittest.main()
