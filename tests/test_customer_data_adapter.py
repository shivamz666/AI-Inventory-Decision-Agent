"""
Unit tests for Customer Data Adapter and Schema Validation (src/customer_data_adapter.py & src/customer_schema.py).
"""

import unittest
import os
import sys
import tempfile
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.customer_schema import REQUIRED_COLUMNS, validate_customer_schema
from src.customer_data_adapter import adapt_customer_data


class TestCustomerDataAdapter(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.valid_data = {
            "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "product_id": ["P001", "P001", "P001"],
            "product_name": ["Wireless Gaming Mouse", "Wireless Gaming Mouse", "Wireless Gaming Mouse"],
            "category": ["Computer Accessories", "Computer Accessories", "Computer Accessories"],
            "brand": ["NovaTech", "NovaTech", "NovaTech"],
            "units_sold": [38, 41, 35],
            "revenue": [34162.0, 36859.0, 31465.0],
            "price": [899.0, 899.0, 899.0],
            "unit_cost": [500.0, 500.0, 500.0],
            "discount_pct": [0.0, 0.0, 0.0],
            "promotion": [0, 0, 0],
            "holiday_event": [0, 0, 0],
            "rating": [4.42, 4.41, 4.42],
            "review_count": [505, 555, 612],
            "current_inventory": [836, 798, 757],
            "reserved_inventory": [4, 2, 2],
            "incoming_inventory": [0, 0, 0],
            "lead_time_days": [7, 7, 7],
            "minimum_order_quantity": [50, 50, 50],
            "reorder_point": [456, 456, 456],
            "competitor_price": [905.9, 964.94, 959.9]
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_temp_csv(self, data_dict, filename="test_customer.csv"):
        path = os.path.join(self.temp_dir.name, filename)
        df = pd.DataFrame(data_dict)
        df.to_csv(path, index=False)
        return path

    def test_1_valid_customer_file(self):
        """1. Test that valid customer CSV is adapted properly."""
        csv_path = self._create_temp_csv(self.valid_data)
        df = adapt_customer_data(csv_path)
        self.assertEqual(len(df), 3)
        self.assertIn("product_id", df.columns)
        self.assertEqual(df.iloc[0]["product_id"], "P001")
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["date"]))

    def test_2_missing_required_column(self):
        """2. Test that missing required columns raise ValueError."""
        data = self.valid_data.copy()
        del data["units_sold"]
        csv_path = self._create_temp_csv(data)
        with self.assertRaises(ValueError) as ctx:
            adapt_customer_data(csv_path)
        self.assertIn("Missing required columns", str(ctx.exception))

    def test_3_invalid_data_type(self):
        """3. Test parsing and coercion of unparseable numbers or dates."""
        data = self.valid_data.copy()
        data["units_sold"] = ["invalid_number", "41", "35"]
        csv_path = self._create_temp_csv(data)
        df = adapt_customer_data(csv_path)
        # Invalid number should be coerced and filled to 0
        self.assertEqual(df.iloc[0]["units_sold"], 0)

    def test_4_empty_file(self):
        """4. Test that empty file raises ValueError."""
        empty_path = os.path.join(self.temp_dir.name, "empty.csv")
        with open(empty_path, "w") as f:
            f.write("")
        with self.assertRaises(ValueError):
            adapt_customer_data(empty_path)

    def test_5_duplicate_records(self):
        """5. Test that duplicate records on (product_id, date) are deduplicated."""
        data = self.valid_data.copy()
        # Duplicate row 1
        data["date"].append("2023-01-01")
        for k in data:
            if k != "date":
                data[k].append(data[k][0])
        csv_path = self._create_temp_csv(data)
        df = adapt_customer_data(csv_path)
        self.assertEqual(len(df), 3)  # Duplicate removed

    def test_6_missing_values(self):
        """6. Test that NaNs in numerical columns are imputed cleanly."""
        data = self.valid_data.copy()
        data["discount_pct"] = [None, 5.0, None]
        csv_path = self._create_temp_csv(data)
        df = adapt_customer_data(csv_path)
        self.assertEqual(df.iloc[0]["discount_pct"], 0.0)

    def test_7_extra_columns(self):
        """7. Test that extra columns provided by customer do not crash adapter."""
        data = self.valid_data.copy()
        data["custom_notes"] = ["Note 1", "Note 2", "Note 3"]
        data["warehouse_loc"] = ["A1", "A2", "A3"]
        csv_path = self._create_temp_csv(data)
        df = adapt_customer_data(csv_path)
        self.assertEqual(len(df), 3)

    def test_8_market_signal_supplied_by_customer(self):
        """8. Test that customer-provided market_signal is ignored and dropped."""
        data = self.valid_data.copy()
        data["market_signal"] = ["positive", "positive", "negative"]
        csv_path = self._create_temp_csv(data)
        df = adapt_customer_data(csv_path)
        self.assertNotIn("market_signal", df.columns)

    def test_9_market_impact_supplied_by_customer(self):
        """9. Test that customer-provided market_impact is ignored and dropped."""
        data = self.valid_data.copy()
        data["market_impact"] = [0.95, 0.85, 0.75]
        csv_path = self._create_temp_csv(data)
        df = adapt_customer_data(csv_path)
        self.assertNotIn("market_impact", df.columns)

    def test_10_market_confidence_supplied_by_customer(self):
        """10. Test that customer-provided market_confidence is ignored and dropped."""
        data = self.valid_data.copy()
        data["market_confidence"] = [0.99, 0.99, 0.99]
        csv_path = self._create_temp_csv(data)
        df = adapt_customer_data(csv_path)
        self.assertNotIn("market_confidence", df.columns)


if __name__ == "__main__":
    unittest.main()
