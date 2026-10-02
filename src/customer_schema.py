"""
Customer Data Schema Definition for BFWAI AI Inventory Decision Agent.

Defines required columns, data types, forbidden/ignored columns, and schema validation rules
for customer-uploaded historical inventory and sales CSV datasets.
"""

from typing import List, Dict, Tuple
import pandas as pd

# Required schema columns that MUST be present in customer upload
REQUIRED_COLUMNS: List[str] = [
    "date",
    "product_id",
    "product_name",
    "category",
    "brand",
    "units_sold",
    "revenue",
    "price",
    "unit_cost",
    "discount_pct",
    "promotion",
    "holiday_event",
    "rating",
    "review_count",
    "current_inventory",
    "reserved_inventory",
    "incoming_inventory",
    "lead_time_days",
    "minimum_order_quantity",
    "reorder_point",
    "competitor_price"
]

# Market intelligence or forecast fields supplied by customer that MUST BE IGNORED / DROPPED
DISCARD_COLUMNS: List[str] = [
    "market_signal",
    "market_impact",
    "market_confidence",
    "market_relevance",
    "evidence_strength",
    "news_available",
    "news_count",
    "search_interest_index",
    "demand_forecast",
    "forecast_7d",
    "forecast_30d",
    "inventory_decision",
    "recommended_order_quantity"
]

# Target data types for parsing and validation
COLUMN_TYPES: Dict[str, str] = {
    "date": "datetime",
    "product_id": "string",
    "product_name": "string",
    "category": "string",
    "brand": "string",
    "units_sold": "int",
    "revenue": "float",
    "price": "float",
    "unit_cost": "float",
    "discount_pct": "float",
    "promotion": "int",
    "holiday_event": "int",
    "rating": "float",
    "review_count": "int",
    "current_inventory": "int",
    "reserved_inventory": "int",
    "incoming_inventory": "int",
    "lead_time_days": "int",
    "minimum_order_quantity": "int",
    "reorder_point": "int",
    "competitor_price": "float"
}


def validate_customer_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validates a customer-uploaded DataFrame against the required customer schema.
    
    Args:
        df: Pandas DataFrame loaded from customer upload CSV.
        
    Returns:
        Tuple of (is_valid: bool, errors: List[str])
    """
    errors: List[str] = []
    
    if df is None or df.empty:
        return False, ["Customer dataset is empty or None."]
        
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns in customer dataset: {missing_cols}")
        
    return len(errors) == 0, errors
