"""
Customer Data Adapter Module for BFWAI AI Inventory Decision Agent.

Loads customer-uploaded CSV files, validates required schema and data types,
strips/ignores customer-supplied market intelligence or forecast values,
handles missing data and duplicates, and sorts historical records.
"""

from typing import Optional
import os
import logging
import pandas as pd
import numpy as np

from src.customer_schema import (
    REQUIRED_COLUMNS,
    DISCARD_COLUMNS,
    COLUMN_TYPES,
    validate_customer_schema
)

logger = logging.getLogger("BFWAI.CustomerDataAdapter")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def adapt_customer_data(input_path: str) -> pd.DataFrame:
    """
    Loads, cleans, validates, and adapts a customer-uploaded inventory CSV.
    
    Args:
        input_path: Absolute or relative file path to customer CSV upload.
        
    Returns:
        Cleaned, validated, and sorted Pandas DataFrame ready for customer-specific forecasting.
        
    Raises:
        FileNotFoundError: If input file path does not exist.
        ValueError: If file is empty or missing required schema columns.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Customer dataset file not found: '{input_path}'")
        
    logger.info(f"Loading customer dataset from '{input_path}'...")
    
    try:
        df = pd.read_csv(input_path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Customer CSV file '{input_path}' is completely empty.")
    except Exception as e:
        raise ValueError(f"Failed to read CSV file '{input_path}': {e}")
        
    if df is None or df.empty:
        raise ValueError(f"Customer dataset loaded from '{input_path}' is empty.")
        
    # 1. Validate Schema Required Columns
    is_valid, errors = validate_customer_schema(df)
    if not is_valid:
        raise ValueError(f"Customer schema validation failed: {'; '.join(errors)}")
        
    # 2. Ignore / Strip Customer-Supplied Market Intelligence or Forecast Columns
    dropped_cols = [c for c in DISCARD_COLUMNS if c in df.columns]
    if dropped_cols:
        logger.info(f"Dropping customer-supplied market intelligence/forecast columns: {dropped_cols}")
        df = df.drop(columns=dropped_cols)
        
    # 3. Clean Critical Key Fields (Date & Product ID)
    df = df.dropna(subset=["date", "product_id"])
    df["product_id"] = df["product_id"].astype(str).str.strip()
    
    # Parse Date
    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception as e:
        logger.warning(f"Coercing unparseable dates: {e}")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        
    if df.empty:
        raise ValueError("Customer dataset contains no valid date records after date parsing.")
        
    # 4. Clean String Meta Fields
    for str_col in ["product_name", "category", "brand"]:
        if str_col in df.columns:
            df[str_col] = df[str_col].astype(str).str.strip()
            
    # 5. Convert and Clean Numeric Fields
    numeric_cols = [
        "units_sold", "revenue", "price", "unit_cost", "discount_pct",
        "promotion", "holiday_event", "rating", "review_count",
        "current_inventory", "reserved_inventory", "incoming_inventory",
        "lead_time_days", "minimum_order_quantity", "reorder_point",
        "competitor_price"
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    # Impute missing numeric values logically
    if "units_sold" in df.columns:
        df["units_sold"] = df["units_sold"].fillna(0).clip(lower=0)
    if "revenue" in df.columns:
        df["revenue"] = df["revenue"].fillna(df["units_sold"] * df["price"])
    if "discount_pct" in df.columns:
        df["discount_pct"] = df["discount_pct"].fillna(0.0).clip(0.0, 100.0)
    if "promotion" in df.columns:
        df["promotion"] = df["promotion"].fillna(0).astype(int)
    if "holiday_event" in df.columns:
        df["holiday_event"] = df["holiday_event"].fillna(0).astype(int)
    if "competitor_price" in df.columns:
        df["competitor_price"] = df["competitor_price"].fillna(df["price"])
        
    # Forward-fill / median-fill inventory and supplier specs per product
    for prod_id, grp_idx in df.groupby("product_id").groups.items():
        for inv_col in ["current_inventory", "reserved_inventory", "incoming_inventory",
                        "lead_time_days", "minimum_order_quantity", "reorder_point"]:
            if inv_col in df.columns:
                col_vals = df.loc[grp_idx, inv_col].ffill().bfill().fillna(0)
                df.loc[grp_idx, inv_col] = col_vals
                
    # 6. Deduplicate records based on (product_id, date)
    duplicate_count = df.duplicated(subset=["product_id", "date"]).sum()
    if duplicate_count > 0:
        logger.info(f"Removing {duplicate_count} duplicate records for (product_id, date).")
        df = df.drop_duplicates(subset=["product_id", "date"], keep="last")
        
    # 7. Sort Records chronologically by product_id and date
    df = df.sort_values(by=["product_id", "date"]).reset_index(drop=True)
    
    logger.info(f"Successfully adapted customer dataset: {len(df)} records across {df['product_id'].nunique()} products.")
    return df
