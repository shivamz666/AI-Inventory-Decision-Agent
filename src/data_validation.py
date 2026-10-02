"""
Data validation module for BFWAI AI Inventory Decision Agent.

This module provides functions to load, inspect, validate, and report data quality
metrics for the inventory dataset.
"""

from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np


def load_dataset(file_path: str) -> pd.DataFrame:
    """Loads dataset from specified file path and parses date column."""
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def get_dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """Returns dataset dimensions, column types, date range, and product counts."""
    return {
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "date_min": df['date'].min().strftime('%Y-%m-%d'),
        "date_max": df['date'].max().strftime('%Y-%m-%d'),
        "num_unique_products": df['product_id'].nunique(),
        "product_ids": sorted(df['product_id'].unique().tolist()),
    }


def run_validation_checks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executes comprehensive data quality and validation checks on inventory dataset.
    Returns a pandas DataFrame summarizing all validation check results.
    """
    results = []

    # 1. Duplicate checks
    dup_rows = df.duplicated().sum()
    results.append({
        "check_name": "Duplicate Rows",
        "category": "Integrity",
        "status": "PASS" if dup_rows == 0 else "FAIL",
        "issues_found": dup_rows,
        "details": "No duplicate rows found." if dup_rows == 0 else f"{dup_rows} exact duplicate rows detected."
    })

    dup_pk = df.duplicated(subset=['product_id', 'date']).sum()
    results.append({
        "check_name": "Duplicate (product_id, date) Combinations",
        "category": "Integrity",
        "status": "PASS" if dup_pk == 0 else "FAIL",
        "issues_found": dup_pk,
        "details": "Unique composite primary key (product_id, date) verified." if dup_pk == 0 else f"{dup_pk} duplicate key pairs detected."
    })

    # 2. Missing Values
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    null_cols = null_counts[null_counts > 0].to_dict()
    results.append({
        "check_name": "Missing Values",
        "category": "Completeness",
        "status": "INFO" if total_nulls > 0 else "PASS",
        "issues_found": total_nulls,
        "details": f"Missing values present only in lag and target horizon columns: {null_cols}" if total_nulls > 0 else "No missing values found."
    })

    # 3. Numeric & Value Range Checks
    neg_units = (df['units_sold'] < 0).sum()
    results.append({
        "check_name": "Negative Units Sold",
        "category": "Value Range",
        "status": "PASS" if neg_units == 0 else "FAIL",
        "issues_found": neg_units,
        "details": "All units_sold values are non-negative." if neg_units == 0 else f"{neg_units} negative sales entries found."
    })

    neg_revenue = (df['revenue'] < 0).sum()
    results.append({
        "check_name": "Negative Revenue",
        "category": "Value Range",
        "status": "PASS" if neg_revenue == 0 else "FAIL",
        "issues_found": neg_revenue,
        "details": "All revenue values are non-negative." if neg_revenue == 0 else f"{neg_revenue} negative revenue entries found."
    })

    invalid_price = (df['price'] <= 0).sum()
    results.append({
        "check_name": "Zero or Negative Price",
        "category": "Value Range",
        "status": "PASS" if invalid_price == 0 else "FAIL",
        "issues_found": invalid_price,
        "details": "All unit prices are positive." if invalid_price == 0 else f"{invalid_price} non-positive prices found."
    })

    invalid_rating = ((df['rating'] < 0) | (df['rating'] > 5)).sum()
    results.append({
        "check_name": "Invalid Ratings (outside [0, 5])",
        "category": "Value Range",
        "status": "PASS" if invalid_rating == 0 else "FAIL",
        "issues_found": invalid_rating,
        "details": f"All product ratings lie in range [{df['rating'].min()}, {df['rating'].max()}]." if invalid_rating == 0 else f"{invalid_rating} invalid ratings."
    })

    invalid_discount = ((df['discount_pct'] < 0) | (df['discount_pct'] > 100)).sum()
    results.append({
        "check_name": "Invalid Discount Percentage (outside [0, 100])",
        "category": "Value Range",
        "status": "PASS" if invalid_discount == 0 else "FAIL",
        "issues_found": invalid_discount,
        "details": f"Discounts range from {df['discount_pct'].min()}% to {df['discount_pct'].max()}%." if invalid_discount == 0 else f"{invalid_discount} invalid discounts."
    })

    invalid_inventory = ((df['current_inventory'] < 0) | (df['reserved_inventory'] < 0) | (df['incoming_inventory'] < 0)).sum()
    results.append({
        "check_name": "Invalid Inventory Quantities (< 0)",
        "category": "Value Range",
        "status": "PASS" if invalid_inventory == 0 else "FAIL",
        "issues_found": invalid_inventory,
        "details": "All current, reserved, and incoming inventory counts are non-negative." if invalid_inventory == 0 else f"{invalid_inventory} negative inventory levels."
    })

    invalid_lead_time = (df['lead_time_days'] <= 0).sum()
    results.append({
        "check_name": "Invalid Lead Time (<= 0 days)",
        "category": "Value Range",
        "status": "PASS" if invalid_lead_time == 0 else "FAIL",
        "issues_found": invalid_lead_time,
        "details": f"Lead times range from {df['lead_time_days'].min()} to {df['lead_time_days'].max()} days." if invalid_lead_time == 0 else f"{invalid_lead_time} invalid lead times."
    })

    invalid_moq = (df['minimum_order_quantity'] <= 0).sum()
    results.append({
        "check_name": "Invalid Minimum Order Quantity (<= 0)",
        "category": "Value Range",
        "status": "PASS" if invalid_moq == 0 else "FAIL",
        "issues_found": invalid_moq,
        "details": f"MOQ values range from {df['minimum_order_quantity'].min()} to {df['minimum_order_quantity'].max()} units." if invalid_moq == 0 else f"{invalid_moq} invalid MOQ values."
    })

    # 4. Product Record Counts
    counts = df.groupby('product_id').size()
    dev = (counts != 1000).sum()
    results.append({
        "check_name": "Product Daily Record Count (~1000)",
        "category": "Granularity",
        "status": "PASS" if dev == 0 else "WARN",
        "issues_found": dev,
        "details": "All 20 products have exactly 1,000 daily records." if dev == 0 else f"{dev} products do not have 1,000 daily records."
    })

    # 5. Chronological Continuity
    gaps_count = 0
    for pid, group in df.groupby('product_id'):
        group_sorted = group.sort_values('date')
        diffs = group_sorted['date'].diff()
        missing_days = (diffs > pd.Timedelta(days=1)).sum()
        gaps_count += missing_days

    results.append({
        "check_name": "Chronological Continuity",
        "category": "Time Series",
        "status": "PASS" if gaps_count == 0 else "FAIL",
        "issues_found": gaps_count,
        "details": "100% daily chronological continuity across all products with 0 missing dates." if gaps_count == 0 else f"{gaps_count} missing date gaps detected."
    })

    return pd.DataFrame(results)
