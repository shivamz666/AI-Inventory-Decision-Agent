"""
End-to-End Pipeline Execution Script for Customer-Data-Driven BFWAI AI Inventory Decision Agent (Person 1).

Source of Truth:
Raw Customer Dataset: BFWAI/data/raw/customer_inventory_upload.csv

Execution Flow:
1. Load & Adapt Raw Customer Dataset (data/raw/customer_inventory_upload.csv)
   ↓
2. Customer-Specific Model Training (src/customer_forecasting.py)
   ↓
3. Customer Demand Forecast Generation
   ↓
4. Gemini + NewsAPI Market Intelligence (src/market_intelligence.py)
   ↓
5. Final Single Handoff Output:
   - JSON: handoff/person2_handoff.json
   ↓
6. Clean Obsolete Prediction Outputs in outputs/predictions/
"""

import os
import sys
import logging
import json
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.dirname(__file__))

from src.customer_data_adapter import adapt_customer_data
from src.customer_forecasting import (
    train_customer_forecasting_model,
    predict_customer_demand
)
from src.product_intelligence import build_person2_handoff_json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BFWAI.PipelineRunner")


def clean_obsolete_prediction_files():
    """Removes old obsolete prediction/handoff files from outputs/predictions/."""
    obsolete_files = [
        "outputs/predictions/person2_handoff.json",
        "outputs/predictions/person2_handoff.csv",
        "outputs/predictions/person2_handoff_schema.json",
        "outputs/predictions/product_intelligence.csv",
        "outputs/predictions/product_intelligence_schema.json",
        "outputs/predictions/demand_predictions.csv",
        "outputs/predictions/customer_demand_predictions.csv",
        "outputs/predictions/market_intelligence_single_product_test.json"
    ]
    
    root_dir = os.path.dirname(__file__)
    removed_count = 0
    for rel_path in obsolete_files:
        full_path = os.path.join(root_dir, rel_path)
        if os.path.exists(full_path):
            try:
                os.remove(full_path)
                logger.info(f"Removed obsolete file: '{rel_path}'")
                removed_count += 1
            except Exception as e:
                logger.warning(f"Could not remove '{rel_path}': {e}")
                
    logger.info(f"Cleaned {removed_count} obsolete prediction files.")


def run_customer_pipeline(
    input_csv_path: str = "data/raw/customer_inventory_upload.csv"
):
    print("=" * 80)
    print("BFWAI AI Inventory Decision Agent - Person 1 Final Handoff Generator")
    print("=" * 80)
    
    # 1. Adapt Customer Raw CSV
    logger.info(f"Step 1: Adapting Raw Customer Input File: '{input_csv_path}'")
    customer_df = adapt_customer_data(input_csv_path)
    print(f"-> Adapted customer historical data: {len(customer_df)} rows across {customer_df['product_id'].nunique()} products.")
    
    # 2. Train Customer Forecasting Model
    logger.info("Step 2: Training Customer-Specific Demand Forecasting Model...")
    model, metrics = train_customer_forecasting_model(customer_df)
    print(f"-> Customer Model Trained. Metrics: MAE={metrics['mae']}, RMSE={metrics['rmse']}, MAPE={metrics['mape_pct']}%")
    
    # 3. Predict Customer Demand
    logger.info("Step 3: Generating Product Demand Predictions...")
    pred_df = predict_customer_demand(model, customer_df)
    print("-> Generated Customer Demand Predictions for all products.")
    
    # 4. Build Final Person 2 JSON Handoff Output at handoff/person2_handoff.json
    logger.info("Step 4: Generating Final Person 2 Handoff JSON: handoff/person2_handoff.json...")
    handoff_json_path = os.path.join(os.path.dirname(__file__), "handoff", "person2_handoff.json")
    handoff_json = build_person2_handoff_json(
        customer_df,
        pred_df,
        output_json_path=handoff_json_path,
        use_live_market_intel=True,
        source_file=input_csv_path
    )
    
    # 5. Clean obsolete old prediction outputs
    logger.info("Step 5: Cleaning obsolete old prediction outputs...")
    clean_obsolete_prediction_files()
    
    print("\n" + "=" * 80)
    print("FINAL PERSON 1 HANDOFF READY")
    print("=" * 80)
    print(f"File: handoff/person2_handoff.json ({len(handoff_json['products'])} products)")
    print("=" * 80)


if __name__ == "__main__":
    raw_file = os.path.join(os.path.dirname(__file__), "data", "raw", "customer_inventory_upload.csv")
    run_customer_pipeline(raw_file)
