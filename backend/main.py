"""
FastAPI Backend Application for Person 2: AI Inventory Decision Agent.

Integrates:
- Person 1's handoff (handoff/person2_handoff.json)
- Inventory Analysis
- Deterministic Multi-Signal Decision Engine
- LLM / Deterministic Explanation Agent
- Alternative Product Re-Allocation Agent
- Human-in-the-Loop Approvals Workflow
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import uuid

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    ProductHandoff,
    HandoffContainer,
    ProductInput,
    DecisionResult,
    FullAnalysisResponse,
    AlternativeProduct,
    ApprovalRequest,
    ApprovalRecord
)
from backend.inventory_analysis import analyze_inventory
from backend.decision_engine import evaluate_decision
from backend.explanation_agent import generate_llm_explanation, generate_deterministic_explanation
from backend.alternative_agent import find_alternative_products

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("BFWAI.Person2Backend")

from contextlib import asynccontextmanager

HANDOFF_FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "handoff", "person2_handoff.json")

# In-memory storage for human approval actions
_APPROVAL_RECORDS: Dict[str, ApprovalRecord] = {}
_PRODUCTS_CACHE: Dict[str, ProductHandoff] = {}


def load_handoff_data() -> Dict[str, ProductHandoff]:
    """Loads and validates the handoff file from Person 1."""
    if not os.path.exists(HANDOFF_FILE_PATH):
        logger.error(f"Handoff file not found at: {HANDOFF_FILE_PATH}")
        return {}

    try:
        with open(HANDOFF_FILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        products_dict: Dict[str, ProductHandoff] = {}
        for p_json in data.get("products", []):
            try:
                p_obj = ProductHandoff.model_validate(p_json)
                products_dict[p_obj.product_id] = p_obj
            except Exception as e:
                logger.warning(f"Failed to validate product item: {e}")
        return products_dict
    except Exception as e:
        logger.error(f"Failed to parse handoff JSON: {e}")
        return {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _PRODUCTS_CACHE
    _PRODUCTS_CACHE = load_handoff_data()
    logger.info(f"Loaded {len(_PRODUCTS_CACHE)} products from Person 1 handoff.")
    yield


app = FastAPI(
    title="AI Inventory Decision Agent (Person 2)",
    description="Operational Decision Support Engine for Inventory Management with Deterministic Rules, LLM Explanations, and Human-in-the-Loop Approval.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for Frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint confirming API status and handoff availability."""
    global _PRODUCTS_CACHE
    handoff_exists = os.path.exists(HANDOFF_FILE_PATH)
    if not _PRODUCTS_CACHE and handoff_exists:
        _PRODUCTS_CACHE = load_handoff_data()
    return {
        "status": "healthy",
        "service": "AI Inventory Decision Agent (Person 2)",
        "handoff_loaded": len(_PRODUCTS_CACHE) > 0,
        "handoff_path": HANDOFF_FILE_PATH,
        "handoff_exists": handoff_exists,
        "total_products": len(_PRODUCTS_CACHE),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/products", tags=["Inventory & Decisions"])
def list_products(
    category: Optional[str] = Query(None, description="Filter by category"),
    decision_filter: Optional[str] = Query(None, description="Filter by decision: INCREASE, MAINTAIN, REDUCE"),
    risk_filter: Optional[str] = Query(None, description="Filter by risk: LOW, MEDIUM, HIGH")
):
    """
    Returns all products from the Person 1 handoff, enriched with real-time evaluated
    inventory metrics, stockout risks, decisions, and approval status.
    """
    global _PRODUCTS_CACHE
    if not _PRODUCTS_CACHE:
        _PRODUCTS_CACHE = load_handoff_data()

    results = []
    counts = {"total": 0, "increase": 0, "maintain": 0, "reduce": 0, "high_risk": 0}

    for pid, product in _PRODUCTS_CACHE.items():
        decision_res = evaluate_decision(product)
        
        d_val = decision_res.decision
        r_val = decision_res.stockout_risk
        
        # Tally metrics
        counts["total"] += 1
        if d_val == "INCREASE":
            counts["increase"] += 1
        elif d_val == "MAINTAIN":
            counts["maintain"] += 1
        elif d_val == "REDUCE":
            counts["reduce"] += 1
            
        if r_val == "HIGH":
            counts["high_risk"] += 1

        # Check filters
        if category and product.category.lower() != category.lower():
            continue
        if decision_filter and d_val.upper() != decision_filter.upper():
            continue
        if risk_filter and r_val.upper() != risk_filter.upper():
            continue

        approval = _APPROVAL_RECORDS.get(pid)

        item = {
            "product_id": product.product_id,
            "product_name": product.product_name,
            "category": product.category,
            "brand": product.brand,
            "current_inventory": product.inventory.current_inventory,
            "reserved_inventory": product.inventory.reserved_inventory,
            "incoming_inventory": product.inventory.incoming_inventory,
            "available_inventory": decision_res.metrics.available_inventory,
            "forecast_7d": product.forecast.forecast_7d,
            "forecast_30d": product.forecast.forecast_30d,
            "sales_trend": product.forecast.sales_trend,
            "average_daily_demand": decision_res.metrics.average_daily_demand,
            "days_of_stock": decision_res.days_of_stock,
            "supplier_lead_time": product.supplier.lead_time_days,
            "moq": product.supplier.minimum_order_quantity,
            "reorder_point": decision_res.reorder_point,
            "stockout_risk": decision_res.stockout_risk,
            "market_signal": product.market_intelligence.market_signal if product.market_intelligence else "neutral",
            "market_confidence": product.market_intelligence.market_confidence if product.market_intelligence else 0.0,
            "decision": decision_res.decision,
            "decision_score": decision_res.decision_score,
            "confidence": decision_res.confidence,
            "recommended_order_quantity": decision_res.recommended_order_quantity,
            "approval_status": approval.action if approval else "PENDING",
            "approval_record": approval
        }
        results.append(item)

    return {
        "summary": counts,
        "count": len(results),
        "products": results
    }


@app.get("/products/{product_id}", tags=["Inventory & Decisions"])
def get_product_details(product_id: str):
    """
    Returns comprehensive product details, complete inventory metrics,
    decision breakdown, AI explanation, alternative recommendations, and human approval status.
    """
    global _PRODUCTS_CACHE
    if not _PRODUCTS_CACHE:
        _PRODUCTS_CACHE = load_handoff_data()

    product = _PRODUCTS_CACHE.get(product_id)
    if not product:
        raise HTTPException(status_code=404, detail=f"Product with ID '{product_id}' not found in handoff data.")

    decision_res = evaluate_decision(product)
    explanation = generate_llm_explanation(product, decision_res)
    
    # Candidate alternative products
    catalog = list(_PRODUCTS_CACHE.values())
    alternatives = find_alternative_products(product, catalog, top_n=3)

    approval = _APPROVAL_RECORDS.get(product_id)

    return {
        "product": product,
        "decision": decision_res,
        "ai_explanation": explanation,
        "alternative_products": alternatives,
        "approval": approval
    }


@app.post("/decision", response_model=DecisionResult, tags=["Decision Engine"])
def compute_decision_endpoint(input_data: ProductInput):
    """
    Calculates inventory metrics, stockout risk, decision, recommended order quantity,
    and transparent reasons for an ad-hoc or custom product payload.
    """
    product = input_data.to_product_handoff()
    return evaluate_decision(product)


@app.post("/analyze", response_model=FullAnalysisResponse, tags=["Decision Engine"])
def analyze_endpoint(input_data: ProductInput):
    """
    Performs full end-to-end analysis:
    Inventory Analysis + Deterministic Decision + AI Explanation + Alternative Products.
    """
    global _PRODUCTS_CACHE
    if not _PRODUCTS_CACHE:
        _PRODUCTS_CACHE = load_handoff_data()

    product = input_data.to_product_handoff()
    decision_res = evaluate_decision(product)
    explanation = generate_llm_explanation(product, decision_res)

    catalog = list(_PRODUCTS_CACHE.values()) if _PRODUCTS_CACHE else [product]
    alternatives = find_alternative_products(product, catalog, top_n=3)

    return FullAnalysisResponse(
        product=product,
        decision=decision_res,
        ai_explanation=explanation,
        alternative_products=alternatives
    )


@app.post("/alternative-products", response_model=List[AlternativeProduct], tags=["Catalog Intelligence"])
def get_alternative_products_endpoint(input_data: ProductInput):
    """
    Returns candidate alternative products for capital re-allocation.
    """
    global _PRODUCTS_CACHE
    if not _PRODUCTS_CACHE:
        _PRODUCTS_CACHE = load_handoff_data()

    product = input_data.to_product_handoff()
    catalog = list(_PRODUCTS_CACHE.values())
    return find_alternative_products(product, catalog, top_n=3)


@app.post("/approval", response_model=ApprovalRecord, tags=["Human-in-the-Loop"])
def submit_approval(request: ApprovalRequest):
    """
    Records a human decision action: APPROVE, MODIFY, or REJECT.
    Ensures strict human-in-the-loop oversight without triggering direct external supplier orders.
    """
    action_upper = request.action.upper()
    if action_upper not in {"APPROVE", "MODIFY", "REJECT"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Action must be one of: 'APPROVE', 'MODIFY', 'REJECT'"
        )

    product = _PRODUCTS_CACHE.get(request.product_id)
    product_name = product.product_name if product else request.product_id
    
    # Calculate original recommended quantity
    if product:
        dec = evaluate_decision(product)
        original_qty = dec.recommended_order_quantity
    else:
        original_qty = request.modified_quantity or 0

    if action_upper == "APPROVE":
        final_qty = original_qty
    elif action_upper == "MODIFY":
        if request.modified_quantity is None or request.modified_quantity < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="For 'MODIFY' action, modified_quantity must be specified and non-negative."
            )
        final_qty = request.modified_quantity
    else: # REJECT
        final_qty = 0

    record_id = f"APP-{uuid.uuid4().hex[:8].upper()}"
    record = ApprovalRecord(
        id=record_id,
        product_id=request.product_id,
        product_name=product_name,
        decision=request.decision,
        action=action_upper,
        original_recommended_quantity=original_qty,
        final_quantity=final_qty,
        notes=request.notes,
        created_at=datetime.now(timezone.utc).isoformat()
    )

    _APPROVAL_RECORDS[request.product_id] = record
    logger.info(f"Recorded human approval for {request.product_id}: {action_upper} (Final Qty: {final_qty})")
    return record


@app.get("/approvals", response_model=List[ApprovalRecord], tags=["Human-in-the-Loop"])
def get_all_approvals():
    """Returns all recorded human approval decisions."""
    return list(_APPROVAL_RECORDS.values())


@app.get("/demo/scenarios", tags=["Demonstration & Presentation"])
def get_demo_scenarios():
    """
    Provides standard presentation scenarios showcasing:
    - Scenario 1: Increasing demand + low inventory -> INCREASE (e.g. P019 Robot Vacuum)
    - Scenario 2: Stable demand + healthy inventory -> MAINTAIN (e.g. P010 Smartwatch)
    - Scenario 3: Decreasing demand + high inventory -> REDUCE (e.g. P001 Wireless Gaming Mouse)
    """
    global _PRODUCTS_CACHE
    if not _PRODUCTS_CACHE:
        _PRODUCTS_CACHE = load_handoff_data()

    scenarios = [
        {
            "scenario_id": "scenario_1",
            "name": "Scenario 1: Surging Demand & Low Stock",
            "expected_decision": "INCREASE",
            "recommended_product_id": "P019",
            "description": "Product with increasing sales trend, limited on-hand inventory coverage below supplier lead time, and high stockout risk."
        },
        {
            "scenario_id": "scenario_2",
            "name": "Scenario 2: Stable Demand & Balanced Coverage",
            "expected_decision": "MAINTAIN",
            "recommended_product_id": "P010",
            "description": "Product with stable demand velocity, comfortable coverage matching lead time, and balanced stock."
        },
        {
            "scenario_id": "scenario_3",
            "name": "Scenario 3: Declining Demand & Inventory Overhang",
            "expected_decision": "REDUCE",
            "recommended_product_id": "P001",
            "description": "Product facing decreasing demand trend with substantial excess inventory coverage."
        }
    ]

    enriched_scenarios = []
    for s in scenarios:
        pid = s["recommended_product_id"]
        prod = _PRODUCTS_CACHE.get(pid)
        if prod:
            dec = evaluate_decision(prod)
            enriched_scenarios.append({
                **s,
                "product_name": prod.product_name,
                "actual_decision": dec.decision,
                "days_of_stock": dec.days_of_stock,
                "stockout_risk": dec.stockout_risk,
                "recommended_order_quantity": dec.recommended_order_quantity
            })
        else:
            enriched_scenarios.append(s)

    return enriched_scenarios
