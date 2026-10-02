"""
Decision Engine Module for Person 2: AI Inventory Decision Agent.

Combines:
- Demand forecast & sales trend
- Inventory level & Days of stock
- Reorder point & Safety stock
- Supplier lead time & MOQ
- Incoming inventory
- Market signal & confidence
- Stockout risk

Produces a deterministic, transparent decision:
- INCREASE
- MAINTAIN
- REDUCE

Includes:
- Calculated decision score (-1.0 to 1.0)
- Decision confidence score based on signal alignment (0.0 to 1.0)
- Structured, human-explainable reason strings
"""

from typing import List, Dict, Any, Tuple
from backend.schemas import ProductHandoff, DecisionResult
from backend.inventory_analysis import analyze_inventory, InventoryMetrics

# Central Configurable Thresholds for Decision Engine
DECISION_THRESHOLDS = {
    "increase_score_cutoff": 0.25,    # Composite score >= 0.25 -> INCREASE
    "reduce_score_cutoff": -0.25,     # Composite score <= -0.25 -> REDUCE
    "overstock_dos_threshold": 60.0,  # Coverage > 60 days signals overstock risk
    "excess_coverage_ratio": 3.5,     # DOS > 3.5x lead time signals overstock
    "max_market_weight": 0.15         # Market signal is bounded as one factor, never sole decider
}


def compute_decision_factors(
    product: ProductHandoff,
    metrics: InventoryMetrics
) -> Tuple[float, List[str], Dict[str, float]]:
    """
    Computes individual signal sub-scores, transparent reasons, and final composite score.
    
    Sub-scores:
    1. Demand Factor (-0.30 to +0.30)
    2. Inventory Coverage Factor (-0.40 to +0.40)
    3. Stockout Risk Factor (-0.20 to +0.30)
    4. Reorder Point Trigger Factor (0.0 to +0.25)
    5. Market Intelligence Factor (-0.15 to +0.15)
    
    Returns:
        (composite_score, reasons_list, factor_breakdown)
    """
    reasons = []
    factors = {}
    
    # -------------------------------------------------------------
    # 1. Demand & Sales Trend Factor
    # -------------------------------------------------------------
    trend = product.forecast.sales_trend.lower()
    if trend == "increasing":
        demand_score = 0.30
        reasons.append(f"Demand is increasing (7d forecast: {product.forecast.forecast_7d} units, 30d forecast: {product.forecast.forecast_30d} units).")
    elif trend == "decreasing":
        demand_score = -0.30
        reasons.append(f"Demand is decreasing (7d forecast: {product.forecast.forecast_7d} units, 30d forecast: {product.forecast.forecast_30d} units).")
    else:
        demand_score = 0.05 if product.forecast.forecast_7d > 0 else 0.0
        reasons.append("Demand velocity is stable across recent forecasting periods.")
    factors["demand_factor"] = demand_score

    # -------------------------------------------------------------
    # 2. Inventory Coverage Factor
    # -------------------------------------------------------------
    coverage_ratio = metrics.coverage_ratio
    lead_time = max(1, product.supplier.lead_time_days)
    
    if metrics.days_of_stock < lead_time:
        # Stock runs out before new shipment can arrive
        coverage_score = 0.40
        reasons.append(f"Current inventory coverage ({metrics.days_of_stock:.1f} days) is below supplier lead time ({lead_time} days).")
    elif coverage_ratio < 1.5:
        # Lean buffer
        coverage_score = 0.20
        reasons.append(f"Inventory coverage ({metrics.days_of_stock:.1f} days) provides a tight window relative to lead time ({lead_time} days).")
    elif (
        metrics.days_of_stock > DECISION_THRESHOLDS["overstock_dos_threshold"]
        or (coverage_ratio > DECISION_THRESHOLDS["excess_coverage_ratio"] and metrics.days_of_stock > 45.0)
    ):
        # Substantial excess inventory
        coverage_score = -0.40 if trend == "decreasing" else -0.20
        reasons.append(f"Significant inventory coverage ({metrics.days_of_stock:.1f} days), exceeding optimal holding thresholds.")
    else:
        coverage_score = 0.0
        reasons.append(f"Inventory coverage ({metrics.days_of_stock:.1f} days) is balanced with supplier lead time ({lead_time} days).")
    factors["coverage_factor"] = coverage_score

    # -------------------------------------------------------------
    # 3. Stockout Risk Factor
    # -------------------------------------------------------------
    risk = metrics.stockout_risk
    if risk == "HIGH":
        stockout_score = 0.30
        reasons.append("Calculated stockout risk is HIGH, posing an immediate out-of-stock threat.")
    elif risk == "MEDIUM":
        stockout_score = 0.10
        reasons.append("Calculated stockout risk is MEDIUM; replenishment window should be monitored.")
    else:
        stockout_score = -0.10 if coverage_score < 0 else 0.0
        reasons.append("Calculated stockout risk is LOW.")
    factors["stockout_factor"] = stockout_score

    # -------------------------------------------------------------
    # 4. Reorder Point Trigger Factor
    # -------------------------------------------------------------
    if metrics.available_inventory <= metrics.reorder_point:
        reorder_score = 0.25
        source_label = "fallback" if metrics.is_reorder_point_fallback else "supplier"
        reasons.append(
            f"Available inventory ({metrics.available_inventory}) has breached the {source_label} reorder threshold ({metrics.reorder_point})."
        )
    else:
        reorder_score = 0.0
    factors["reorder_trigger_factor"] = reorder_score

    # -------------------------------------------------------------
    # 5. Market Intelligence Factor (bounded influence)
    # -------------------------------------------------------------
    mi = product.market_intelligence
    if mi and mi.external_signal_available:
        conf = max(0.0, min(1.0, mi.market_confidence))
        sig = mi.market_signal.lower()
        if sig == "positive":
            market_score = round(DECISION_THRESHOLDS["max_market_weight"] * conf, 3)
            reasons.append(f"External market intelligence is positive (confidence: {conf:.0%}).")
        elif sig == "negative":
            market_score = round(-DECISION_THRESHOLDS["max_market_weight"] * conf, 3)
            reasons.append(f"External market intelligence is negative (confidence: {conf:.0%}).")
        else:
            market_score = 0.0
            reasons.append("External market intelligence is neutral.")
    else:
        market_score = 0.0
        reasons.append("No active external market signal; defaulting to neutral market factor.")
    factors["market_factor"] = market_score

    # -------------------------------------------------------------
    # Sum composite score clamped between -1.0 and 1.0
    # -------------------------------------------------------------
    raw_score = sum(factors.values())
    composite_score = max(-1.0, min(1.0, round(raw_score, 3)))
    return composite_score, reasons, factors


def calculate_decision_confidence(factors: Dict[str, float], decision: str) -> float:
    """
    Computes a transparent decision confidence score (0.0 to 1.0)
    reflecting the degree of harmony/alignment among the various signals.
    
    If all signals align (e.g. demand up, stock low, risk high for INCREASE),
    confidence is high (0.85 - 0.96).
    If signals conflict (e.g. demand falling but stock also very low),
    confidence is lower (0.55 - 0.72).
    """
    pos_count = sum(1 for v in factors.values() if v > 0.05)
    neg_count = sum(1 for v in factors.values() if v < -0.05)
    total_active = pos_count + neg_count
    
    if decision == "INCREASE":
        alignment = pos_count / max(1, total_active)
        base_confidence = 0.65 + (0.30 * alignment)
    elif decision == "REDUCE":
        alignment = neg_count / max(1, total_active)
        base_confidence = 0.65 + (0.30 * alignment)
    else: # MAINTAIN
        # For maintain, balance near 0 gives higher confidence
        dispersion = abs(pos_count - neg_count)
        base_confidence = 0.85 - (0.15 * dispersion)
        
    return round(max(0.50, min(0.98, base_confidence)), 2)


def evaluate_decision(product: ProductHandoff) -> DecisionResult:
    """
    Main entry point for evaluating inventory decisions for a product.
    Coordinates inventory analysis and deterministic multi-signal decision logic.
    """
    # 1. Calculate inventory metrics
    metrics = analyze_inventory(product)
    
    # 2. Compute transparent factors and raw score
    score, reasons, factors = compute_decision_factors(product, metrics)
    
    # 3. Determine deterministic decision state
    # Strict guardrails ensure explainability:
    if (
        metrics.stockout_risk == "HIGH"
        and metrics.days_of_stock < product.supplier.lead_time_days
        and product.forecast.sales_trend != "decreasing"
    ):
        decision = "INCREASE"
    elif (
        product.forecast.sales_trend == "decreasing"
        and (metrics.days_of_stock > DECISION_THRESHOLDS["overstock_dos_threshold"] or metrics.coverage_ratio >= 3.0)
        and metrics.stockout_risk == "LOW"
    ):
        decision = "REDUCE"
    elif score >= DECISION_THRESHOLDS["increase_score_cutoff"]:
        decision = "INCREASE"
    elif score <= DECISION_THRESHOLDS["reduce_score_cutoff"]:
        decision = "REDUCE"
    else:
        decision = "MAINTAIN"

    # Edge-case safety checks:
    # If decision is REDUCE but stockout risk is HIGH, force to MAINTAIN
    if decision == "REDUCE" and metrics.stockout_risk == "HIGH":
        decision = "MAINTAIN"
        reasons.append("Overriding REDUCE to MAINTAIN due to high stockout risk.")
        
    # If decision is INCREASE but recommended order is 0 and inventory is high, adjust to MAINTAIN
    if decision == "INCREASE" and metrics.recommended_order_quantity == 0 and metrics.days_of_stock > product.supplier.lead_time_days * 2:
        decision = "MAINTAIN"
        reasons.append("Existing inventory and pipeline sufficiently cover demand; maintaining current holdings.")

    # 4. Calculate decision confidence
    confidence = calculate_decision_confidence(factors, decision)

    return DecisionResult(
        product_id=product.product_id,
        product_name=product.product_name,
        decision=decision,
        confidence=confidence,
        stockout_risk=metrics.stockout_risk,
        days_of_stock=metrics.days_of_stock,
        reorder_point=metrics.reorder_point,
        recommended_order_quantity=metrics.recommended_order_quantity,
        decision_score=score,
        reasons=reasons,
        metrics=metrics
    )
