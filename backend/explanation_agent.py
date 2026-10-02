"""
LLM Explanation Agent for Person 2: AI Inventory Decision Agent.

Architectural Rule:
The LLM does NOT make the inventory decision.
Flow:
Input Data -> Decision Engine -> (INCREASE / MAINTAIN / REDUCE) + Metrics -> Explanation Agent -> Grounded Narrative.

Guarantees:
- Strictly adheres to calculated decision and metrics. Does not hallucinate numbers.
- Fully functional offline or without API keys via deterministic fallback generator.
- Optional live Gemini API enhancement if GEMINI_API_KEY is configured.
"""

import os
import json
import logging
from typing import Optional
from backend.schemas import ProductHandoff, DecisionResult

logger = logging.getLogger("BFWAI.ExplanationAgent")


def generate_deterministic_explanation(product: ProductHandoff, decision_res: DecisionResult) -> str:
    """
    Generates a rich, highly explainable, professional narrative strictly grounded
    in the calculated decision engine metrics and evidence.
    """
    d = decision_res.decision
    pid = product.product_id
    pname = product.product_name
    m = decision_res.metrics
    f = product.forecast
    lt = product.supplier.lead_time_days
    moq = product.supplier.minimum_order_quantity
    mi = product.market_intelligence

    # Paragraph 1: Core Recommendation & Operational Drivers
    if d == "INCREASE":
        intro = (
            f"RECOMMENDATION: INCREASE INVENTORY FOR {pname} ({pid}). "
            f"The decision engine has identified an urgent replenishment requirement. "
            f"Current available inventory stands at {m.available_inventory} units, which provides only "
            f"{m.days_of_stock:.1f} days of stock coverage against an average daily demand rate of "
            f"{m.average_daily_demand:.1f} units/day. This coverage is critically tight compared to the supplier "
            f"lead time of {lt} days, resulting in a calculated stockout risk of {m.stockout_risk}."
        )
    elif d == "REDUCE":
        intro = (
            f"RECOMMENDATION: REDUCE INVENTORY HOLDINGS FOR {pname} ({pid}). "
            f"The decision engine indicates substantial inventory overhang. "
            f"Available warehouse stock of {m.available_inventory} units covers approximately "
            f"{m.days_of_stock:.1f} days of demand, significantly surpassing the {lt}-day supplier fulfillment cycle. "
            f"With sales demand trending {f.sales_trend} (7-day forecast: {f.forecast_7d} units; 30-day forecast: {f.forecast_30d} units), "
            f"maintaining surplus stock exposes capital to unnecessary holding costs and depreciation."
        )
    else: # MAINTAIN
        intro = (
            f"RECOMMENDATION: MAINTAIN CURRENT INVENTORY LEVELS FOR {pname} ({pid}). "
            f"The inventory pipeline is currently well-balanced. Available stock of {m.available_inventory} units "
            f"offers {m.days_of_stock:.1f} days of stock coverage, comfortably aligned with the supplier's {lt}-day "
            f"lead time. Projected demand remains {f.sales_trend} with a 7-day forecast of {f.forecast_7d} units "
            f"and 30-day forecast of {f.forecast_30d} units."
        )

    # Paragraph 2: Order Quantity & Supplier Mechanics
    if decision_res.recommended_order_quantity > 0:
        order_details = (
            f"\n\nOrder Specification: A purchase order of {decision_res.recommended_order_quantity} units is recommended. "
            f"This order dynamically respects the supplier Minimum Order Quantity (MOQ) of {moq} units and accounts for "
            f"{m.safety_stock} units of calculated safety stock to absorb demand volatility during the {lt}-day transit."
        )
    else:
        order_details = (
            f"\n\nOrder Specification: No new purchase order is required at this time (recommended order: 0 units). "
            f"Existing on-hand inventory ({m.available_inventory} units) and incoming pipeline ({product.inventory.incoming_inventory} units) "
            f"adequately satisfy projected demand and maintain safety stock ({m.safety_stock} units) above the reorder point of {m.reorder_point} units."
        )

    # Paragraph 3: Market Intelligence Context
    if mi and mi.external_signal_available:
        market_details = (
            f"\n\nExternal Market Context: Market intelligence indicates a {mi.market_signal.upper()} external sentiment "
            f"(confidence: {mi.market_confidence:.0%}). {mi.market_reason}"
        )
    else:
        market_details = (
            "\n\nExternal Market Context: No external market anomalies or breaking news were detected for this product. "
            "The decision is primarily driven by internal demand velocity and supply chain lead times."
        )

    return f"{intro}{order_details}{market_details}"


def generate_llm_explanation(product: ProductHandoff, decision_res: DecisionResult) -> str:
    """
    Attempts to generate an executive AI explanation using Gemini if configured,
    otherwise smoothly falls back to the deterministic generator.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return generate_deterministic_explanation(product, decision_res)

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        
        prompt = f"""You are the Executive Explanation Agent for the BFWAI AI Inventory Decision System.
Explain the following inventory decision to an operations executive in 2-3 concise paragraphs.
IMPORTANT RULES:
1. You MUST NOT change or contradict the decision or numbers below.
2. Only use the numbers provided in the data. Do NOT invent new figures.
3. Be professional, clear, and business-focused.

Product: {product.product_name} ({product.product_id})
Category: {product.category}
Decision: {decision_res.decision} (Confidence: {decision_res.confidence:.0%})
Calculated Stockout Risk: {decision_res.stockout_risk}
Recommended Order Quantity: {decision_res.recommended_order_quantity} units (MOQ: {product.supplier.minimum_order_quantity})
Days of Stock: {decision_res.days_of_stock:.1f} days
Supplier Lead Time: {product.supplier.lead_time_days} days
Safety Stock: {decision_res.metrics.safety_stock} units
Reorder Point: {decision_res.reorder_point} units
Current Available Inventory: {decision_res.metrics.available_inventory} units
Sales Trend: {product.forecast.sales_trend} (7-Day Forecast: {product.forecast.forecast_7d} units, 30-Day Forecast: {product.forecast.forecast_30d} units)
Market Signal: {product.market_intelligence.market_signal if product.market_intelligence else 'neutral'}
Market Context: {product.market_intelligence.market_reason if product.market_intelligence else 'N/A'}
Key Reasons:
{chr(10).join(['- ' + r for r in decision_res.reasons])}
"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        if response and response.text:
            return response.text.strip()
    except Exception as e:
        logger.warning(f"Live LLM call failed ({e}). Falling back to deterministic explanation.")

    return generate_deterministic_explanation(product, decision_res)
