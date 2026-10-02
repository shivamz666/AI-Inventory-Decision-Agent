"""
Alternative Product Recommendation Agent for Person 2: AI Inventory Decision Agent.

Identifies alternative products from the catalog when a product faces declining demand,
excessive inventory holding (REDUCE decision), or negative market sentiment.
Helps inventory managers re-allocate working capital toward higher-velocity opportunities.
"""

from typing import List, Dict, Any, Optional
from backend.schemas import ProductHandoff, AlternativeProduct
from backend.inventory_analysis import calculate_average_daily_demand, calculate_days_of_stock, calculate_available_inventory


def find_alternative_products(
    target_product: ProductHandoff,
    catalog: List[ProductHandoff],
    top_n: int = 3
) -> List[AlternativeProduct]:
    """
    Evaluates catalog products to find suitable alternatives for capital re-allocation.
    
    Filtering & Scoring criteria:
    - Excludes the target product itself.
    - Category affinity: Same category preferred, but cross-category evaluated if top-tier.
    - Trend superiority: Candidate has 'increasing' or 'stable' trend while target is 'decreasing'.
    - Velocity: Higher forecast / demand velocity.
    - Inventory health: Candidate has lower inventory overhang (healthy inventory turnover).
    - Market signal: Positive or neutral market intelligence.
    """
    target_demand = calculate_average_daily_demand(target_product.forecast.forecast_7d, target_product.forecast.forecast_30d)
    target_trend = target_product.forecast.sales_trend.lower()
    target_avail = calculate_available_inventory(target_product.inventory.current_inventory, target_product.inventory.reserved_inventory)
    target_dos = calculate_days_of_stock(target_avail, target_demand)

    candidates = []

    for item in catalog:
        if item.product_id == target_product.product_id:
            continue

        item_demand = calculate_average_daily_demand(item.forecast.forecast_7d, item.forecast.forecast_30d)
        item_trend = item.forecast.sales_trend.lower()
        item_avail = calculate_available_inventory(item.inventory.current_inventory, item.inventory.reserved_inventory)
        item_dos = calculate_days_of_stock(item_avail, item_demand)
        
        mi_signal = item.market_intelligence.market_signal.lower() if item.market_intelligence else "neutral"

        # Candidate Score calculation
        score = 0.0

        # Category match bonus
        same_category = item.category.strip().lower() == target_product.category.strip().lower()
        if same_category:
            score += 3.0

        # Trend comparison
        if item_trend == "increasing":
            score += 4.0
        elif item_trend == "stable" and target_trend == "decreasing":
            score += 2.0
        elif item_trend == "decreasing":
            score -= 3.0

        # Demand velocity comparison
        if item_demand > target_demand:
            score += 2.0
            
        # Market signal bonus
        if mi_signal == "positive":
            score += 1.5
        elif mi_signal == "negative":
            score -= 2.0

        # Inventory turnover health (lower excess days of stock)
        if item_dos < 60.0:
            score += 1.0

        if score > 2.0:
            # Build tailored explanation reason
            reasons = []
            if same_category:
                reasons.append(f"within the same category ('{item.category}')")
            if item_trend == "increasing":
                reasons.append(f"exhibits increasing demand trend (7d forecast: {item.forecast.forecast_7d} vs target: {target_product.forecast.forecast_7d})")
            if mi_signal == "positive":
                reasons.append("benefits from positive market news signals")
            if item_demand > target_demand:
                reasons.append(f"higher sales velocity ({item_demand:.1f} vs {target_demand:.1f} units/day)")

            reason_str = (
                f"{item.product_name} is a high-potential alternative for capital re-allocation: "
                + ", and ".join(reasons) + "."
            )

            candidates.append({
                "product": item,
                "score": score,
                "reason": reason_str,
                "metrics": {
                    "candidate_daily_demand": item_demand,
                    "target_daily_demand": target_demand,
                    "candidate_days_of_stock": item_dos,
                    "target_days_of_stock": target_dos,
                    "same_category": same_category,
                    "candidate_score": round(score, 2)
                }
            })

    # Sort descending by score
    candidates.sort(key=lambda x: x["score"], reverse=True)

    results: List[AlternativeProduct] = []
    for c in candidates[:top_n]:
        prod: ProductHandoff = c["product"]
        results.append(
            AlternativeProduct(
                product_id=prod.product_id,
                product_name=prod.product_name,
                category=prod.category,
                brand=prod.brand,
                sales_trend=prod.forecast.sales_trend,
                forecast_7d=prod.forecast.forecast_7d,
                forecast_30d=prod.forecast.forecast_30d,
                current_inventory=prod.inventory.current_inventory,
                market_signal=prod.market_intelligence.market_signal if prod.market_intelligence else "neutral",
                reason=c["reason"],
                supporting_metrics=c["metrics"]
            )
        )

    return results
