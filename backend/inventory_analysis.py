"""
Inventory Analysis Module for Person 2: AI Inventory Decision Agent.

Calculates key inventory metrics:
- Available Inventory (current - reserved)
- Average Daily Demand
- Days of Stock (DOS)
- Inventory Coverage vs Lead Time
- Stockout Risk (Explainable heuristic: LOW / MEDIUM / HIGH)
- Safety Stock
- Reorder Point (Person 1 provided or calculated fallback)
- Recommended Order Quantity (accounting for MOQ, incoming, safety stock)
"""

import math
from typing import Dict, Any, Tuple
from backend.schemas import ProductHandoff, InventoryMetrics

# Central Configurable Thresholds for Explainable Risk Assessment
STOCKOUT_RISK_CONFIG = {
    "coverage_high_risk_ratio": 1.0,     # DOS < 1.0x lead_time => HIGH risk baseline
    "coverage_medium_risk_ratio": 1.75,  # DOS between 1.0x and 1.75x lead_time => MEDIUM risk baseline
    "reorder_point_buffer_ratio": 1.1,   # Inventory within 10% of reorder point => elevate risk
    "safety_stock_factor": 1.65,         # 95% service level factor for sqrt(lead_time) formula
    "planning_horizon_days": 30          # Horizon for recommended order demand cycle
}


def calculate_available_inventory(current_inventory: int, reserved_inventory: int) -> int:
    """
    Available Inventory is stock physically present minus reserved customer commitments.
    Incoming inventory is tracked separately and not assumed immediately available on day 0.
    """
    return max(0, int(current_inventory) - int(reserved_inventory))


def calculate_average_daily_demand(forecast_7d: int, forecast_30d: int) -> float:
    """
    Calculates average daily demand rate safely from forecasts without division by zero.
    Prioritizes 7-day forecast for near-term velocity, falling back to 30-day forecast.
    """
    if forecast_7d > 0:
        return round(float(forecast_7d) / 7.0, 3)
    elif forecast_30d > 0:
        return round(float(forecast_30d) / 30.0, 3)
    return 0.0


def calculate_days_of_stock(available_inventory: int, average_daily_demand: float) -> float:
    """
    Calculates days of stock coverage. Handles zero demand safely.
    """
    if average_daily_demand <= 0.0:
        # If no demand is projected, stock lasts indefinitely (represented as 999.0 days)
        # or 0 days if there is literally 0 inventory.
        return 999.0 if available_inventory > 0 else 0.0
    return round(float(available_inventory) / average_daily_demand, 2)


def calculate_safety_stock(average_daily_demand: float, lead_time_days: int) -> int:
    """
    Calculates safety stock using standard supply chain heuristic:
    Safety Stock = ceil(safety_factor * sqrt(lead_time_days) * average_daily_demand)
    Uses 95% service level factor (1.65).
    """
    if average_daily_demand <= 0.0 or lead_time_days <= 0:
        return 0
    factor = STOCKOUT_RISK_CONFIG["safety_stock_factor"]
    std_dev_approx = math.sqrt(lead_time_days) * average_daily_demand
    ss = math.ceil(factor * std_dev_approx)
    return max(1, ss)


def determine_reorder_point(
    person1_reorder_point: Any,
    average_daily_demand: float,
    lead_time_days: int,
    safety_stock: int
) -> Tuple[int, bool]:
    """
    Uses Person 1-provided reorder point if valid (> 0).
    Otherwise calculates transparent fallback:
    Reorder Point = (average_daily_demand * lead_time_days) + safety_stock
    
    Returns:
        (reorder_point_value, is_fallback_boolean)
    """
    if person1_reorder_point is not None:
        try:
            val = int(person1_reorder_point)
            if val > 0:
                return val, False
        except (ValueError, TypeError):
            pass

    # Transparent fallback calculation
    lead_time_demand = average_daily_demand * float(lead_time_days)
    fallback_rop = math.ceil(lead_time_demand + float(safety_stock))
    return max(1, fallback_rop), True


def calculate_stockout_risk(
    days_of_stock: float,
    lead_time_days: int,
    available_inventory: int,
    incoming_inventory: int,
    reorder_point: int,
    sales_trend: str,
    average_daily_demand: float
) -> Tuple[str, Dict[str, Any]]:
    """
    Calculates stockout risk dynamically using an explainable heuristic:
    1. Coverage ratio = days_of_stock / lead_time_days
    2. Check if incoming inventory mitigates imminent stockout
    3. Check proximity to reorder point
    4. Factor in sales trend momentum (increasing accelerates risk)
    
    Returns:
        (risk_level: "LOW" | "MEDIUM" | "HIGH", evidence_dict)
    """
    lead_time = max(1, lead_time_days)
    coverage_ratio = days_of_stock / float(lead_time)
    
    # Calculate days of stock when incoming shipment arrives
    total_pipeline_stock = available_inventory + incoming_inventory
    pipeline_dos = (
        round(total_pipeline_stock / average_daily_demand, 2)
        if average_daily_demand > 0
        else (999.0 if total_pipeline_stock > 0 else 0.0)
    )

    is_below_reorder_point = available_inventory <= reorder_point
    is_increasing_trend = sales_trend.lower() == "increasing"
    is_decreasing_trend = sales_trend.lower() == "decreasing"

    # Base risk determination from coverage ratio
    if coverage_ratio < STOCKOUT_RISK_CONFIG["coverage_high_risk_ratio"]:
        # Available stock runs out BEFORE supplier lead time!
        if pipeline_dos >= lead_time * 1.5 and incoming_inventory > 0:
            # Significant incoming stock is already in transit
            risk = "MEDIUM"
            reason = "Available coverage is below lead time, but incoming inventory provides mitigation."
        else:
            risk = "HIGH"
            reason = f"Available coverage ({days_of_stock} days) is less than supplier lead time ({lead_time} days)."
    elif coverage_ratio < STOCKOUT_RISK_CONFIG["coverage_medium_risk_ratio"]:
        # Stock covers lead time with narrow buffer
        if is_increasing_trend or is_below_reorder_point:
            risk = "HIGH" if (is_increasing_trend and is_below_reorder_point) else "MEDIUM"
            reason = "Inventory coverage is tight and demand is accelerating or below reorder point."
        else:
            risk = "MEDIUM"
            reason = f"Inventory coverage ({days_of_stock} days) is close to supplier lead time ({lead_time} days)."
    else:
        # Healthy coverage (>= 1.75x lead time)
        if is_below_reorder_point and is_increasing_trend:
            risk = "MEDIUM"
            reason = "Healthy coverage ratio but inventory is approaching reorder threshold under surging demand."
        else:
            risk = "LOW"
            reason = f"Inventory coverage ({days_of_stock} days) comfortably exceeds supplier lead time ({lead_time} days)."

    # Edge case: zero demand and zero inventory
    if available_inventory == 0 and average_daily_demand > 0:
        risk = "HIGH"
        reason = "Zero available inventory while active customer demand exists."
    elif available_inventory == 0 and average_daily_demand == 0:
        risk = "LOW"
        reason = "Zero inventory with zero forecasted demand."

    evidence = {
        "coverage_ratio": round(coverage_ratio, 2),
        "pipeline_days_of_stock": pipeline_dos,
        "is_below_reorder_point": is_below_reorder_point,
        "risk_rationale": reason
    }
    return risk, evidence


def calculate_recommended_order_quantity(
    forecast_30d: int,
    average_daily_demand: float,
    lead_time_days: int,
    safety_stock: int,
    available_inventory: int,
    incoming_inventory: int,
    minimum_order_quantity: int
) -> Tuple[int, Dict[str, Any]]:
    """
    Calculates dynamic recommended order quantity respecting MOQ.
    
    Formula:
    Demand During Horizon = forecast_30d (or average_daily_demand * 30)
    Gross Requirement = Demand During Horizon + Safety Stock
    Net Requirement = Gross Requirement - Available Inventory - Incoming Inventory
    
    If Net Requirement <= 0:
        recommended_order_quantity = 0
    Else:
        recommended_order_quantity = ceil(Net Requirement / MOQ) * MOQ
        
    Example:
        net_requirement = 73, MOQ = 50 => ceil(73/50)*50 = 100 units.
    """
    moq = max(1, minimum_order_quantity)
    planning_days = STOCKOUT_RISK_CONFIG["planning_horizon_days"]
    
    demand_horizon = forecast_30d if forecast_30d > 0 else math.ceil(average_daily_demand * planning_days)
    gross_requirement = demand_horizon + safety_stock
    net_requirement = gross_requirement - available_inventory - incoming_inventory

    if net_requirement <= 0:
        recommended_quantity = 0
    else:
        multiplier = math.ceil(float(net_requirement) / float(moq))
        recommended_quantity = int(multiplier * moq)

    evidence = {
        "planning_horizon_days": planning_days,
        "forecast_demand_horizon": demand_horizon,
        "safety_stock": safety_stock,
        "gross_requirement": gross_requirement,
        "available_inventory": available_inventory,
        "incoming_inventory": incoming_inventory,
        "net_requirement": net_requirement,
        "minimum_order_quantity": moq,
        "moq_multiples": math.ceil(float(net_requirement) / float(moq)) if net_requirement > 0 else 0,
        "recommended_order_quantity": recommended_quantity
    }
    return recommended_quantity, evidence


def analyze_inventory(product: ProductHandoff) -> InventoryMetrics:
    """
    Performs complete inventory analysis on a ProductHandoff item.
    Returns populated InventoryMetrics schema.
    """
    curr_inv = max(0, product.inventory.current_inventory)
    res_inv = max(0, product.inventory.reserved_inventory)
    inc_inv = max(0, product.inventory.incoming_inventory)
    
    avail_inv = calculate_available_inventory(curr_inv, res_inv)
    daily_demand = calculate_average_daily_demand(product.forecast.forecast_7d, product.forecast.forecast_30d)
    dos = calculate_days_of_stock(avail_inv, daily_demand)
    
    lead_time = max(1, product.supplier.lead_time_days)
    coverage_ratio = round(dos / float(lead_time), 2)
    
    safety_stk = calculate_safety_stock(daily_demand, lead_time)
    reorder_pt, is_fallback = determine_reorder_point(
        product.supplier.reorder_point,
        daily_demand,
        lead_time,
        safety_stk
    )
    
    risk, risk_evidence = calculate_stockout_risk(
        days_of_stock=dos,
        lead_time_days=lead_time,
        available_inventory=avail_inv,
        incoming_inventory=inc_inv,
        reorder_point=reorder_pt,
        sales_trend=product.forecast.sales_trend,
        average_daily_demand=daily_demand
    )
    
    moq = max(1, product.supplier.minimum_order_quantity)
    rec_qty, order_evidence = calculate_recommended_order_quantity(
        forecast_30d=product.forecast.forecast_30d,
        average_daily_demand=daily_demand,
        lead_time_days=lead_time,
        safety_stock=safety_stk,
        available_inventory=avail_inv,
        incoming_inventory=inc_inv,
        minimum_order_quantity=moq
    )
    
    # Merge calculation evidence
    combined_evidence = {
        **order_evidence,
        **risk_evidence
    }

    return InventoryMetrics(
        available_inventory=avail_inv,
        average_daily_demand=daily_demand,
        days_of_stock=dos,
        coverage_ratio=coverage_ratio,
        stockout_risk=risk,
        safety_stock=safety_stk,
        reorder_point=reorder_pt,
        is_reorder_point_fallback=is_fallback,
        recommended_order_quantity=rec_qty,
        order_calculation_evidence=combined_evidence
    )
