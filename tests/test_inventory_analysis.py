"""
Unit Tests for Person 2: Inventory Analysis Module.

Tests:
1. Available inventory calculation (current - reserved, positive handling)
2. Average daily demand calculation (7d priority, 30d fallback, zero demand handling)
3. Days of stock calculation (safe division, zero demand, zero inventory)
4. Safety stock calculation (configurable factor, square root of lead time)
5. Reorder point determination (Person 1 provided vs fallback calculation)
6. Stockout risk calculation:
   - High risk (DOS < lead time)
   - Medium risk (DOS ~ lead time, or tight buffer)
   - Low risk (DOS significantly exceeds lead time)
   - Incoming inventory mitigation
   - Acceleration under increasing sales trend
7. Recommended order quantity calculation:
   - Net requirement calculation
   - Dynamic MOQ rounding (e.g. 73 requirement with MOQ 50 -> 100)
   - Zero order when inventory is sufficient
   - Incoming inventory deduction
8. Edge cases:
   - Zero demand
   - Negative / invalid inventory
   - Missing optional fields
"""

import pytest
from backend.schemas import (
    ProductHandoff,
    ForecastData,
    InventoryData,
    SupplierData,
    MarketIntelligenceData
)
from backend.inventory_analysis import (
    calculate_available_inventory,
    calculate_average_daily_demand,
    calculate_days_of_stock,
    calculate_safety_stock,
    determine_reorder_point,
    calculate_stockout_risk,
    calculate_recommended_order_quantity,
    analyze_inventory
)


class TestInventoryAnalysis:

    def test_available_inventory_basic(self):
        # Current 100, Reserved 20 -> 80
        avail = calculate_available_inventory(100, 20)
        assert avail == 80

    def test_available_inventory_reserved_exceeds_current(self):
        # Current 10, Reserved 50 -> clamped to 0, no negative
        avail = calculate_available_inventory(10, 50)
        assert avail == 0

    def test_available_inventory_negative_inputs(self):
        # Safeguards against invalid negative numbers
        avail = calculate_available_inventory(-5, -2)
        assert avail == 0

    def test_average_daily_demand_7d_priority(self):
        # 7d forecast: 70 -> 10.0 units/day
        rate = calculate_average_daily_demand(70, 300)
        assert rate == 10.0

    def test_average_daily_demand_30d_fallback(self):
        # 7d forecast: 0, 30d forecast: 60 -> 2.0 units/day
        rate = calculate_average_daily_demand(0, 60)
        assert rate == 2.0

    def test_average_daily_demand_zero_demand(self):
        # Zero demand safely returns 0.0
        rate = calculate_average_daily_demand(0, 0)
        assert rate == 0.0

    def test_days_of_stock_standard(self):
        # 100 units / 10 units/day = 10.0 days
        dos = calculate_days_of_stock(100, 10.0)
        assert dos == 10.0

    def test_days_of_stock_zero_demand_with_inventory(self):
        # 100 units with 0 demand -> 999.0 safe representation
        dos = calculate_days_of_stock(100, 0.0)
        assert dos == 999.0

    def test_days_of_stock_zero_inventory_zero_demand(self):
        # 0 units with 0 demand -> 0.0 days
        dos = calculate_days_of_stock(0, 0.0)
        assert dos == 0.0

    def test_safety_stock_heuristic(self):
        # Daily demand = 10, Lead time = 4 days
        # sqrt(4) = 2 -> 1.65 * 2 * 10 = 33 units
        ss = calculate_safety_stock(10.0, 4)
        assert ss == 33

    def test_safety_stock_zero_demand(self):
        ss = calculate_safety_stock(0.0, 7)
        assert ss == 0

    def test_reorder_point_person1_provided(self):
        # When Person 1 provided valid reorder point
        rop, is_fallback = determine_reorder_point(456, 10.0, 7, 30)
        assert rop == 456
        assert is_fallback is False

    def test_reorder_point_calculated_fallback(self):
        # When Person 1 reorder point is None or 0
        # lead_time_demand = 10 * 7 = 70; safety_stock = 30 -> 100
        rop, is_fallback = determine_reorder_point(None, 10.0, 7, 30)
        assert rop == 100
        assert is_fallback is True

    def test_stockout_risk_high_when_dos_below_lead_time(self):
        # DOS = 5.0, Lead time = 10 -> HIGH risk
        risk, evidence = calculate_stockout_risk(
            days_of_stock=5.0,
            lead_time_days=10,
            available_inventory=50,
            incoming_inventory=0,
            reorder_point=100,
            sales_trend="stable",
            average_daily_demand=10.0
        )
        assert risk == "HIGH"
        assert evidence["coverage_ratio"] == 0.5

    def test_stockout_risk_incoming_inventory_mitigation(self):
        # DOS = 5.0, Lead time = 10 (coverage < lead time), but incoming = 200 units
        # total pipeline = 250 units -> pipeline DOS = 25 days >= 1.5 * 10 days
        # Risk is mitigated to MEDIUM
        risk, evidence = calculate_stockout_risk(
            days_of_stock=5.0,
            lead_time_days=10,
            available_inventory=50,
            incoming_inventory=200,
            reorder_point=100,
            sales_trend="stable",
            average_daily_demand=10.0
        )
        assert risk == "MEDIUM"
        assert "mitigation" in evidence["risk_rationale"]

    def test_stockout_risk_increasing_trend_accelerator(self):
        # DOS = 12.0, Lead time = 10 (coverage ratio 1.2, between 1.0 and 1.75)
        # With increasing trend + below reorder point -> elevates to HIGH
        risk, _ = calculate_stockout_risk(
            days_of_stock=12.0,
            lead_time_days=10,
            available_inventory=120,
            incoming_inventory=0,
            reorder_point=150,
            sales_trend="increasing",
            average_daily_demand=10.0
        )
        assert risk == "HIGH"

    def test_stockout_risk_low_for_excess_coverage(self):
        # DOS = 50.0, Lead time = 7 -> coverage ratio > 7x -> LOW
        risk, _ = calculate_stockout_risk(
            days_of_stock=50.0,
            lead_time_days=7,
            available_inventory=500,
            incoming_inventory=0,
            reorder_point=100,
            sales_trend="stable",
            average_daily_demand=10.0
        )
        assert risk == "LOW"

    def test_recommended_order_quantity_dynamic_moq(self):
        # Forecast 30d = 300
        # Safety Stock = 50
        # Gross = 350
        # Available = 200, Incoming = 50
        # Net Requirement = 350 - 200 - 50 = 100
        # MOQ = 40 -> ceil(100/40) * 40 = 3 * 40 = 120 units
        qty, evidence = calculate_recommended_order_quantity(
            forecast_30d=300,
            average_daily_demand=10.0,
            lead_time_days=7,
            safety_stock=50,
            available_inventory=200,
            incoming_inventory=50,
            minimum_order_quantity=40
        )
        assert qty == 120
        assert evidence["net_requirement"] == 100
        assert evidence["moq_multiples"] == 3

    def test_recommended_order_quantity_zero_when_stock_exceeds(self):
        # Available = 500, Gross Requirement = 350 -> Net = -150 -> 0 order
        qty, evidence = calculate_recommended_order_quantity(
            forecast_30d=300,
            average_daily_demand=10.0,
            lead_time_days=7,
            safety_stock=50,
            available_inventory=500,
            incoming_inventory=0,
            minimum_order_quantity=50
        )
        assert qty == 0
        assert evidence["net_requirement"] <= 0

    def test_analyze_inventory_full_integration(self):
        product = ProductHandoff(
            product_id="TEST-01",
            product_name="Wireless Gadget",
            category="Electronics",
            brand="BrandX",
            forecast=ForecastData(forecast_7d=70, forecast_30d=300, sales_trend="increasing"),
            inventory=InventoryData(current_inventory=50, reserved_inventory=10, incoming_inventory=0),
            supplier=SupplierData(lead_time_days=7, minimum_order_quantity=50, reorder_point=120),
            market_intelligence=MarketIntelligenceData(market_signal="positive", market_confidence=0.8)
        )
        metrics = analyze_inventory(product)
        
        assert metrics.available_inventory == 40
        assert metrics.average_daily_demand == 10.0
        assert metrics.days_of_stock == 4.0
        assert metrics.stockout_risk == "HIGH"
        assert metrics.reorder_point == 120
        assert metrics.is_reorder_point_fallback is False
        assert metrics.recommended_order_quantity > 0
        assert metrics.recommended_order_quantity % 50 == 0  # Multiple of MOQ
