"""
Unit Tests for Person 2: Decision Engine Module.

Tests:
1. Increasing demand + low inventory -> INCREASE decision
2. Decreasing demand + high inventory -> REDUCE decision
3. Stable demand + healthy inventory -> MAINTAIN decision
4. High supplier lead time impacting decision
5. Incoming inventory mitigation impact
6. Missing / neutral market signal behavior
7. Zero demand behavior
8. Decision confidence score behavior (aligned signals vs divergent signals)
9. Transparent scoring logic and explainable reasons list
"""

import pytest
from backend.schemas import (
    ProductHandoff,
    ForecastData,
    InventoryData,
    SupplierData,
    MarketIntelligenceData
)
from backend.decision_engine import (
    evaluate_decision,
    compute_decision_factors,
    calculate_decision_confidence,
    DECISION_THRESHOLDS
)
from backend.inventory_analysis import analyze_inventory


class TestDecisionEngine:

    def test_scenario_1_increasing_demand_low_inventory(self):
        """
        Scenario 1:
        Increasing demand + low inventory coverage (< lead time) + positive market
        Expected: INCREASE, HIGH stockout risk, recommended order quantity > 0
        """
        product = ProductHandoff(
            product_id="P-INC",
            product_name="Surging Item",
            category="Electronics",
            brand="Nova",
            forecast=ForecastData(forecast_7d=140, forecast_30d=600, sales_trend="increasing"),
            inventory=InventoryData(current_inventory=50, reserved_inventory=0, incoming_inventory=0),
            supplier=SupplierData(lead_time_days=7, minimum_order_quantity=50, reorder_point=200),
            market_intelligence=MarketIntelligenceData(
                market_signal="positive",
                market_confidence=0.8,
                external_signal_available=True
            )
        )
        result = evaluate_decision(product)

        assert result.decision == "INCREASE"
        assert result.stockout_risk == "HIGH"
        assert result.days_of_stock < product.supplier.lead_time_days
        assert result.recommended_order_quantity > 0
        assert result.confidence >= 0.80
        assert any("increasing" in r.lower() for r in result.reasons)
        assert any("lead time" in r.lower() for r in result.reasons)

    def test_scenario_2_stable_demand_healthy_inventory(self):
        """
        Scenario 2:
        Stable demand + healthy inventory coverage (balanced with lead time)
        Expected: MAINTAIN, LOW or MEDIUM risk, recommended order quantity 0 or minimal
        """
        product = ProductHandoff(
            product_id="P-MAINTAIN",
            product_name="Steady Seller",
            category="Office",
            brand="DeskPro",
            forecast=ForecastData(forecast_7d=70, forecast_30d=300, sales_trend="stable"),
            inventory=InventoryData(current_inventory=350, reserved_inventory=0, incoming_inventory=0),
            supplier=SupplierData(lead_time_days=7, minimum_order_quantity=50, reorder_point=120),
            market_intelligence=MarketIntelligenceData(
                market_signal="neutral",
                market_confidence=0.5,
                external_signal_available=True
            )
        )
        result = evaluate_decision(product)

        assert result.decision == "MAINTAIN"
        assert result.stockout_risk == "LOW"
        assert result.days_of_stock >= product.supplier.lead_time_days * 2
        assert result.recommended_order_quantity == 0
        assert result.decision_score > DECISION_THRESHOLDS["reduce_score_cutoff"]
        assert result.decision_score < DECISION_THRESHOLDS["increase_score_cutoff"]

    def test_scenario_3_decreasing_demand_high_inventory(self):
        """
        Scenario 3:
        Decreasing demand + high inventory (> 60 days / 3.5x lead time)
        Expected: REDUCE, LOW stockout risk, recommended order quantity == 0
        """
        product = ProductHandoff(
            product_id="P-REDUCE",
            product_name="Legacy Gadget",
            category="Audio",
            brand="SoundMax",
            forecast=ForecastData(forecast_7d=35, forecast_30d=150, sales_trend="decreasing"),
            inventory=InventoryData(current_inventory=2500, reserved_inventory=0, incoming_inventory=0),
            supplier=SupplierData(lead_time_days=7, minimum_order_quantity=50, reorder_point=80),
            market_intelligence=MarketIntelligenceData(
                market_signal="negative",
                market_confidence=0.7,
                external_signal_available=True
            )
        )
        result = evaluate_decision(product)

        assert result.decision == "REDUCE"
        assert result.stockout_risk == "LOW"
        assert result.recommended_order_quantity == 0
        assert result.days_of_stock > 60.0
        assert result.decision_score <= DECISION_THRESHOLDS["reduce_score_cutoff"]
        assert any("decreasing" in r.lower() for r in result.reasons)

    def test_high_supplier_lead_time_triggers_risk(self):
        """
        When supplier lead time is high (e.g. 45 days) relative to 30 days inventory,
        stockout risk must escalate and influence the decision.
        """
        product = ProductHandoff(
            product_id="P-LEADTIME",
            product_name="Imported Part",
            forecast=ForecastData(forecast_7d=70, forecast_30d=300, sales_trend="stable"),
            inventory=InventoryData(current_inventory=300, reserved_inventory=0, incoming_inventory=0),
            supplier=SupplierData(lead_time_days=45, minimum_order_quantity=100, reorder_point=500),
            market_intelligence=None
        )
        result = evaluate_decision(product)

        # DOS = 30 days, lead time = 45 days -> DOS < lead time -> HIGH risk
        assert result.stockout_risk == "HIGH"
        assert result.days_of_stock < product.supplier.lead_time_days
        assert result.decision == "INCREASE"

    def test_incoming_inventory_prevents_unnecessary_increase(self):
        """
        When available inventory is low, but incoming inventory will arrive in time,
        stockout risk and order recommendations should account for it.
        """
        product = ProductHandoff(
            product_id="P-INCOMING",
            product_name="Restocked Widget",
            forecast=ForecastData(forecast_7d=70, forecast_30d=300, sales_trend="stable"),
            inventory=InventoryData(current_inventory=40, reserved_inventory=0, incoming_inventory=500),
            supplier=SupplierData(lead_time_days=10, minimum_order_quantity=50, reorder_point=120),
            market_intelligence=None
        )
        metrics = analyze_inventory(product)
        result = evaluate_decision(product)

        # Incoming inventory mitigates risk to MEDIUM and satisfies order horizon
        assert metrics.stockout_risk == "MEDIUM"
        assert result.recommended_order_quantity == 0
        assert result.decision != "REDUCE"

    def test_missing_market_signal_defaults_cleanly(self):
        """
        When market intelligence is None or has external_signal_available = False,
        the decision engine defaults cleanly without crashing.
        """
        product = ProductHandoff(
            product_id="P-NOMARKET",
            product_name="Generic Product",
            forecast=ForecastData(forecast_7d=70, forecast_30d=300, sales_trend="stable"),
            inventory=InventoryData(current_inventory=100, reserved_inventory=0, incoming_inventory=0),
            supplier=SupplierData(lead_time_days=7, minimum_order_quantity=20),
            market_intelligence=None
        )
        result = evaluate_decision(product)

        assert result.decision in {"INCREASE", "MAINTAIN", "REDUCE"}
        assert any("no active external market signal" in r.lower() for r in result.reasons)

    def test_zero_demand_handling(self):
        """
        Zero demand should result in safe DOS representation and not crash.
        """
        product = ProductHandoff(
            product_id="P-ZERODEMAND",
            product_name="Dormant SKU",
            forecast=ForecastData(forecast_7d=0, forecast_30d=0, sales_trend="stable"),
            inventory=InventoryData(current_inventory=50, reserved_inventory=0, incoming_inventory=0),
            supplier=SupplierData(lead_time_days=7, minimum_order_quantity=10),
            market_intelligence=None
        )
        result = evaluate_decision(product)

        assert result.metrics.average_daily_demand == 0.0
        assert result.recommended_order_quantity == 0
        assert result.decision in {"MAINTAIN", "REDUCE"}

    def test_decision_confidence_alignment(self):
        """
        Verifies that confidence increases when signals agree and is lower when signals conflict.
        """
        # Fully aligned factors
        aligned_factors = {
            "demand_factor": 0.30,
            "coverage_factor": 0.40,
            "stockout_factor": 0.30,
            "reorder_trigger_factor": 0.25,
            "market_factor": 0.12
        }
        high_conf = calculate_decision_confidence(aligned_factors, "INCREASE")
        assert high_conf >= 0.85

        # Conflicting factors (demand down but coverage tight)
        conflicted_factors = {
            "demand_factor": -0.30,
            "coverage_factor": 0.35,
            "stockout_factor": 0.20,
            "reorder_trigger_factor": 0.0,
            "market_factor": -0.10
        }
        lower_conf = calculate_decision_confidence(conflicted_factors, "INCREASE")
        assert lower_conf < high_conf
