"""
Pydantic Schemas for Person 2: AI Inventory Decision Agent.

These schemas interface with Person 1's handoff output (handoff/person2_handoff.json)
and define the data models for inventory analysis, decision engine, AI explanation,
alternative product recommendations, and human approvals.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator
from datetime import datetime, timezone


class ForecastData(BaseModel):
    forecast_7d: int = Field(default=0, ge=0, description="7-day demand forecast (units)")
    forecast_30d: int = Field(default=0, ge=0, description="30-day demand forecast (units)")
    sales_trend: str = Field(default="stable", description="Demand trend: increasing, decreasing, or stable")

    @field_validator("sales_trend", mode="before")
    @classmethod
    def normalize_trend(cls, v: Any) -> str:
        if isinstance(v, str):
            val = v.strip().lower()
            if val in {"increasing", "up", "growth", "positive"}:
                return "increasing"
            elif val in {"decreasing", "down", "decline", "negative"}:
                return "decreasing"
        return "stable"


class InventoryData(BaseModel):
    current_inventory: int = Field(default=0, ge=0, description="Units currently physically in warehouse")
    reserved_inventory: int = Field(default=0, ge=0, description="Units allocated to customer orders but not yet shipped")
    incoming_inventory: int = Field(default=0, ge=0, description="Units ordered from supplier in transit")

    @field_validator("current_inventory", "reserved_inventory", "incoming_inventory", mode="before")
    @classmethod
    def ensure_non_negative(cls, v: Any) -> int:
        try:
            val = int(v)
            return max(0, val)
        except (ValueError, TypeError):
            return 0


class SupplierData(BaseModel):
    lead_time_days: int = Field(default=7, gt=0, description="Supplier fulfillment lead time in days")
    minimum_order_quantity: int = Field(default=1, gt=0, description="Supplier MOQ in units")
    reorder_point: Optional[int] = Field(default=None, description="Person 1 reorder point threshold if provided")

    @field_validator("lead_time_days", "minimum_order_quantity", mode="before")
    @classmethod
    def ensure_positive_int(cls, v: Any) -> int:
        try:
            val = int(v)
            return max(1, val)
        except (ValueError, TypeError):
            return 1


class MarketIntelligenceData(BaseModel):
    market_signal: str = Field(default="neutral", description="Market signal: positive, neutral, or negative")
    market_impact: float = Field(default=0.0, ge=0.0, le=1.0, description="Estimated impact magnitude")
    market_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence in market signal")
    market_relevance: float = Field(default=0.0, ge=0.0, le=1.0, description="Relevance to product")
    evidence_strength: float = Field(default=0.0, ge=0.0, le=1.0, description="Evidence strength from external news")
    external_signal_available: bool = Field(default=False, description="Whether live external news was found")
    market_reason: str = Field(default="", description="Summary of market news context")
    key_events: List[str] = Field(default_factory=list, description="Key news events identified")
    risks: List[str] = Field(default_factory=list, description="Identified market risks")
    opportunities: List[str] = Field(default_factory=list, description="Identified market opportunities")

    @field_validator("market_signal", mode="before")
    @classmethod
    def normalize_market_signal(cls, v: Any) -> str:
        if isinstance(v, str):
            val = v.strip().lower()
            if val in {"positive", "bullish", "high"}:
                return "positive"
            elif val in {"negative", "bearish", "low"}:
                return "negative"
        return "neutral"


class ProductHandoff(BaseModel):
    """
    Exact schema matching Person 1's product representation in handoff/person2_handoff.json.
    """
    product_id: str = Field(..., description="Unique product identifier (e.g. P001)")
    product_name: str = Field(default="Unknown Product", description="Product name")
    category: str = Field(default="General", description="Product category")
    brand: str = Field(default="Generic", description="Product brand")
    forecast: ForecastData = Field(default_factory=ForecastData)
    inventory: InventoryData = Field(default_factory=InventoryData)
    supplier: SupplierData = Field(default_factory=SupplierData)
    market_intelligence: Optional[MarketIntelligenceData] = Field(default_factory=MarketIntelligenceData)


class HandoffContainer(BaseModel):
    """Container for the full handoff/person2_handoff.json file."""
    project: str = Field(default="BFWAI")
    handoff_version: str = Field(default="1.0")
    source: str = Field(default="")
    generated_at: str = Field(default="")
    products: List[ProductHandoff] = Field(default_factory=list)


class ProductInput(BaseModel):
    """
    Flexible input model supporting both nested Person 1 format and flattened payloads.
    Used for ad-hoc API decision/analysis requests.
    """
    product_id: str
    product_name: Optional[str] = "Product"
    category: Optional[str] = "General"
    brand: Optional[str] = "Generic"
    
    # Flat or nested fields
    forecast_7d: Optional[int] = None
    forecast_30d: Optional[int] = None
    sales_trend: Optional[str] = None
    
    current_inventory: Optional[int] = None
    reserved_inventory: Optional[int] = None
    incoming_inventory: Optional[int] = None
    
    lead_time_days: Optional[int] = None
    supplier_lead_time: Optional[int] = None
    minimum_order_quantity: Optional[int] = None
    moq: Optional[int] = None
    reorder_point: Optional[int] = None
    
    market_signal: Optional[str] = None
    market_confidence: Optional[float] = None
    market_impact: Optional[float] = None
    market_reason: Optional[str] = None
    
    # Optional nested objects
    forecast: Optional[ForecastData] = None
    inventory: Optional[InventoryData] = None
    supplier: Optional[SupplierData] = None
    market_intelligence: Optional[MarketIntelligenceData] = None

    def to_product_handoff(self) -> ProductHandoff:
        """Converts flexible input into a canonical ProductHandoff instance."""
        # Forecast
        f_7d = self.forecast_7d if self.forecast_7d is not None else (self.forecast.forecast_7d if self.forecast else 0)
        f_30d = self.forecast_30d if self.forecast_30d is not None else (self.forecast.forecast_30d if self.forecast else 0)
        trend = self.sales_trend if self.sales_trend is not None else (self.forecast.sales_trend if self.forecast else "stable")
        forecast_obj = ForecastData(forecast_7d=max(0, f_7d), forecast_30d=max(0, f_30d), sales_trend=trend)

        # Inventory
        curr = self.current_inventory if self.current_inventory is not None else (self.inventory.current_inventory if self.inventory else 0)
        res = self.reserved_inventory if self.reserved_inventory is not None else (self.inventory.reserved_inventory if self.inventory else 0)
        inc = self.incoming_inventory if self.incoming_inventory is not None else (self.inventory.incoming_inventory if self.inventory else 0)
        inv_obj = InventoryData(current_inventory=max(0, curr), reserved_inventory=max(0, res), incoming_inventory=max(0, inc))

        # Supplier
        lt = self.lead_time_days or self.supplier_lead_time or (self.supplier.lead_time_days if self.supplier else 7)
        m_order = self.minimum_order_quantity or self.moq or (self.supplier.minimum_order_quantity if self.supplier else 1)
        rop = self.reorder_point if self.reorder_point is not None else (self.supplier.reorder_point if self.supplier else None)
        sup_obj = SupplierData(lead_time_days=max(1, lt), minimum_order_quantity=max(1, m_order), reorder_point=rop)

        # Market Intelligence
        sig = self.market_signal or (self.market_intelligence.market_signal if self.market_intelligence else "neutral")
        conf = self.market_confidence if self.market_confidence is not None else (self.market_intelligence.market_confidence if self.market_intelligence else 0.0)
        impact = self.market_impact if self.market_impact is not None else (self.market_intelligence.market_impact if self.market_intelligence else 0.0)
        reason = self.market_reason or (self.market_intelligence.market_reason if self.market_intelligence else "")
        events = self.market_intelligence.key_events if self.market_intelligence else []
        risks = self.market_intelligence.risks if self.market_intelligence else []
        opps = self.market_intelligence.opportunities if self.market_intelligence else []
        ext_avail = self.market_intelligence.external_signal_available if self.market_intelligence else bool(self.market_signal)

        mi_obj = MarketIntelligenceData(
            market_signal=sig,
            market_impact=impact,
            market_confidence=conf,
            external_signal_available=ext_avail,
            market_reason=reason,
            key_events=events,
            risks=risks,
            opportunities=opps
        )

        return ProductHandoff(
            product_id=self.product_id,
            product_name=self.product_name or self.product_id,
            category=self.category or "General",
            brand=self.brand or "Generic",
            forecast=forecast_obj,
            inventory=inv_obj,
            supplier=sup_obj,
            market_intelligence=mi_obj
        )


class InventoryMetrics(BaseModel):
    available_inventory: int = Field(..., description="current_inventory - reserved_inventory")
    average_daily_demand: float = Field(..., description="Calculated daily demand rate from forecast")
    days_of_stock: float = Field(..., description="Days until stockout at current demand rate")
    coverage_ratio: float = Field(..., description="days_of_stock / supplier_lead_time")
    stockout_risk: str = Field(..., description="LOW, MEDIUM, or HIGH")
    safety_stock: int = Field(..., description="Calculated buffer stock quantity")
    reorder_point: int = Field(..., description="Trigger point for reordering")
    is_reorder_point_fallback: bool = Field(..., description="True if calculated by Person 2 fallback rather than Person 1")
    recommended_order_quantity: int = Field(..., description="Final order quantity respecting MOQ")
    order_calculation_evidence: Dict[str, Any] = Field(default_factory=dict, description="Step-by-step quantity calculation breakdown")


class DecisionResult(BaseModel):
    product_id: str
    product_name: str
    decision: str = Field(..., description="INCREASE, MAINTAIN, or REDUCE")
    confidence: float = Field(..., description="Decision confidence score (0.0 to 1.0)")
    stockout_risk: str = Field(..., description="LOW, MEDIUM, or HIGH")
    days_of_stock: float
    reorder_point: int
    recommended_order_quantity: int
    decision_score: float = Field(..., description="Overall calculated decision score (-1.0 to 1.0)")
    reasons: List[str] = Field(default_factory=list, description="List of transparent reasons explaining the decision")
    metrics: InventoryMetrics


class AlternativeProduct(BaseModel):
    product_id: str
    product_name: str
    category: str
    brand: str
    sales_trend: str
    forecast_7d: int
    forecast_30d: int
    current_inventory: int
    market_signal: str
    reason: str
    supporting_metrics: Dict[str, Any] = Field(default_factory=dict)


class FullAnalysisResponse(BaseModel):
    product: ProductHandoff
    decision: DecisionResult
    ai_explanation: str
    alternative_products: List[AlternativeProduct] = Field(default_factory=list)


class ApprovalRequest(BaseModel):
    product_id: str
    decision: str
    action: str = Field(..., description="APPROVE, MODIFY, or REJECT")
    modified_quantity: Optional[int] = None
    notes: Optional[str] = None


class ApprovalRecord(BaseModel):
    id: str
    product_id: str
    product_name: Optional[str] = ""
    decision: str
    action: str
    original_recommended_quantity: int
    final_quantity: int
    notes: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
