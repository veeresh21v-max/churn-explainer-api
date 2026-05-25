from pydantic import BaseModel
from typing import Literal

# ── Input — customer features ──────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    customer_id:          int
    days_since_purchase:  int
    # How many days since the customer last bought something
    total_purchases:      int
    # Total number of purchases ever made
    support_tickets:      int
    # Number of support tickets raised in last 30 days
    avg_order_value:      float
    # Average spend per order in INR
    completed_onboarding: bool
    # Whether customer completed onboarding in first 7 days

# ── ML prediction result ───────────────────────────────────────────────────
class PredictionResult(BaseModel):
    customer_id:       int
    churn_probability: float
    risk_level:        Literal["HIGH", "MEDIUM", "LOW"]

# ── LLM explanation result ─────────────────────────────────────────────────
class ChurnExplanation(BaseModel):
    customer_id:        int
    churn_probability:  float
    risk_level:         Literal["HIGH", "MEDIUM", "LOW"]
    summary:            str
    top_reasons:        list[str]
    recommended_action: str
    input_tokens:       int
    output_tokens:      int