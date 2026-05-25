from models import CustomerFeatures, PredictionResult

def predict_churn(customer: CustomerFeatures) -> PredictionResult:
    """
    Simulated ML churn prediction model.

    In a real project this would load a trained scikit-learn model:
        model = joblib.load("churn_model.pkl")
        probability = model.predict_proba([[features]])[0][1]

    For this portfolio project we simulate the prediction using
    a rule-based scoring system that mimics real churn patterns.

    Scoring logic based on real churn research:
    - Long time since last purchase = high risk
    - Many support tickets = high risk
    - Few total purchases = high risk
    - Did not complete onboarding = high risk
    - High average order value = lower risk (invested customer)
    """
    score = 0.0
    # Start with base score of 0

    # Days since last purchase — strongest churn signal
    if customer.days_since_purchase >= 60:
        score += 0.35
    elif customer.days_since_purchase >= 30:
        score += 0.20
    elif customer.days_since_purchase >= 14:
        score += 0.10

    # Support tickets — frustrated customers churn
    if customer.support_tickets >= 4:
        score += 0.25
    elif customer.support_tickets >= 2:
        score += 0.15
    elif customer.support_tickets == 1:
        score += 0.05

    # Total purchases — loyal customers are less likely to churn
    if customer.total_purchases <= 2:
        score += 0.20
    elif customer.total_purchases <= 5:
        score += 0.10

    # Onboarding — customers who completed onboarding churn less
    if not customer.completed_onboarding:
        score += 0.15

    # High average order value = invested customer = lower churn risk
    if customer.avg_order_value >= 2000:
        score -= 0.10
    elif customer.avg_order_value >= 1000:
        score -= 0.05

    # Clamp score between 0 and 1
    churn_probability = round(min(max(score, 0.0), 1.0), 2)

    # Determine risk level
    if churn_probability >= 0.65:
        risk_level = "HIGH"
    elif churn_probability >= 0.35:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return PredictionResult(
        customer_id=customer.customer_id,
        churn_probability=churn_probability,
        risk_level=risk_level,
    )