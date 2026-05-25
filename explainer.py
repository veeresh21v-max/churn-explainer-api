import os
import json
from dotenv import load_dotenv
from groq import Groq
from pydantic import ValidationError
from models import CustomerFeatures, PredictionResult, ChurnExplanation
from rag_retriever import retrieve_churn_context

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.3-70b-versatile"

def explain_churn_risk(
    customer:   CustomerFeatures,
    prediction: PredictionResult
) -> ChurnExplanation:
    """
    Generate a plain English explanation of a customer's churn risk.

    Combines ML prediction result with RAG context to produce
    an explanation that non-technical stakeholders can understand.

    customer   : customer features used for prediction
    prediction : ML model output with probability and risk level

    Returns ChurnExplanation with summary, reasons, and action.
    """

    # Step 1 — Build RAG query from customer profile
    rag_query = f"""Customer has not purchased in {customer.days_since_purchase} days,
raised {customer.support_tickets} support tickets,
made {customer.total_purchases} total purchases,
average order value {customer.avg_order_value} INR,
onboarding completed: {customer.completed_onboarding}.
Churn probability: {prediction.churn_probability}.
What are the risk factors and recommended retention actions?"""

    # Step 2 — Retrieve relevant churn knowledge
    context_chunks = retrieve_churn_context(rag_query, top_k=2)

    context = "\n\n".join([
        f"[{chunk['source']}]: {chunk['text']}"
        for chunk in context_chunks
    ]) if context_chunks else "No specific context available."

    # Step 3 — Build explanation prompt
    prompt = f"""You are a data scientist explaining ML churn predictions 
to non-technical business stakeholders.

Customer Profile:
- Customer ID: {customer.customer_id}
- Days since last purchase: {customer.days_since_purchase}
- Total purchases: {customer.total_purchases}
- Support tickets (last 30 days): {customer.support_tickets}
- Average order value: ₹{customer.avg_order_value}
- Completed onboarding: {customer.completed_onboarding}

ML Model Prediction:
- Churn probability: {prediction.churn_probability:.0%}
- Risk level: {prediction.risk_level}

Relevant Research Context:
{context}

Based on the customer profile, ML prediction, and research context above,
provide a JSON response with exactly these fields:
- summary: one sentence risk summary for a business stakeholder
- top_reasons: list of exactly 3 strings, each explaining one risk factor
- recommended_action: one specific actionable recommendation

Return only the JSON object. No explanation. No markdown."""

    # Step 4 — Generate explanation
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=400,
        response_format={"type": "json_object"},
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    raw           = response.choices[0].message.content
    input_tokens  = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens

    # Step 5 — Parse and validate
    try:
        parsed = json.loads(raw)
        return ChurnExplanation(
            customer_id=customer.customer_id,
            churn_probability=prediction.churn_probability,
            risk_level=prediction.risk_level,
            summary=parsed["summary"],
            top_reasons=parsed["top_reasons"],
            recommended_action=parsed["recommended_action"],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    except (json.JSONDecodeError, KeyError, ValidationError) as e:
        raise ValueError(f"Explanation generation failed: {e}. Raw: {raw}")