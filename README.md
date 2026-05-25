# Churn Risk Explainer API

A FastAPI application that predicts customer churn probability using an ML model
and generates plain English explanations using RAG + LLM.

## What it does
- Predicts churn probability from customer features (ML model)
- Retrieves relevant churn research context (RAG + ChromaDB)
- Generates plain English risk explanation (Groq LLM)
- Returns risk level, top reasons, and recommended action

## Tech Stack
- FastAPI — API framework
- Groq + Llama 3.3 70b — LLM for explanation generation
- ChromaDB — local vector store for churn knowledge base
- sentence-transformers (all-MiniLM-L6-v2) — free local embeddings
- Pydantic — request/response validation

## Setup
pip install -r requirements.txt
Add GROQ_API_KEY to .env
uvicorn main:app --reload

## Endpoints
- POST /predict — run ML churn prediction only
- POST /explain — predict + generate plain English explanation
- GET /health — health check

## Example Request
POST /explain
{
  "customer_id": 1042,
  "days_since_purchase": 52,
  "total_purchases": 2,
  "support_tickets": 4,
  "avg_order_value": 1500.0,
  "completed_onboarding": false
}

## Example Response
{
  "customer_id": 1042,
  "churn_probability": 0.75,
  "risk_level": "HIGH",
  "summary": "Customer 1042 has a high risk of churning...",
  "top_reasons": [
    "Multiple unresolved support tickets indicate frustration",
    "Failure to complete onboarding increases churn likelihood",
    "Inactivity since last purchase 52 days ago elevates risk"
  ],
  "recommended_action": "Reach out to resolve support tickets and offer onboarding assistance",
  "input_tokens": 308,
  "output_tokens": 135
}

## Architecture
Customer Features → ML Model → Churn Probability
                             → RAG Context Retrieval
                             → LLM Explanation
                             → Structured Risk Report
