import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from models import CustomerFeatures, PredictionResult, ChurnExplanation
from churn_model import predict_churn
from explainer import explain_churn_risk
from rag_retriever import initialize_knowledge_base

load_dotenv()
app = FastAPI(title="Churn Risk Explainer API")

# Initialize churn knowledge base when server starts
initialize_knowledge_base()

# ── Endpoints ──────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictionResult)
async def predict(customer: CustomerFeatures):
    """
    Run churn prediction on customer features.
    Returns probability and risk level.
    No LLM involved — pure ML prediction.
    """
    return predict_churn(customer)

@app.post("/explain", response_model=ChurnExplanation)
async def explain(customer: CustomerFeatures):
    """
    Run churn prediction AND generate plain English explanation.
    Uses RAG to retrieve relevant churn research context.
    Returns complete risk report for business stakeholders.
    """
    try:
        prediction  = predict_churn(customer)
        explanation = explain_churn_risk(customer, prediction)
        return explanation
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "model": "llama-3.3-70b-versatile"}

# Run: uvicorn main:app --reload
# Test: http://127.0.0.1:8000/docs