import chromadb
from sentence_transformers import SentenceTransformer

# ── Initialize ─────────────────────────────────────────────────────────────
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client   = chromadb.PersistentClient(path="./chroma_db")

COLLECTION_NAME    = "churn_knowledge"
SIMILARITY_THRESHOLD = 0.3

# ── Churn knowledge base ───────────────────────────────────────────────────
CHURN_DOCS = [
    {
        "id":   "churn_001",
        "text": "Customers who have not made a purchase in 45 or more days are considered high churn risk. Immediate outreach is recommended before they disengage completely.",
        "source": "churn_playbook"
    },
    {
        "id":   "churn_002",
        "text": "Customers with 3 or more support tickets in the last 30 days are 4x more likely to churn than customers with no tickets. Unresolved support issues are a leading churn driver.",
        "source": "churn_research"
    },
    {
        "id":   "churn_003",
        "text": "Offering a 10 to 15 percent discount to high-risk customers has shown to reduce churn by 23 percent in e-commerce businesses. Personalized offers work better than generic promotions.",
        "source": "retention_strategies"
    },
    {
        "id":   "churn_004",
        "text": "Customers who completed onboarding within the first 7 days have 60 percent lower churn rate than those who did not. Poor onboarding is one of the strongest early churn predictors.",
        "source": "onboarding_research"
    },
    {
        "id":   "churn_005",
        "text": "High average order value customers who churn represent significant revenue loss. Priority retention efforts should focus on customers spending above 1000 INR per order.",
        "source": "revenue_analysis"
    },
    {
        "id":   "churn_006",
        "text": "Customers with fewer than 3 total purchases are significantly more likely to churn than loyal repeat customers. Early engagement programs in the first 90 days reduce churn by 40 percent.",
        "source": "loyalty_research"
    },
]

def initialize_knowledge_base():
    """
    Index churn knowledge documents into ChromaDB.
    Called once when the application starts.
    Skips indexing if collection already exists with data.
    """
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    if collection.count() == 0:
        print("Indexing churn knowledge base...")
        texts      = [doc["text"] for doc in CHURN_DOCS]
        embeddings = embedding_model.encode(texts).tolist()

        collection.add(
            ids=[doc["id"] for doc in CHURN_DOCS],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"source": doc["source"]} for doc in CHURN_DOCS]
        )
        print(f"Indexed {len(CHURN_DOCS)} churn knowledge chunks.")
    else:
        print(f"Churn knowledge base ready — {collection.count()} chunks.")

    return collection

def retrieve_churn_context(query: str, top_k: int = 2) -> list[dict]:
    """
    Retrieve relevant churn knowledge chunks for a given query.

    query  : question about the customer's churn risk
    top_k  : number of chunks to retrieve

    Returns list of relevant chunks above similarity threshold.
    """
    collection     = chroma_client.get_collection(name=COLLECTION_NAME)
    query_embedding = embedding_model.encode([query]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances", "metadatas"]
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        similarity = round(1 - results["distances"][0][i], 4)
        if similarity >= SIMILARITY_THRESHOLD:
            chunks.append({
                "text":       results["documents"][0][i],
                "similarity": similarity,
                "source":     results["metadatas"][0][i]["source"]
            })

    return chunks