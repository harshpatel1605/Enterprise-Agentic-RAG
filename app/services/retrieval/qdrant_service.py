import os
import logfire
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import settings
from app.services.retrieval.embedding import embed_query

LOCAL_QDRANT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "local_qdrant"
)

_client = None


def get_qdrant_client() -> QdrantClient:
    """
    Returns a connected QdrantClient.
    Attempts remote cloud endpoint first; falls back to local disk storage
    if the cloud instance is unreachable, unconfigured, or expired.
    """
    global _client
    if _client is not None:
        return _client

    if settings.QDRANT_URL:
        try:
            client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=5)
            client.get_collections()
            _client = client
            return _client
        except Exception as e:
            logfire.warning(f"⚠️ Remote Qdrant connection failed ({e}). Falling back to local storage at {LOCAL_QDRANT_PATH}.")

    os.makedirs(LOCAL_QDRANT_PATH, exist_ok=True)
    try:
        _client = QdrantClient(path=LOCAL_QDRANT_PATH)
        logfire.info(f"📂 Local Qdrant storage active at {LOCAL_QDRANT_PATH}.")
    except Exception as e:
        logfire.warning(f"⚠️ Local Qdrant path lock ({e}). Using in-memory vector store.")
        _client = QdrantClient(location=":memory:")
    return _client


# Default client accessor for backward compatibility
client = get_qdrant_client()


def search_enterprise_knowledge(query: str, limit: int = 8):
    """
    Performs a high-precision search in the enterprise knowledge base.
    Uses the modern query_points interface.
    """
    try:
        q_client = get_qdrant_client()
        if not q_client.collection_exists(settings.QDRANT_COLLECTION):
            logfire.warning(f"Collection '{settings.QDRANT_COLLECTION}' does not exist yet. Run ingestion first.")
            return []

        query_vector = embed_query(query)

        response = q_client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=query_vector,
            limit=limit,
            with_payload=True
        )

        results = []
        for res in response.points:
            results.append({
                "content": res.payload.get("text", ""),
                "source": res.payload.get("source", "Unknown"),
                "score": res.score
            })

        return results
    except Exception as e:
        logfire.error(f"❌ Qdrant Search Failed: {e}")
        return []
