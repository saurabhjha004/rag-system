import logging

from app.retrieval.embedder import EmbeddingModel
from app.retrieval.vector_store import VectorStore


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Retriever
# ---------------------------------------------------------

class Retriever:

    def __init__(
        self,
        embedding_model,
        vector_store
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    # -----------------------------------------------------
    # Retrieve relevant chunks
    # -----------------------------------------------------

    def retrieve(
        self,
        query,
        top_k=5
    ):
        """
        Convert a query into an embedding and retrieve
        the most relevant chunks.
        """

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string"
            )

        if not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        logger.info(
            f"Retrieving results for query: {query}"
        )

        # Query text → query vector
        query_embedding = (
            self.embedding_model
            .embed_text(query)
        )

        # Query vector → relevant chunks
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )

        return results