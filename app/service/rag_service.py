import logging


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# RAG Service
# ---------------------------------------------------------

class RAGService:

    def __init__(self, retriever, generator, top_k=5):
        """
        Application-level orchestration for the RAG system.

        Parameters
        ----------
        retriever:
            Retriever instance responsible for finding
            relevant chunks.

        generator:
            Generator instance responsible for generating
            an answer from retrieved evidence.

        top_k:
            Number of chunks to retrieve for each query.
        """

        # -------------------------------------------------
        # Validate configuration
        # -------------------------------------------------

        if not isinstance(top_k, int):
            raise TypeError(
                "top_k must be an integer"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    # -----------------------------------------------------
    # Ask
    # -----------------------------------------------------

    def ask(self, query):
        """
        Run the complete RAG workflow.

        Flow:

            query
              ↓
            Retriever
              ↓
        retrieved chunks
              ↓
            Generator
              ↓
        answer + sources
        """

        # -------------------------------------------------
        # Validate query
        # -------------------------------------------------

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string"
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "query cannot be empty"
            )

        logger.info(
            f"Processing RAG query: {query}"
        )

        # -------------------------------------------------
        # Retrieve evidence
        # -------------------------------------------------

        retrieved_chunks = self.retriever.retrieve(
            query=query,
            top_k=self.top_k
        )

        logger.info(
            f"Retrieved {len(retrieved_chunks)} chunks"
        )

        # -------------------------------------------------
        # Generate grounded answer
        # -------------------------------------------------

        result = self.generator.generate(
            query=query,
            retrieved_chunks=retrieved_chunks
        )

        logger.info(
            "RAG query completed successfully"
        )

        return result