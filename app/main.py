from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.generation.generator import Generator
from app.retrieval.embedder import EmbeddingModel
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore
from app.service.rag_service import RAGService


# ---------------------------------------------------------
# Application Configuration
# ---------------------------------------------------------

TOP_K = 5


# ---------------------------------------------------------
# Application Lifespan
# ---------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize expensive application resources once
    when the FastAPI application starts.
    """

    # -----------------------------------------------------
    # Load embedding model
    # -----------------------------------------------------

    embedding_model = EmbeddingModel()
    embedding_model.load()

    # -----------------------------------------------------
    # Load vector store
    # -----------------------------------------------------

    vector_store = VectorStore()
    vector_store.load()

    # -----------------------------------------------------
    # Create retriever
    # -----------------------------------------------------

    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store
    )

    # -----------------------------------------------------
    # Load generator
    # -----------------------------------------------------

    generator = Generator()
    generator.load()

    # -----------------------------------------------------
    # Create RAG service
    # -----------------------------------------------------

    rag_service = RAGService(
        retriever=retriever,
        generator=generator,
        top_k=TOP_K
    )

    # -----------------------------------------------------
    # Store service on FastAPI application state
    # -----------------------------------------------------

    app.state.rag_service = rag_service

    yield


# ---------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------

app = FastAPI(
    title="RAG System API",
    version="1.0.0",
    lifespan=lifespan
)


# ---------------------------------------------------------
# Register Routes
# ---------------------------------------------------------

app.include_router(router)