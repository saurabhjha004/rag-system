import json
import logging

import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 32

# Expected dimension for the model we intentionally selected.
# This is used only as a sanity check.
EXPECTED_EMBEDDING_DIMENSION = 384


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Embedding Model
# ---------------------------------------------------------

class EmbeddingModel:

    def __init__(
        self,
        model_name=MODEL_NAME,
        batch_size=DEFAULT_BATCH_SIZE,
        device=None
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device

        self.model = None
        self.embedding_dimension = None

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    def load(self):
        """
        Load the sentence-transformer model and determine
        its actual embedding dimension.
        """

        logger.info(
            f"Loading embedding model: {self.model_name}"
        )

        if self.device:
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
        else:
            self.model = SentenceTransformer(
                self.model_name
            )

        # Model is the source of truth.
        self.embedding_dimension = (
            self.model.get_embedding_dimension()
        )

        logger.info(
            "Embedding model loaded successfully"
        )

        logger.info(
            f"Embedding dimension: "
            f"{self.embedding_dimension}"
        )

        # Optional sanity check for our selected model.
        if (
            self.embedding_dimension
            != EXPECTED_EMBEDDING_DIMENSION
        ):
            raise ValueError(
                f"Unexpected embedding dimension. "
                f"Expected {EXPECTED_EMBEDDING_DIMENSION}, "
                f"got {self.embedding_dimension}"
            )

        return self

    # -----------------------------------------------------
    # Ensure model is loaded
    # -----------------------------------------------------

    def _ensure_loaded(self):

        if self.model is None:
            raise RuntimeError(
                "Embedding model is not loaded. "
                "Call load() first."
            )

    # -----------------------------------------------------
    # Embed one text
    # -----------------------------------------------------

    def embed_text(self, text):
        """
        Convert one text into one embedding vector.
        """

        self._ensure_loaded()

        if not isinstance(text, str):
            raise TypeError(
                "text must be a string"
            )

        if not text.strip():
            raise ValueError(
                "Cannot embed empty text"
            )

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        self._validate_embedding(embedding)

        return embedding

    # -----------------------------------------------------
    # Embed multiple texts
    # -----------------------------------------------------

    def embed_documents(self, texts):
        """
        Convert multiple texts into embedding vectors.

        Input:
            list[str]

        Output:
            numpy array with shape:
            (number_of_texts, embedding_dimension)
        """

        self._ensure_loaded()

        if not isinstance(texts, list):
            raise TypeError(
                "texts must be a list"
            )

        if not texts:
            logger.warning(
                "Received empty text list"
            )

            return np.empty(
                (0, self.embedding_dimension),
                dtype=np.float32
            )

        for index, text in enumerate(texts):

            if not isinstance(text, str):
                raise TypeError(
                    f"Text at index {index} "
                    f"must be a string"
                )

            if not text.strip():
                raise ValueError(
                    f"Text at index {index} "
                    f"is empty"
                )

        logger.info(
            f"Generating embeddings for "
            f"{len(texts)} texts"
        )

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        expected_shape = (
            len(texts),
            self.embedding_dimension
        )

        if embeddings.shape != expected_shape:
            raise ValueError(
                f"Unexpected embedding matrix shape: "
                f"{embeddings.shape}. "
                f"Expected {expected_shape}"
            )

        for index, embedding in enumerate(embeddings):

            self._validate_embedding(
                embedding,
                index
            )

        logger.info(
            f"Generated {len(embeddings)} embeddings"
        )

        return embeddings

    # -----------------------------------------------------
    # Validate one embedding
    # -----------------------------------------------------

    def _validate_embedding(
        self,
        embedding,
        index=None
    ):
        """
        Validate a single embedding vector.
        """

        if not isinstance(
            embedding,
            np.ndarray
        ):
            raise TypeError(
                f"Expected numpy.ndarray, "
                f"got {type(embedding)}"
            )

        expected_shape = (
            self.embedding_dimension,
        )

        if embedding.shape != expected_shape:

            location = (
                f" at index {index}"
                if index is not None
                else ""
            )

            raise ValueError(
                f"Invalid embedding shape"
                f"{location}: "
                f"{embedding.shape}. "
                f"Expected {expected_shape}"
            )

        if not np.issubdtype(
            embedding.dtype,
            np.floating
        ):
            raise TypeError(
                "Embedding must contain "
                "floating point values"
            )

        if not np.all(
            np.isfinite(embedding)
        ):
            raise ValueError(
                "Embedding contains NaN "
                "or infinite values"
            )


# ---------------------------------------------------------
# Run Sprint 3 batch validation
# ---------------------------------------------------------

if __name__ == "__main__":

    # -----------------------------------------------------
    # 1. Initialize and load model
    # -----------------------------------------------------

    model = EmbeddingModel()

    model.load()

    # -----------------------------------------------------
    # 2. Load chunk records
    # -----------------------------------------------------

    input_path = (
        "data/processed/"
        "Returns of Private Equity v S&P500_chunks.json"
    )

    logger.info(
        f"Loading chunks from: {input_path}"
    )

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as file:
        chunks = json.load(file)

    logger.info(
        f"Loaded {len(chunks)} chunks"
    )

    # -----------------------------------------------------
    # 3. Extract text from chunk records
    # -----------------------------------------------------

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # -----------------------------------------------------
    # 4. Generate embeddings
    # -----------------------------------------------------

    embeddings = model.embed_documents(texts)

    # -----------------------------------------------------
    # 5. Final validation
    # -----------------------------------------------------

    logger.info(
        "Running final validation..."
    )

    # Number of embeddings must equal number of chunks.
    if len(embeddings) != len(chunks):
        raise ValueError(
            f"Expected {len(chunks)} embeddings, "
            f"got {len(embeddings)}"
        )

    # Overall shape.
    expected_shape = (
        len(chunks),
        model.embedding_dimension
    )

    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Expected embedding matrix shape "
            f"{expected_shape}, "
            f"got {embeddings.shape}"
        )

    # Check every individual vector.
    for index, embedding in enumerate(embeddings):

        model._validate_embedding(
            embedding,
            index
        )

    logger.info(
        "✓ Number of embeddings: "
        f"{len(embeddings)}"
    )

    logger.info(
        "✓ Embedding matrix shape: "
        f"{embeddings.shape}"
    )

    logger.info(
        "✓ Every embedding has shape: "
        f"({model.embedding_dimension},)"
    )

    logger.info(
        "✓ No missing or malformed embeddings"
    )

    logger.info(
        "Sprint 3 embedding validation completed successfully"
    )