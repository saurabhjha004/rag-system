import logging
import os

from groq import Groq


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_NAME = "openai/gpt-oss-20b"
MAX_COMPLETION_TOKENS = 512


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Generator
# ---------------------------------------------------------

class Generator:

    def __init__(
        self,
        model_name=MODEL_NAME
    ):
        self.model_name = model_name
        self.client = None

    # -----------------------------------------------------
    # Load Groq client
    # -----------------------------------------------------

    def load(self):
        """
        Initialize the Groq client using GROQ_API_KEY.
        """

        api_key = os.environ.get("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set."
            )

        logger.info(
            f"Initializing Groq generator: "
            f"{self.model_name}"
        )

        self.client = Groq(
            api_key=api_key
        )

        logger.info(
            "Groq client initialized successfully"
        )

        return self

    # -----------------------------------------------------
    # Ensure client is loaded
    # -----------------------------------------------------

    def _ensure_loaded(self):

        if self.client is None:
            raise RuntimeError(
                "Generator is not loaded. "
                "Call load() first."
            )

    # -----------------------------------------------------
    # Validate inputs
    # -----------------------------------------------------

    def _validate_inputs(
        self,
        query,
        retrieved_chunks
    ):
        """
        Validate the query and retrieved chunks.
        """

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string"
            )

        if not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        if not isinstance(
            retrieved_chunks,
            list
        ):
            raise TypeError(
                "retrieved_chunks must be a list"
            )

        for index, chunk in enumerate(
            retrieved_chunks
        ):

            if not isinstance(
                chunk,
                dict
            ):
                raise TypeError(
                    f"Retrieved chunk at index "
                    f"{index} must be a dictionary"
                )

            required_fields = {
                "id",
                "text",
                "metadata"
            }

            missing_fields = (
                required_fields - chunk.keys()
            )

            if missing_fields:
                raise ValueError(
                    f"Retrieved chunk at index "
                    f"{index} is missing: "
                    f"{missing_fields}"
                )

            if not isinstance(
                chunk["text"],
                str
            ):
                raise TypeError(
                    f"Chunk text at index "
                    f"{index} must be a string"
                )

            if not chunk["text"].strip():
                raise ValueError(
                    f"Chunk at index {index} "
                    f"has empty text"
                )

            # -------------------------------------------------
            # Metadata must be a dictionary.
            # -------------------------------------------------

            if not isinstance(
                chunk["metadata"],
                dict
            ):
                raise TypeError(
                    f"Metadata at index {index} "
                    f"must be a dictionary"
                )

    # -----------------------------------------------------
    # Build grounded prompt
    # -----------------------------------------------------

    def _build_prompt(
        self,
        query,
        retrieved_chunks
    ):
        """
        Build a grounded prompt.

        Each source includes its chunk ID so there is a
        deterministic mapping between prompt evidence
        and stored chunk metadata.
        """

        evidence_sections = []

        for index, chunk in enumerate(
            retrieved_chunks,
            start=1
        ):

            metadata = chunk["metadata"]

            document_name = metadata.get(
                "document_name",
                "Unknown document"
            )

            page_number = metadata.get(
                "page_number",
                "Unknown"
            )

            chunk_id = chunk["id"]

            evidence_sections.append(
                (
                    f"Source {index}\n"
                    f"Chunk ID: {chunk_id}\n"
                    f"Document: {document_name}\n"
                    f"Page: {page_number}\n\n"
                    f"{chunk['text']}"
                )
            )

        evidence = "\n\n".join(
            evidence_sections
        )

        system_prompt = """
You are a grounded question-answering assistant.

Answer the user's question using only the provided context.

Rules:
- Do not use outside knowledge.
- Do not invent facts.
- If the context does not contain enough information to answer
  the question, clearly say that the provided context is insufficient.
- Give a clear, complete, and concise answer.
""".strip()

        user_prompt = (
            f"Question:\n"
            f"{query}\n\n"
            f"Context:\n"
            f"{evidence}"
        )

        return system_prompt, user_prompt

    # -----------------------------------------------------
    # Generate answer
    # -----------------------------------------------------

    def generate(
        self,
        query,
        retrieved_chunks
    ):
        """
        Generate a grounded answer using Groq.

        Parameters
        ----------
        query:
            User question.

        retrieved_chunks:
            Clean application-level retrieval results.

        Returns
        -------
        dict
            {
                "answer": str,
                "sources": [...]
            }

        Note:
            sources represent the retrieved chunks supplied
            to the generator as evidence. They do not claim
            which chunks the model internally relied upon.
        """

        self._ensure_loaded()

        self._validate_inputs(
            query,
            retrieved_chunks
        )

        # -------------------------------------------------
        # No retrieved evidence
        # -------------------------------------------------

        if not retrieved_chunks:

            logger.warning(
                "No retrieved chunks provided"
            )

            return {
                "answer": (
                    "The provided evidence is not "
                    "sufficient to answer this question."
                ),
                "sources": []
            }

        logger.info(
            f"Generating answer using "
            f"{len(retrieved_chunks)} retrieved chunks"
        )

        # -------------------------------------------------
        # Build grounded prompt
        # -------------------------------------------------

        system_prompt, user_prompt = (
            self._build_prompt(
                query,
                retrieved_chunks
            )
        )

        # -------------------------------------------------
        # Call Groq
        # -------------------------------------------------

        try:

            response = (
                self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ],
                    temperature=0.2,
                    max_completion_tokens=(
                        MAX_COMPLETION_TOKENS
                    ),
                    include_reasoning=False
                )
            )

        except Exception as error:

            logger.error(
                f"Groq generation failed: {error}"
            )

            raise

        # -------------------------------------------------
        # Extract answer
        # -------------------------------------------------

        answer = (
            response
            .choices[0]
            .message
            .content
        )

        if not answer or not answer.strip():

            raise RuntimeError(
                "Groq returned an empty answer"
            )

        answer = answer.strip()

        # -------------------------------------------------
        # Build source information
        # -------------------------------------------------

        sources = []

        for chunk in retrieved_chunks:

            metadata = chunk["metadata"]

            sources.append({
                "chunk_id": chunk["id"],
                "document_id": metadata.get(
                    "document_id"
                ),
                "document_name": metadata.get(
                    "document_name"
                ),
                "page_number": metadata.get(
                    "page_number"
                ),
                "chunk_index": metadata.get(
                    "chunk_index"
                )
            })

        logger.info(
            "Answer generated successfully"
        )

        return {
            "answer": answer,
            "sources": sources
        }