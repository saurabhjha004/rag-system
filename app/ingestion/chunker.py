import json
import logging
import re
from pathlib import Path

import tiktoken


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

CHUNK_TARGET_TOKENS = 350
CHUNK_MAX_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 50


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------

# Used only for token counting.
# This can be replaced later with the tokenizer of the
# actual embedding model.
TOKENIZER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text):
    """Return the number of tokens in the given text."""
    return len(TOKENIZER.encode(text))


# ---------------------------------------------------------
# Input validation
# ---------------------------------------------------------

def validate_page_records(page_records):
    """Validate the input produced by loader.py."""

    if not isinstance(page_records, list):
        raise ValueError("page_records must be a list")

    required_fields = {
        "document_id",
        "document_name",
        "page_number",
        "text"
    }

    for index, page in enumerate(page_records):

        if not isinstance(page, dict):
            raise ValueError(
                f"Page record at index {index} must be a dictionary"
            )

        missing_fields = required_fields - page.keys()

        if missing_fields:
            raise ValueError(
                f"Page record at index {index} is missing fields: "
                f"{missing_fields}"
            )

        if not page["document_id"]:
            raise ValueError(
                f"Page record at index {index} has an empty document_id"
            )

        if not page["document_name"]:
            raise ValueError(
                f"Page record at index {index} has an empty document_name"
            )

        if not isinstance(page["page_number"], int):
            raise ValueError(
                f"Page record at index {index} has invalid page_number"
            )

        if page["page_number"] < 1:
            raise ValueError(
                f"Page record at index {index} has invalid page_number"
            )

        if not isinstance(page["text"], str):
            raise ValueError(
                f"Page record at index {index} has non-string text"
            )


# ---------------------------------------------------------
# Text normalization
# ---------------------------------------------------------

def normalize_text(text):
    """
    Normalize whitespace while preserving paragraph boundaries.

    Multiple blank lines are reduced to one paragraph boundary.
    """

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Normalize spaces/tabs inside lines.
    text = re.sub(r"[ \t]+", " ", text)

    # Preserve paragraph boundaries.
    text = re.sub(r"\n[ \t]*\n+", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------
# Paragraph extraction
# ---------------------------------------------------------

def extract_paragraphs(text):
    """Extract non-empty paragraphs from normalized text."""

    normalized_text = normalize_text(text)

    if not normalized_text:
        return []

    paragraphs = re.split(r"\n\s*\n", normalized_text)

    return [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]


# ---------------------------------------------------------
# Sentence extraction
# ---------------------------------------------------------

def split_into_sentences(text):
    """
    Split text approximately at sentence boundaries.

    This intentionally keeps the implementation simple.
    It handles common ., ! and ? sentence endings.
    """

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip()
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ---------------------------------------------------------
# Oversized paragraph handling
# ---------------------------------------------------------

def split_oversized_paragraph(paragraph):
    """
    Split a paragraph larger than CHUNK_MAX_TOKENS.

    Sentences are accumulated until adding another sentence
    would exceed the hard maximum.

    Adjacent sub-chunks receive modest sentence-level overlap.
    """

    sentences = split_into_sentences(paragraph)

    chunks = []
    current_sentences = []
    current_tokens = 0

    for sentence in sentences:

        sentence_tokens = count_tokens(sentence)

        # -------------------------------------------------
        # Pathological case:
        # One sentence itself exceeds the hard maximum.
        # -------------------------------------------------

        if sentence_tokens > CHUNK_MAX_TOKENS:

            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_tokens = 0

            logger.warning(
                "Sentence exceeds maximum token limit; "
                "using fallback token splitting"
            )

            tokens = TOKENIZER.encode(sentence)

            for start in range(
                0,
                len(tokens),
                CHUNK_MAX_TOKENS
            ):
                token_slice = tokens[
                    start:start + CHUNK_MAX_TOKENS
                ]

                chunks.append(
                    TOKENIZER.decode(token_slice)
                )

            continue

        # -------------------------------------------------
        # Normal sentence
        # -------------------------------------------------

        if (
            current_tokens + sentence_tokens
            <= CHUNK_MAX_TOKENS
        ):
            current_sentences.append(sentence)
            current_tokens += sentence_tokens

        else:
            chunks.append(" ".join(current_sentences))

            # -------------------------------------------------
            # Sentence-level overlap.
            #
            # Instead of blindly taking the last 50 tokens,
            # preserve whole sentences where possible.
            # -------------------------------------------------

            overlap_sentences = []
            overlap_tokens = 0

            for previous_sentence in reversed(current_sentences):

                previous_tokens = count_tokens(
                    previous_sentence
                )

                if (
                    overlap_tokens + previous_tokens
                    > CHUNK_OVERLAP_TOKENS
                ):
                    break

                overlap_sentences.insert(
                    0,
                    previous_sentence
                )

                overlap_tokens += previous_tokens

            current_sentences = (
                overlap_sentences + [sentence]
            )

            current_tokens = (
                overlap_tokens + sentence_tokens
            )

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


# ---------------------------------------------------------
# Chunk assembly
# ---------------------------------------------------------

def assemble_chunks(paragraphs, stats):
    """
    Combine consecutive paragraphs without exceeding
    CHUNK_MAX_TOKENS.
    """

    chunks = []

    current_paragraphs = []
    current_tokens = 0

    for paragraph in paragraphs:

        paragraph_tokens = count_tokens(paragraph)

        # -------------------------------------------------
        # Oversized paragraph
        # -------------------------------------------------

        if paragraph_tokens > CHUNK_MAX_TOKENS:

            # Emit any accumulated normal paragraphs first.
            if current_paragraphs:

                chunks.append(
                    "\n\n".join(current_paragraphs)
                )

                current_paragraphs = []
                current_tokens = 0

            stats["oversized_paragraphs_split"] += 1

            oversized_chunks = split_oversized_paragraph(
                paragraph
            )

            chunks.extend(oversized_chunks)

            continue

        # -------------------------------------------------
        # Normal paragraph
        # -------------------------------------------------

        if (
            current_tokens + paragraph_tokens
            <= CHUNK_MAX_TOKENS
        ):
            current_paragraphs.append(paragraph)
            current_tokens += paragraph_tokens

        else:

            if current_paragraphs:
                chunks.append(
                    "\n\n".join(current_paragraphs)
                )

            current_paragraphs = [paragraph]
            current_tokens = paragraph_tokens

    # -----------------------------------------------------
    # Remaining paragraphs
    # -----------------------------------------------------

    if current_paragraphs:
        chunks.append(
            "\n\n".join(current_paragraphs)
        )

    return chunks


# ---------------------------------------------------------
# Metadata
# ---------------------------------------------------------

def create_chunk_record(
    document_id,
    document_name,
    page_number,
    chunk_index,
    text
):
    """Create one chunk record."""

    return {
        "chunk_id": f"{document_id}_c{chunk_index:04d}",
        "document_id": document_id,
        "document_name": document_name,
        "page_number": page_number,
        "chunk_index": chunk_index,
        "text": text
    }


# ---------------------------------------------------------
# Main chunking function
# ---------------------------------------------------------

def chunk_pages(page_records):
    """
    Transform page records into chunk records.
    """

    validate_page_records(page_records)

    chunks = []

    stats = {
        "documents": set(),
        "pages_processed": 0,
        "empty_pages": 0,
        "chunks_created": 0,
        "oversized_paragraphs_split": 0,
        "total_tokens": 0,
        "min_tokens": None,
        "max_tokens": None
    }

    # Keep chunk index sequential within each document.
    document_chunk_indices = {}

    for page in page_records:

        document_id = page["document_id"]
        document_name = page["document_name"]
        page_number = page["page_number"]
        text = page["text"]

        stats["documents"].add(document_id)
        stats["pages_processed"] += 1

        normalized_text = normalize_text(text)

        if not normalized_text:
            stats["empty_pages"] += 1

            logger.warning(
                f"No text found on page: "
                f"{document_name} - page {page_number}"
            )

            continue

        paragraphs = extract_paragraphs(
            normalized_text
        )

        page_chunks = assemble_chunks(
            paragraphs,
            stats
        )

        if document_id not in document_chunk_indices:
            document_chunk_indices[document_id] = 0

        for chunk_text in page_chunks:

            chunk_text = chunk_text.strip()

            if not chunk_text:
                continue

            token_count = count_tokens(chunk_text)

            # Hard safety check.
            if token_count > CHUNK_MAX_TOKENS:
                raise ValueError(
                    f"Chunk exceeds maximum token limit: "
                    f"{token_count} tokens"
                )

            chunk_index = document_chunk_indices[
                document_id
            ]

            chunk = create_chunk_record(
                document_id=document_id,
                document_name=document_name,
                page_number=page_number,
                chunk_index=chunk_index,
                text=chunk_text
            )

            chunks.append(chunk)

            document_chunk_indices[document_id] += 1

            # Statistics
            stats["chunks_created"] += 1
            stats["total_tokens"] += token_count

            if (
                stats["min_tokens"] is None
                or token_count < stats["min_tokens"]
            ):
                stats["min_tokens"] = token_count

            if (
                stats["max_tokens"] is None
                or token_count > stats["max_tokens"]
            ):
                stats["max_tokens"] = token_count

    # Convert set to count-friendly value.
    stats["documents_processed"] = len(
        stats["documents"]
    )

    del stats["documents"]

    return chunks, stats


# ---------------------------------------------------------
# Chunk validation
# ---------------------------------------------------------

def validate_chunks(chunks):
    """Validate chunk output."""

    seen_ids = set()
    document_indices = {}

    for chunk in chunks:

        required_fields = {
            "chunk_id",
            "document_id",
            "document_name",
            "page_number",
            "chunk_index",
            "text"
        }

        missing_fields = required_fields - chunk.keys()

        if missing_fields:
            raise ValueError(
                f"Chunk is missing fields: {missing_fields}"
            )

        # No empty chunks.
        if not chunk["text"].strip():
            raise ValueError(
                f"Empty chunk found: {chunk['chunk_id']}"
            )

        # Unique IDs.
        if chunk["chunk_id"] in seen_ids:
            raise ValueError(
                f"Duplicate chunk ID: {chunk['chunk_id']}"
            )

        seen_ids.add(chunk["chunk_id"])

        # Hard token limit.
        token_count = count_tokens(chunk["text"])

        if token_count > CHUNK_MAX_TOKENS:
            raise ValueError(
                f"Chunk {chunk['chunk_id']} has "
                f"{token_count} tokens"
            )

        # Sequential indices per document.
        document_id = chunk["document_id"]
        chunk_index = chunk["chunk_index"]

        expected_index = document_indices.get(
            document_id,
            0
        )

        if chunk_index != expected_index:
            raise ValueError(
                f"Invalid chunk index for "
                f"{document_id}: expected "
                f"{expected_index}, got {chunk_index}"
            )

        document_indices[document_id] = (
            expected_index + 1
        )


# ---------------------------------------------------------
# Statistics logging
# ---------------------------------------------------------

def log_chunking_stats(stats):
    """Log final chunking statistics."""

    chunks_created = stats["chunks_created"]

    if chunks_created > 0:
        average_tokens = (
            stats["total_tokens"] / chunks_created
        )
    else:
        average_tokens = 0

    logger.info(
        f"Documents processed: "
        f"{stats['documents_processed']}"
    )

    logger.info(
        f"Pages processed: "
        f"{stats['pages_processed']}"
    )

    logger.info(
        f"Empty pages: "
        f"{stats['empty_pages']}"
    )

    logger.info(
        f"Chunks created: "
        f"{chunks_created}"
    )

    logger.info(
        f"Average chunk size: "
        f"{average_tokens:.1f} tokens"
    )

    logger.info(
        f"Minimum chunk size: "
        f"{stats['min_tokens'] or 0} tokens"
    )

    logger.info(
        f"Maximum chunk size: "
        f"{stats['max_tokens'] or 0} tokens"
    )

    logger.info(
        f"Oversized paragraphs split: "
        f"{stats['oversized_paragraphs_split']}"
    )


# ---------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------

def load_page_json(input_path):
    """Load page records generated by loader.py."""

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def save_chunk_json(chunks, output_path):
    """Save chunk records to JSON."""

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            chunks,
            file,
            indent=4,
            ensure_ascii=False
        )

    logger.info(
        f"Chunk output saved to: {output_path}"
    )


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":

    input_path = (
        "data/processed/"
        "Returns of Private Equity v S&P500.json"
    )

    output_path = (
        "data/processed/"
        "Returns of Private Equity v S&P500_chunks.json"
    )

    page_records = load_page_json(input_path)

    chunks, stats = chunk_pages(page_records)

    validate_chunks(chunks)

    log_chunking_stats(stats)

    save_chunk_json(
        chunks,
        output_path
    )