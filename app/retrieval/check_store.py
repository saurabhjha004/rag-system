import logging

from vector_store import VectorStore


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Check Chroma database
# ---------------------------------------------------------

if __name__ == "__main__":

    vector_store = VectorStore()

    # Connect to existing Chroma database
    vector_store.load()

    # Read number of stored records
    count = vector_store.count()

    logger.info(
        f"Records currently stored in Chroma: {count}"
    )

    # Expected number for our current document
    expected_count = 53

    if count == expected_count:
        logger.info(
            "✓ Chroma contains all 53 chunk records"
        )
    else:
        logger.warning(
            f"Expected {expected_count} records, "
            f"but found {count}"
        )