import json
import logging
import pymupdf
from pathlib import Path


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s"
)

logger = logging.getLogger(__name__)


def load_pdf(pdf_path, document_id):
    pdf_path = Path(pdf_path)
    document_name = pdf_path.name

    logger.info(f"Loading document: {document_name}")
    logger.info(f"Document ID: {document_id}")

    try:
        document = pymupdf.open(pdf_path)
    except Exception as e:
        logger.error(f"Failed to load document: {e}")
        raise

    pages_processed = len(document)
    pages_with_text = 0
    pages_without_text = 0

    logger.info(f"Total pages: {pages_processed}")

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text").strip()

        if text:
            pages_with_text += 1
        else:
            pages_without_text += 1
            logger.warning(f"No text found on page: {page_number}")

        pages.append({
            "document_id": document_id,
            "document_name": document_name,
            "page_number": page_number,
            "text": text
        })

    document.close()

    logger.info(f"Pages processed: {pages_processed}")
    logger.info(f"Pages with text: {pages_with_text}")
    logger.info(f"Pages without text: {pages_without_text}")

    return pages


def save_to_json(data, output_path):
    output_path = Path(output_path)

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    logger.info(f"Output saved to: {output_path}")


if __name__ == "__main__":

    pdf_path = "data/documents/Returns of Private Equity v S&P500.pdf"
    output_path = "data/processed/Returns of Private Equity v S&P500.json"

    # Temporary ID for testing.
    # We will decide the proper document ID strategy later.
    document_id = "doc_001"

    data = load_pdf(pdf_path, document_id)

    save_to_json(data, output_path)