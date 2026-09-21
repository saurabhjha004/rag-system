from unittest.mock import Mock

import pytest

from app.service.rag_service import RAGService


# ---------------------------------------------------------
# Valid query
# ---------------------------------------------------------

def test_valid_query_calls_retriever_and_generator():
    retriever = Mock()
    generator = Mock()

    retrieved_chunks = [
        {
            "id": "chunk_001",
            "text": "Some evidence",
            "metadata": {
                "page_number": 1
            }
        }
    ]

    expected_result = {
        "answer": "The study uses a deductive approach.",
        "sources": [
            {
                "chunk_id": "chunk_001"
            }
        ]
    }

    retriever.retrieve.return_value = retrieved_chunks
    generator.generate.return_value = expected_result

    service = RAGService(
        retriever=retriever,
        generator=generator,
        top_k=5
    )

    result = service.ask(
        "  What methodology does the study use?  "
    )

    retriever.retrieve.assert_called_once_with(
        query="What methodology does the study use?",
        top_k=5
    )

    generator.generate.assert_called_once_with(
        query="What methodology does the study use?",
        retrieved_chunks=retrieved_chunks
    )

    assert result is expected_result


# ---------------------------------------------------------
# Empty query
# ---------------------------------------------------------

def test_empty_query_is_rejected():
    retriever = Mock()
    generator = Mock()

    service = RAGService(
        retriever=retriever,
        generator=generator
    )

    with pytest.raises(ValueError, match="query cannot be empty"):
        service.ask("   ")

    retriever.retrieve.assert_not_called()
    generator.generate.assert_not_called()


# ---------------------------------------------------------
# Non-string query
# ---------------------------------------------------------

def test_non_string_query_is_rejected():
    retriever = Mock()
    generator = Mock()

    service = RAGService(
        retriever=retriever,
        generator=generator
    )

    with pytest.raises(TypeError, match="query must be a string"):
        service.ask(123)

    retriever.retrieve.assert_not_called()
    generator.generate.assert_not_called()


# ---------------------------------------------------------
# Invalid top_k
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "top_k, expected_exception, expected_message",
    [
        (0, ValueError, "top_k must be greater than 0"),
        (-5, ValueError, "top_k must be greater than 0"),
        ("5", TypeError, "top_k must be an integer"),
        (5.0, TypeError, "top_k must be an integer"),
    ]
)
def test_invalid_top_k_is_rejected(
    top_k,
    expected_exception,
    expected_message
):
    retriever = Mock()
    generator = Mock()

    with pytest.raises(
        expected_exception,
        match=expected_message
    ):
        RAGService(
            retriever=retriever,
            generator=generator,
            top_k=top_k
        )


# ---------------------------------------------------------
# Generator result is returned unchanged
# ---------------------------------------------------------

def test_generator_result_is_returned_unchanged():
    retriever = Mock()
    generator = Mock()

    retrieved_chunks = [
        {
            "id": "chunk_001",
            "text": "Evidence",
            "metadata": {}
        }
    ]

    generator_result = {
        "answer": "Generated answer",
        "sources": ["chunk_001"]
    }

    retriever.retrieve.return_value = retrieved_chunks
    generator.generate.return_value = generator_result

    service = RAGService(
        retriever=retriever,
        generator=generator
    )

    result = service.ask("What does the paper say?")

    assert result is generator_result