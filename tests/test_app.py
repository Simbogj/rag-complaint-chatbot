"""Smoke tests for the Streamlit app module."""

from unittest.mock import MagicMock, patch

from src.rag import RAGResponse
from src.retriever import RetrievedChunk


def test_app_main_renders_without_streamlit_runtime() -> None:
    import app

    mock_response = RAGResponse(
        question="Why are credit card customers unhappy?",
        answer="Customers mention billing disputes and unauthorized charges.",
        sources=[
            RetrievedChunk(
                text="I was charged twice for the same purchase.",
                score=0.88,
                metadata={
                    "product_category": "Credit Card",
                    "issue": "Billing dispute",
                    "company": "TEST BANK",
                    "state": "CA",
                    "complaint_id": "123",
                },
            )
        ],
        model_name="test-model",
    )

    mock_rag = MagicMock()
    mock_rag.ask.return_value = mock_response

    with patch("app.load_rag_pipeline", return_value=mock_rag), patch(
        "app.st"
    ) as mock_st:

        class SessionState:
            def __init__(self):
                self.messages = []

            def __contains__(self, key: str) -> bool:
                return hasattr(self, key)

        mock_st.session_state = SessionState()
        mock_st.sidebar.__enter__ = MagicMock(return_value=mock_st.sidebar)
        mock_st.sidebar.__exit__ = MagicMock(return_value=False)
        mock_st.chat_input.return_value = None

        app.main()

    mock_st.set_page_config.assert_called_once()
    mock_rag.ask.assert_not_called()
