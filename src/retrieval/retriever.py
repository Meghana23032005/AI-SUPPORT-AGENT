"""Retriever service for retrieving and formatting historical customer support interactions."""
from typing import List, Dict, Any, Optional
from src.retrieval.vector_store import ConversationVectorStore
from src.preprocessing.cleaner import clean_tweet_text

class HistoricalSupportRetriever:
    """
    High-level interface to retrieve historically resolved Dell interactions
    and format them for grounded response generation.
    """
    def __init__(self, vector_store: Optional[ConversationVectorStore] = None):
        self.vector_store = vector_store or ConversationVectorStore()

    def retrieve(
        self,
        query: str,
        predicted_intent: Optional[str] = None,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Retrieves top_k similar conversations."""
        clean_q = clean_tweet_text(query)
        results = self.vector_store.search(
            query=clean_q,
            top_k=top_k,
            filter_intent=predicted_intent
        )
        return results

    def format_for_prompt(self, retrieved_examples: List[Dict[str, Any]]) -> str:
        """Formats retrieved historical examples as a prompt context block."""
        if not retrieved_examples:
            return "No historical examples available."

        formatted_blocks = []
        for i, ex in enumerate(retrieved_examples, 1):
            cust = str(ex.get("customer_message") or "").strip()
            brand = str(ex.get("brand_response") or "").strip()
            score = ex.get("similarity_score", 0.0)
            intent = ex.get("intent", "GENERAL_INQUIRY_FEEDBACK")
            block = (
                f"[Example {i}] (Relevance Score: {score:.2f} | Intent: {intent})\n"
                f"Customer: {cust}\n"
                f"Historical Dell Resolution: {brand}"
            )
            formatted_blocks.append(block)

        return "\n\n".join(formatted_blocks)
