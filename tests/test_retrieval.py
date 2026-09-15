"""Unit tests for historical conversation vector store and retrieval grounding."""
from src.retrieval.vector_store import ConversationVectorStore
from src.retrieval.retriever import HistoricalSupportRetriever

def test_vector_store_indexing_and_search():
    sample_convs = [
        {
            "conversation_id": "conv_001",
            "customer_message_clean": "Battery plugged in not charging on Inspiron",
            "brand_response_clean": "Please disconnect adapter, remove battery and hold power button 15 seconds.",
            "intent": "HARDWARE_POWER_BOOT",
            "created_at": "2017-10-31"
        },
        {
            "conversation_id": "conv_002",
            "customer_message_clean": "Where is my delivery tracking number?",
            "brand_response_clean": "You can check order tracking at dell.com/orders.",
            "intent": "ORDER_SHIPPING_DELIVERY",
            "created_at": "2017-10-31"
        }
    ]

    store = ConversationVectorStore()
    store.index_conversations(sample_convs)
    assert len(store.documents) == 2

    # Search
    results = store.search("battery won't charge", top_k=1)
    assert len(results) == 1
    assert results[0]["conversation_id"] == "conv_001"
    assert results[0]["similarity_score"] > 0.1

def test_retriever_formatting():
    store = ConversationVectorStore()
    store.index_conversations([
        {
            "conversation_id": "conv_100",
            "customer_message_clean": "Need service tag",
            "brand_response_clean": "Service tag is on sticker at bottom.",
            "intent": "WARRANTY_SERVICE_TAG"
        }
    ])
    retriever = HistoricalSupportRetriever(store)
    matches = retriever.retrieve("Where is service tag?")
    assert len(matches) == 1
    prompt_str = retriever.format_for_prompt(matches)
    assert "Historical Dell Resolution:" in prompt_str
    assert "Service tag is on sticker at bottom." in prompt_str
