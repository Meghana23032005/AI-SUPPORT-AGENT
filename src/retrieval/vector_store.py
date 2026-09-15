"""In-memory vector store for historical customer-support conversation retrieval."""
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ConversationVectorStore:
    """
    Lightweight, deterministic vector store for historical conversation retrieval.
    Embeds customer queries and brand responses using sublinear TF-IDF representation,
    enabling sub-millisecond retrieval with zero external service dependencies.
    """
    def __init__(self, max_features: int = 5000):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=self.max_features,
            sublinear_tf=True,
            stop_words="english"
        )
        self.vectors: Optional[np.ndarray] = None
        self.documents: List[Dict[str, Any]] = []

    def index_conversations(self, conversations: List[Dict[str, Any]]) -> "ConversationVectorStore":
        """Indexes historical conversation records."""
        self.documents = []
        texts_to_embed = []

        for conv in conversations:
            # We index both customer query and resolved brand response to capture semantic matching
            query_text = conv.get("customer_message_clean") or conv.get("customer_message_raw") or ""
            reply_text = conv.get("brand_response_clean") or conv.get("brand_response_raw") or ""
            intent = conv.get("intent", "GENERAL_INQUIRY_FEEDBACK")
            conv_id = conv.get("conversation_id", "")

            # Index document representation
            doc_repr = f"{query_text} {reply_text}".strip()
            if not doc_repr:
                continue

            self.documents.append({
                "conversation_id": conv_id,
                "customer_message": query_text,
                "brand_response": reply_text,
                "intent": intent,
                "created_at": conv.get("created_at", "")
            })
            texts_to_embed.append(doc_repr)

        if texts_to_embed:
            self.vectors = self.vectorizer.fit_transform(texts_to_embed)
        return self

    def search(
        self,
        query: str,
        top_k: int = 3,
        filter_intent: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the top_k most similar historical conversations for a given query.
        Optionally filters by intent.
        """
        if not query or self.vectors is None or len(self.documents) == 0:
            return []

        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.vectors)[0]

        # Candidate indices
        if filter_intent:
            candidate_indices = [
                i for i, doc in enumerate(self.documents)
                if doc["intent"] == filter_intent
            ]
            if not candidate_indices: # Fallback to all if intent-specific is empty
                candidate_indices = list(range(len(self.documents)))
        else:
            candidate_indices = list(range(len(self.documents)))

        # Rank candidates by similarity
        scored_candidates = [(idx, float(sims[idx])) for idx in candidate_indices]
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scored_candidates[:top_k]:
            doc = self.documents[idx].copy()
            doc["similarity_score"] = round(score, 4)
            results.append(doc)

        return results
