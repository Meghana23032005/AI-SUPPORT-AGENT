"""Unified AI Customer Support Agent for Apple Support."""
from typing import Dict, Any, Optional
from src.classifiers.base import BaseIntentClassifier
from src.retrieval.retriever import HistoricalSupportRetriever
from src.generation.escalation import EscalationDecisionEngine
from src.generation.generator import GroundedReplyGenerator
from src.preprocessing.cleaner import clean_tweet_text

class AppleSupportAgent:
    """
    End-to-end AI Support Agent orchestrating Intent Classification,
    Historical Conversation Retrieval, Grounded Reply Generation,
    and Auditable Escalation Policy for Apple Support.
    """
    def __init__(
        self,
        classifier: BaseIntentClassifier,
        retriever: HistoricalSupportRetriever,
        escalation_engine: Optional[EscalationDecisionEngine] = None,
        generator: Optional[GroundedReplyGenerator] = None
    ):
        self.classifier = classifier
        self.retriever = retriever
        self.escalation_engine = escalation_engine or EscalationDecisionEngine()
        self.generator = generator or GroundedReplyGenerator()

    def process_message(self, raw_customer_message: str) -> Dict[str, Any]:
        """
        Executes complete support agent pipeline for an incoming message.
        """
        # 1. Clean message
        clean_text = clean_tweet_text(raw_customer_message)

        # 2. Classify intent
        cls_result = self.classifier.predict_one(clean_text)
        predicted_intent = cls_result["intent"]
        confidence = cls_result.get("confidence", 0.0)
        is_ambiguous = cls_result.get("is_ambiguous", False)

        # 3. Retrieve historically resolved conversations
        retrieved_examples = self.retriever.retrieve(
            query=clean_text,
            predicted_intent=predicted_intent,
            top_k=3
        )

        # 4. Decide AUTO_HANDLE vs ESCALATE
        escalation_result = self.escalation_engine.evaluate(
            customer_message=clean_text,
            predicted_intent=predicted_intent,
            intent_confidence=confidence,
            is_ambiguous=is_ambiguous,
            retrieved_examples=retrieved_examples
        )

        # 5. Generate Grounded Reply
        gen_result = self.generator.generate(
            customer_message=clean_text,
            predicted_intent=predicted_intent,
            retrieved_examples=retrieved_examples,
            escalation_result=escalation_result
        )

        return {
            "customer_message_raw": raw_customer_message,
            "customer_message_clean": clean_text,
            "intent": predicted_intent,
            "confidence": confidence,
            "is_ambiguous": is_ambiguous,
            "all_probabilities": cls_result.get("probabilities", {}),
            "retrieved_examples": retrieved_examples,
            "decision": escalation_result["decision"],
            "escalation_reason": escalation_result["reason"],
            "evidence": escalation_result["evidence"],
            "escalation_confidence": escalation_result["confidence"],
            "generated_response": gen_result["reply"],
            "generation_method": gen_result["generation_method"]
        }

# Aliases for compatibility
DellSupportAgent = AppleSupportAgent
SupportAgent = AppleSupportAgent
