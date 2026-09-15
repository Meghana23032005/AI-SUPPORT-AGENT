"""Trivial baseline classifier (Majority class / naive heuristic)."""
from typing import Dict, Any, List
from collections import Counter
from src.classifiers.base import BaseIntentClassifier
from src.taxonomy.intent_definitions import IntentCategory, ALL_INTENTS, DEFAULT_INTENT

class TrivialIntentClassifier(BaseIntentClassifier):
    """
    Baseline 1: Trivial Intent Classifier.
    Predicts the majority class observed during training with empirical prior probability.
    """
    def __init__(self, default_intent: str = IntentCategory.IOS_UPDATE_SYSTEM_CRASH.value):
        self.majority_intent = default_intent
        self.majority_confidence = 0.50
        self.priors: Dict[str, float] = {i: 1.0 / len(ALL_INTENTS) for i in ALL_INTENTS}

    def fit(self, texts: List[str], labels: List[str]) -> "TrivialIntentClassifier":
        if not labels:
            return self
        counts = Counter(labels)
        most_common, freq = counts.most_common(1)[0]
        self.majority_intent = most_common
        total = len(labels)
        self.majority_confidence = freq / total
        self.priors = {i: counts.get(i, 0) / total for i in ALL_INTENTS}
        return self

    def predict_one(self, text: str) -> Dict[str, Any]:
        return {
            "intent": self.majority_intent,
            "confidence": round(self.majority_confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.priors.items()}
        }
