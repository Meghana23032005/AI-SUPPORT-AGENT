"""Base interface for intent classifiers."""
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseIntentClassifier(ABC):
    """Abstract Base Class for all Intent Classifiers."""

    @abstractmethod
    def fit(self, texts: List[str], labels: List[str]) -> "BaseIntentClassifier":
        """Trains the classifier on labeled text examples."""
        pass

    @abstractmethod
    def predict_one(self, text: str) -> Dict[str, Any]:
        """
        Predicts intent for a single text.
        Returns:
            {
                "intent": str,
                "confidence": float,
                "probabilities": Dict[str, float]
            }
        """
        pass

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predicts intent for a list of texts."""
        return [self.predict_one(t) for t in texts]
