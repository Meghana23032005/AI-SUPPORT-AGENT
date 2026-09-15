"""Simple ML Baseline Intent Classifier using TF-IDF and Complement Naive Bayes."""
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from src.classifiers.base import BaseIntentClassifier
from src.taxonomy.intent_definitions import ALL_INTENTS, DEFAULT_INTENT

class MLIntentClassifier(BaseIntentClassifier):
    """
    Baseline 2: Simple ML Intent Classifier.
    Uses TF-IDF feature extraction (unigrams + bigrams) paired with
    Complement Naive Bayes (specifically optimized for imbalanced text corpora).
    """
    def __init__(self, alpha: float = 0.5, max_features: int = 3000):
        self.alpha = alpha
        self.max_features = max_features
        self.pipeline: Optional[Pipeline] = None
        self.classes_: List[str] = ALL_INTENTS

    def fit(self, texts: List[str], labels: List[str]) -> "MLIntentClassifier":
        valid_pairs = [(t, l) for t, l in zip(texts, labels) if t and l and isinstance(t, str)]
        if not valid_pairs:
            raise ValueError("No valid training samples provided to MLIntentClassifier.")

        clean_texts, clean_labels = zip(*valid_pairs)

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=self.max_features,
                sublinear_tf=True,
                stop_words="english",
                token_pattern=r'(?u)\b\w+\b'
            )),
            ("clf", ComplementNB(alpha=self.alpha, norm=True))
        ])

        self.pipeline.fit(clean_texts, clean_labels)
        self.classes_ = list(self.pipeline.named_steps["clf"].classes_)
        return self

    def predict_one(self, text: str) -> Dict[str, Any]:
        if not text or not self.pipeline:
            return {
                "intent": DEFAULT_INTENT,
                "confidence": 0.0,
                "probabilities": {c: 0.0 for c in self.classes_}
            }

        probs = self.pipeline.predict_proba([text])[0]
        max_idx = int(np.argmax(probs))
        predicted_intent = self.classes_[max_idx]
        confidence = float(probs[max_idx])

        prob_dict = {
            self.classes_[i]: round(float(probs[i]), 4)
            for i in range(len(self.classes_))
        }

        return {
            "intent": predicted_intent,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict
        }
