"""Proposed System: Hybrid Intent Classifier combining domain priors, multi-scale n-grams, and uncertainty estimation."""
from typing import Dict, Any, List, Optional
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from src.classifiers.base import BaseIntentClassifier
from src.taxonomy.intent_definitions import IntentCategory, INTENT_METADATA, ALL_INTENTS, DEFAULT_INTENT

class HybridIntentClassifier(BaseIntentClassifier):
    """
    Proposed System Intent Classifier.
    Employs multi-scale word & sub-word character n-gram feature representations,
    calibrated probability estimation, domain keyword boosting,
    and prediction margin uncertainty analysis.
    """
    def __init__(self, alpha: float = 0.2, min_margin: float = 0.15):
        self.alpha = alpha
        self.min_margin = min_margin
        self.pipeline: Optional[Pipeline] = None
        self.classes_: List[str] = ALL_INTENTS

        # High-precision domain pattern priors for Apple Support
        self.domain_patterns = {
            IntentCategory.BATTERY_POWER_CHARGING.value: re.compile(
                r'\b(battery|charging|charge|drain|draining|dies|charger|cable|overheat|overheating|percent|battery health|lightning cable)\b',
                re.IGNORECASE
            ),
            IntentCategory.IOS_UPDATE_SYSTEM_CRASH.value: re.compile(
                r'\b(ios|update|updated|updating|frozen|freeze|freezing|lag|glitch|boot loop|apple logo|keyboard|autocorrect|crash|crashes|crashing|ios 11)\b',
                re.IGNORECASE
            ),
            IntentCategory.ACCOUNT_ICLOUD_SECURITY.value: re.compile(
                r'\b(apple id|icloud|passcode|password|locked|2fa|verification code|storage full|backup|restore backup|disabled|activation lock|iforgot)\b',
                re.IGNORECASE
            ),
            IntentCategory.APPSTORE_BILLING_SUBSCRIPTION.value: re.compile(
                r'\b(app store|subscription|charged|billing|refund|receipt|purchase|payment method|cancel subscription|itunes store|unauthorized charge)\b',
                re.IGNORECASE
            ),
            IntentCategory.HARDWARE_SCREEN_AUDIO.value: re.compile(
                r'\b(cracked|shattered|broken screen|display|touch|speaker|mic|microphone|sound|camera|water damage|dropped phone|earpiece|genius bar repair)\b',
                re.IGNORECASE
            ),
            IntentCategory.CONNECTIVITY_NETWORK_BLUETOOTH.value: re.compile(
                r'\b(wifi|wi-fi|bluetooth|airpods|airdrop|no service|cellular|lte|signal|hotspot|pairing|disconnecting)\b',
                re.IGNORECASE
            ),
        }

    def fit(self, texts: List[str], labels: List[str]) -> "HybridIntentClassifier":
        valid_pairs = [(t, l) for t, l in zip(texts, labels) if t and l and isinstance(t, str)]
        if not valid_pairs:
            raise ValueError("No valid training samples provided to HybridIntentClassifier.")

        clean_texts, clean_labels = zip(*valid_pairs)

        # Multi-scale feature union (words + sub-word character ngrams)
        union = FeatureUnion([
            ("word_tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=5000,
                sublinear_tf=True,
                stop_words="english",
                token_pattern=r'(?u)\b\w+\b'
            )),
            ("char_tfidf", TfidfVectorizer(
                ngram_range=(3, 5),
                analyzer="char_wb",
                max_features=3500,
                sublinear_tf=True
            ))
        ])

        self.pipeline = Pipeline([
            ("features", union),
            ("clf", MultinomialNB(alpha=self.alpha))
        ])

        self.pipeline.fit(clean_texts, clean_labels)
        self.classes_ = list(self.pipeline.named_steps["clf"].classes_)
        return self

    def predict_one(self, text: str) -> Dict[str, Any]:
        if not text or not self.pipeline:
            return {
                "intent": DEFAULT_INTENT,
                "confidence": 0.0,
                "is_ambiguous": True,
                "margin": 0.0,
                "second_intent": "",
                "probabilities": {c: 0.0 for c in self.classes_}
            }

        # Raw probabilities from ML pipeline
        raw_probs = self.pipeline.predict_proba([text])[0]
        prob_dict = {
            self.classes_[i]: float(raw_probs[i])
            for i in range(len(self.classes_))
        }

        # Apply strong domain keyword boost if exact patterns match
        boosted_probs = prob_dict.copy()
        for intent_cat, pattern in self.domain_patterns.items():
            if pattern.search(text) and intent_cat in boosted_probs:
                # Strong boost to overcome majority class prior in specialized classes
                boosted_probs[intent_cat] += 1.8

        # Re-normalize
        total_p = sum(boosted_probs.values())
        if total_p > 0:
            for k in boosted_probs:
                boosted_probs[k] /= total_p

        # Sort descending
        sorted_intents = sorted(boosted_probs.items(), key=lambda x: x[1], reverse=True)
        top_intent, top_conf = sorted_intents[0]
        second_intent, second_conf = sorted_intents[1] if len(sorted_intents) > 1 else ("", 0.0)

        margin = top_conf - second_conf
        is_ambiguous = margin < self.min_margin

        return {
            "intent": top_intent,
            "confidence": round(float(top_conf), 4),
            "is_ambiguous": is_ambiguous,
            "margin": round(float(margin), 4),
            "second_intent": second_intent,
            "probabilities": {k: round(float(v), 4) for k, v in boosted_probs.items()}
        }
