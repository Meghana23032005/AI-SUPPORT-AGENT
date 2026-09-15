"""Unit tests for Apple Support intent classifiers (Trivial, ML, and Hybrid)."""
from src.classifiers.trivial_classifier import TrivialIntentClassifier
from src.classifiers.ml_classifier import MLIntentClassifier
from src.classifiers.hybrid_classifier import HybridIntentClassifier
from src.taxonomy.intent_definitions import IntentCategory, ALL_INTENTS

TRAIN_TEXTS = [
    "Battery draining super fast on my iPhone 7",
    "iPhone won't charge with lightning cable and gets hot",
    "After updating to iOS 11 keyboard changes letter I to A",
    "iPhone frozen on Apple logo boot loop after update",
    "Apple ID is locked and I cannot receive 2FA code",
    "Forgot passcode and iCloud account disabled",
    "Charged twice for an App Store subscription refund",
    "Unauthorized purchase on iTunes bill",
    "Cracked screen and touch display is broken",
    "Dropped iPhone glass shattered microphone no sound",
    "Wi-Fi keeps dropping and Bluetooth AirPods won't connect",
    "Cellular says No Service carrier settings update",
    "What are Apple Store Covent Garden opening hours",
    "Hello can I make a Genius Bar appointment"
]

TRAIN_LABELS = [
    IntentCategory.BATTERY_POWER_CHARGING.value,
    IntentCategory.BATTERY_POWER_CHARGING.value,
    IntentCategory.IOS_UPDATE_SYSTEM_CRASH.value,
    IntentCategory.IOS_UPDATE_SYSTEM_CRASH.value,
    IntentCategory.ACCOUNT_ICLOUD_SECURITY.value,
    IntentCategory.ACCOUNT_ICLOUD_SECURITY.value,
    IntentCategory.APPSTORE_BILLING_SUBSCRIPTION.value,
    IntentCategory.APPSTORE_BILLING_SUBSCRIPTION.value,
    IntentCategory.HARDWARE_SCREEN_AUDIO.value,
    IntentCategory.HARDWARE_SCREEN_AUDIO.value,
    IntentCategory.CONNECTIVITY_NETWORK_BLUETOOTH.value,
    IntentCategory.CONNECTIVITY_NETWORK_BLUETOOTH.value,
    IntentCategory.GENERAL_INQUIRY_FEEDBACK.value,
    IntentCategory.GENERAL_INQUIRY_FEEDBACK.value
]

def test_trivial_classifier():
    clf = TrivialIntentClassifier()
    clf.fit(TRAIN_TEXTS, TRAIN_LABELS)
    res = clf.predict_one("My screen is completely broken")
    assert "intent" in res
    assert 0.0 <= res["confidence"] <= 1.0

def test_ml_classifier():
    clf = MLIntentClassifier()
    clf.fit(TRAIN_TEXTS, TRAIN_LABELS)
    res = clf.predict_one("Charged twice for my subscription refund?")
    assert "intent" in res
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["probabilities"]) == len(ALL_INTENTS)

def test_hybrid_classifier():
    clf = HybridIntentClassifier()
    clf.fit(TRAIN_TEXTS, TRAIN_LABELS)
    res = clf.predict_one("iPhone battery is draining super fast and overheating")
    assert res["intent"] == IntentCategory.BATTERY_POWER_CHARGING.value
    assert res["confidence"] > 0.3
    assert "is_ambiguous" in res
