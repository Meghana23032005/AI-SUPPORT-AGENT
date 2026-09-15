"""Unit tests for the Apple Support Escalation Decision Engine."""
from src.generation.escalation import EscalationDecisionEngine
from src.taxonomy.intent_definitions import IntentCategory

def test_escalate_on_legal_threat():
    engine = EscalationDecisionEngine()
    res = engine.evaluate(
        customer_message="I will sue Apple and take this to court with my attorney",
        predicted_intent=IntentCategory.GENERAL_INQUIRY_FEEDBACK.value,
        intent_confidence=0.85,
        retrieved_examples=[{"similarity_score": 0.40}]
    )
    assert res["decision"] == "ESCALATE"
    assert "Legal" in res["reason"]

def test_escalate_on_physical_damage():
    engine = EscalationDecisionEngine()
    res = engine.evaluate(
        customer_message="I dropped my phone and the screen is completely shattered",
        predicted_intent=IntentCategory.HARDWARE_SCREEN_AUDIO.value,
        intent_confidence=0.90,
        retrieved_examples=[{"similarity_score": 0.40}]
    )
    assert res["decision"] == "ESCALATE"
    assert "hardware repair" in res["reason"].lower() or "physical" in res["reason"].lower()

def test_auto_handle_on_standard_battery_faq():
    engine = EscalationDecisionEngine()
    res = engine.evaluate(
        customer_message="My battery drains fast after iOS update how do I check battery health",
        predicted_intent=IntentCategory.BATTERY_POWER_CHARGING.value,
        intent_confidence=0.88,
        retrieved_examples=[{"similarity_score": 0.35}]
    )
    assert res["decision"] == "AUTO_HANDLE"
    assert "BATTERY_POWER_CHARGING" in res["evidence"][0]
