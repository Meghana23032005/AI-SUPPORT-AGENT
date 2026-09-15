"""Escalation Policy Engine: Decides AUTO_HANDLE vs ESCALATE with auditable reasoning for Apple Support."""
import re
from typing import Dict, Any, List
from src.taxonomy.intent_definitions import IntentCategory

class EscalationDecisionEngine:
    """
    Determines whether an incoming Apple customer inquiry can be safely AUTO_HANDLED
    by the AI agent or must be ESCALATED to a human specialist tier.
    """
    def __init__(self, confidence_threshold: float = 0.60, min_similarity_threshold: float = 0.18):
        self.confidence_threshold = confidence_threshold
        self.min_similarity_threshold = min_similarity_threshold

        # High-urgency escalation patterns (legal, fraud, severe churn, physical hazards)
        self.urgency_patterns = [
            (re.compile(r'\b(lawyer|attorney|sue|legal action|court|consumer court|class action)\b', re.IGNORECASE), "Legal action threat detected"),
            (re.compile(r'\b(chargeback|dispute charge|fraud|unauthorized charge|stolen|scam)\b', re.IGNORECASE), "Financial dispute or unauthorized charge claim"),
            (re.compile(r'\b(manager|supervisor|human agent|talk to someone|human representative|live person)\b', re.IGNORECASE), "Explicit request for human manager or live representative"),
            (re.compile(r'\b(furious|horrible|garbage|disaster|unacceptable|worst customer service|utterly useless)\b', re.IGNORECASE), "High customer frustration and severe negative sentiment"),
            (re.compile(r'\b(smoke|spark|fire|explosion|swollen battery|bulging|expanded battery)\b', re.IGNORECASE), "Critical safety hazard (thermal/battery expansion) requiring immediate containment protocol"),
        ]

    def evaluate(
        self,
        customer_message: str,
        predicted_intent: str,
        intent_confidence: float,
        is_ambiguous: bool = False,
        retrieved_examples: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates message against Apple Support policy rules and returns decision, reason, and evidence.
        """
        retrieved_examples = retrieved_examples or []
        evidence: List[str] = []

        # 1. Check Urgency / Safety / Legal Triggers
        for pattern, reason_text in self.urgency_patterns:
            match = pattern.search(customer_message)
            if match:
                evidence.append(f"Keyword match: '{match.group(0)}'")
                return {
                    "decision": "ESCALATE",
                    "reason": f"{reason_text} ('{match.group(0)}'). Immediate human specialist required.",
                    "confidence": 0.95,
                    "evidence": evidence,
                    "trigger_type": "URGENCY_SENTIMENT"
                }

        # 2. Check Intent-Specific Policy Triggers
        # A. Physical Hardware Repair
        if predicted_intent == IntentCategory.HARDWARE_SCREEN_AUDIO.value:
            evidence.append("Intent: HARDWARE_SCREEN_AUDIO")
            return {
                "decision": "ESCALATE",
                "reason": "Physical hardware repair or component damage requires Genius Bar appointment reservation or mail-in repair depot protocol.",
                "confidence": 0.92,
                "evidence": evidence,
                "trigger_type": "HARDWARE_REPAIR_POLICY"
            }

        # B. Account Security / Apple ID / Passcode
        if predicted_intent == IntentCategory.ACCOUNT_ICLOUD_SECURITY.value:
            # Check if it's a critical lockout/2FA issue
            if any(k in customer_message.lower() for k in ["locked", "passcode", "password", "disabled", "activation lock", "2fa", "two-factor", "verification code"]):
                evidence.append("Intent: ACCOUNT_ICLOUD_SECURITY (Account Lockout/2FA)")
                return {
                    "decision": "ESCALATE",
                    "reason": "Apple ID security lockouts and two-factor authentication recovery require secure authentication protocol or Apple Security Team review.",
                    "confidence": 0.90,
                    "evidence": evidence,
                    "trigger_type": "ACCOUNT_SECURITY_POLICY"
                }

        # C. Financial Billing / Refund Disputes
        if predicted_intent == IntentCategory.APPSTORE_BILLING_SUBSCRIPTION.value:
            if any(k in customer_message.lower() for k in ["refund", "charged", "billing", "unauthorized", "subscription", "double charge", "money"]):
                evidence.append("Intent: APPSTORE_BILLING_SUBSCRIPTION")
                return {
                    "decision": "ESCALATE",
                    "reason": "App Store refund requests and subscription billing inquiries require account purchase verification by a billing specialist.",
                    "confidence": 0.88,
                    "evidence": evidence,
                    "trigger_type": "FINANCIAL_BILLING_POLICY"
                }

        # 3. Check Classification Ambiguity & Low Confidence
        if intent_confidence < self.confidence_threshold or is_ambiguous:
            evidence.append(f"Intent Confidence: {intent_confidence:.2f} (Threshold: {self.confidence_threshold:.2f})")
            if is_ambiguous:
                evidence.append("Ambiguous classification: Top two intents have narrow confidence margin.")
            return {
                "decision": "ESCALATE",
                "reason": f"Underlying customer issue is ambiguous or classification confidence ({intent_confidence:.2f}) is below safe automated threshold.",
                "confidence": round(1.0 - intent_confidence, 2),
                "evidence": evidence,
                "trigger_type": "LOW_CONFIDENCE_AMBIGUITY"
            }

        # 4. Check Historical Retrieval Grounding Quality
        if not retrieved_examples:
            evidence.append("Retrieval: 0 historical examples matched")
            return {
                "decision": "ESCALATE",
                "reason": "No historically similar verified Apple Support resolution was found in knowledge base.",
                "confidence": 0.85,
                "evidence": evidence,
                "trigger_type": "MISSING_GROUNDING"
            }

        best_sim = retrieved_examples[0].get("similarity_score", 0.0)
        if best_sim < self.min_similarity_threshold:
            evidence.append(f"Top Retrieval Similarity: {best_sim:.2f} (Minimum: {self.min_similarity_threshold:.2f})")
            return {
                "decision": "ESCALATE",
                "reason": f"Historical knowledge match relevance ({best_sim:.2f}) is below safe threshold to reliably ground an automated reply.",
                "confidence": 0.80,
                "evidence": evidence,
                "trigger_type": "WEAK_GROUNDING"
            }

        # 5. Passed all escalation hurdles -> Safe to AUTO_HANDLE
        evidence.append(f"Intent '{predicted_intent}' verified with {intent_confidence:.2f} confidence")
        evidence.append(f"Grounded by historical Apple Support resolution (Relevance: {best_sim:.2f})")

        return {
            "decision": "AUTO_HANDLE",
            "reason": "Issue matches established Apple self-serve troubleshooting procedures with high confidence and verified grounding.",
            "confidence": round(intent_confidence, 2),
            "evidence": evidence,
            "trigger_type": "STANDARD_AUTO_HANDLE"
        }
