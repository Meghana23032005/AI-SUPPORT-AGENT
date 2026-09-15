"""LLM-as-a-Judge Evaluation Harness implementing 5-dimension enterprise Apple Support rubric."""
import json
import re
from typing import Dict, Any, List, Optional
import requests
from config.settings import settings

class LLMJudgeEvaluator:
    """
    Evaluates customer support replies across 5 core dimensions:
    1. Intent Appropriateness (1-5)
    2. Groundedness & Faithfulness (1-5)
    3. Technical Actionability (1-5)
    4. Brand Tone & Privacy Compliance (1-5)
    5. Escalation Appropriateness (1-5)
    """
    def __init__(self, rubric_path: Optional[str] = None):
        path = rubric_path or (settings.PROMPTS_DIR / "llm_judge_rubric.txt")
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.rubric = f.read()
        except Exception:
            self.rubric = "Evaluate Apple Support reply quality on scale 1-5."

    def judge_example(
        self,
        customer_message: str,
        predicted_intent: str,
        gold_intent: str,
        generated_reply: str,
        decision: str,
        gold_decision: str,
        escalation_reason: str,
        retrieved_context: str
    ) -> Dict[str, Any]:
        """Judges a single interaction example."""
        if settings.GEMINI_API_KEY:
            try:
                llm_eval = self._call_gemini_judge(
                    customer_message, predicted_intent, gold_intent,
                    generated_reply, decision, gold_decision,
                    escalation_reason, retrieved_context
                )
                if llm_eval:
                    return llm_eval
            except Exception:
                pass  # Fall back to deterministic rubric judge

        return self._deterministic_rubric_judge(
            predicted_intent, gold_intent,
            generated_reply, decision, gold_decision,
            escalation_reason, retrieved_context
        )

    def _deterministic_rubric_judge(
        self,
        predicted_intent: str,
        gold_intent: str,
        generated_reply: str,
        decision: str,
        gold_decision: str,
        escalation_reason: str,
        retrieved_context: str
    ) -> Dict[str, Any]:
        """
        Objective rubric scoring engine based on established Apple Support criteria.
        Guarantees deterministic, reproducible scores across offline and CI runs.
        """
        # 1. Intent Score
        if predicted_intent == gold_intent:
            intent_score = 5.0
        elif predicted_intent in ["GENERAL_INQUIRY_FEEDBACK", "IOS_UPDATE_SYSTEM_CRASH"]:
            intent_score = 3.5  # Acceptable broader category
        else:
            intent_score = 2.0

        # 2. Groundedness Score
        reply_lower = generated_reply.lower()
        context_tokens = set(retrieved_context.lower().split())
        reply_tokens = set(reply_lower.split())
        norm_len = min(50, len(context_tokens)) if context_tokens else 1
        overlap = len(reply_tokens.intersection(context_tokens)) / norm_len if context_tokens else 0.0

        if overlap > 0.20 or "apple.co" in reply_lower or "settings > general" in reply_lower:
            groundedness_score = 5.0
        elif overlap > 0.08:
            groundedness_score = 4.0
        elif "dm" in reply_lower:
            groundedness_score = 3.5
        else:
            groundedness_score = 2.5

        # 3. Actionability Score
        action_indicators = ["settings", "restart", "force restart", "battery", "reset", "network", "genius bar", "dm", "direct message", "check", "verify", "link"]
        actions_found = sum(1 for a in action_indicators if a in reply_lower)
        if actions_found >= 3:
            actionability_score = 5.0
        elif actions_found >= 1:
            actionability_score = 4.0
        else:
            actionability_score = 2.5

        # 4. Brand Tone & Privacy Compliance Score
        privacy_pass = not any(p in reply_lower for p in ["post your password", "tweet your passcode", "tweet serial number", "post apple id password"])
        has_polite_greeting = any(g in reply_lower for g in ["we're here to help", "we'd like to help", "hi", "hello", "thanks for reaching out"])
        has_dm_guidance = "dm" in reply_lower or "direct message" in reply_lower or "getsupport.apple.com" in reply_lower

        if privacy_pass and has_polite_greeting and has_dm_guidance:
            tone_score = 5.0
        elif privacy_pass and (has_polite_greeting or has_dm_guidance):
            tone_score = 4.0
        else:
            tone_score = 2.0

        # 5. Escalation Appropriateness Score
        if decision == gold_decision:
            escalation_score = 5.0
        elif gold_decision == "ESCALATE" and decision == "AUTO_HANDLE":
            # Dangerous under-escalation
            escalation_score = 1.5
        else:
            # Over-cautious escalation
            escalation_score = 3.5

        # Overall weighted composite
        overall_score = round(
            0.25 * intent_score +
            0.25 * groundedness_score +
            0.20 * actionability_score +
            0.15 * tone_score +
            0.15 * escalation_score,
            2
        )

        verdict = "PASS" if (overall_score >= 3.8 and min(intent_score, groundedness_score, escalation_score) > 1.0) else "FAIL"

        return {
            "intent_score": intent_score,
            "groundedness_score": groundedness_score,
            "actionability_score": actionability_score,
            "tone_score": tone_score,
            "escalation_score": escalation_score,
            "overall_score": overall_score,
            "verdict": verdict,
            "critique": f"Intent matched ({intent_score}/5), Grounding ({groundedness_score}/5), Escalation routing: {decision} vs Gold: {gold_decision} ({escalation_score}/5)."
        }

    def _call_gemini_judge(
        self,
        customer_message: str,
        predicted_intent: str,
        gold_intent: str,
        generated_reply: str,
        decision: str,
        gold_decision: str,
        escalation_reason: str,
        retrieved_context: str
    ) -> Optional[Dict[str, Any]]:
        """Calls Gemini API for LLM Judge scoring with structured JSON response."""
        prompt = (
            f"{self.rubric}\n\n"
            f"EVALUATION TARGET:\n"
            f"- Customer Message: {customer_message}\n"
            f"- Predicted Intent: {predicted_intent} (Ground Truth: {gold_intent})\n"
            f"- Generated Reply: {generated_reply}\n"
            f"- Escalation Decision: {decision} (Ground Truth: {gold_decision}, Reason: {escalation_reason})\n"
            f"- Grounding Context:\n{retrieved_context}\n\n"
            f"Output strictly valid JSON matching the requested schema."
        )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={settings.GEMINI_API_KEY}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json"
            }
        }
        res = requests.post(url, json=payload, timeout=10)
        if res.status_code == 200:
            data = res.json()
            candidates = data.get("candidates", [])
            if candidates:
                raw_json = candidates[0]["content"]["parts"][0]["text"]
                return json.loads(raw_json)
        return None
