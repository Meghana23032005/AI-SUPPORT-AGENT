"""Grounded reply generator supporting LLM API inference with deterministic grounded fallback for Apple Support."""
import os
import re
import json
from typing import Dict, Any, List, Optional
import requests
from config.settings import settings
from src.taxonomy.intent_definitions import INTENT_METADATA

class GroundedReplyGenerator:
    """
    Synthesizes brand-aligned, empathetic replies grounded in retrieved historical
    Apple Support resolutions. Uses Gemini/OpenAI if configured; otherwise uses a
    deterministic grounded RAG synthesis engine.
    """
    def __init__(self, system_prompt_path: Optional[str] = None):
        prompt_file = system_prompt_path or (settings.PROMPTS_DIR / "generation_system.txt")
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                self.system_prompt = f.read()
        except Exception:
            self.system_prompt = "You are an expert customer support agent for Apple Support (@AppleSupport)."

    def generate(
        self,
        customer_message: str,
        predicted_intent: str,
        retrieved_examples: List[Dict[str, Any]],
        escalation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generates an Apple Support response.
        """
        decision = escalation_result.get("decision", "AUTO_HANDLE")
        reason = escalation_result.get("reason", "")

        # Try LLM API first if configured
        if settings.GEMINI_API_KEY:
            try:
                llm_response = self._call_gemini_api(
                    customer_message, predicted_intent, retrieved_examples, decision, reason
                )
                if llm_response:
                    return {
                        "reply": llm_response,
                        "generation_method": "LLM_API_GEMINI",
                        "is_grounded": bool(retrieved_examples)
                    }
            except Exception:
                pass  # Fall back to deterministic grounded synthesis

        # Deterministic Grounded Synthesis
        fallback_reply = self._synthesize_grounded_reply(
            customer_message, predicted_intent, retrieved_examples, decision, reason
        )
        return {
            "reply": fallback_reply,
            "generation_method": "DETERMINISTIC_GROUNDED_RAG",
            "is_grounded": bool(retrieved_examples)
        }

    def _synthesize_grounded_reply(
        self,
        customer_message: str,
        predicted_intent: str,
        retrieved_examples: List[Dict[str, Any]],
        decision: str,
        reason: str
    ) -> str:
        """
        Synthesizes a response strictly grounded in top historical resolution.
        """
        intent_info = INTENT_METADATA.get(predicted_intent, {})
        intent_label = intent_info.get("label", "Technical Support")

        if decision == "ESCALATE":
            return (
                f"We're here to help. Regarding your {intent_label} issue: "
                f"Because this requires specialist handling ({reason}), "
                f"we would like to continue this securely. "
                f"Please join us in a Direct Message (DM) here: https://apple.co/dm "
                f"or schedule a Genius Bar reservation at https://getsupport.apple.com. "
                f"Be sure to let us know your exact device model and iOS version when messaging!"
            )

        # Grounded Auto-Handle Reply
        if retrieved_examples:
            top_ex = retrieved_examples[0]
            hist_reply = str(top_ex.get("brand_response") or "").strip()

            # Clean and adapt historical guidance
            clean_hist = re.sub(r'@[A-Za-z0-9_]+', '', hist_reply)
            clean_hist = re.sub(r'\[CUSTOMER\]', '', clean_hist)
            clean_hist = re.sub(r'\[USER\]', '', clean_hist)
            clean_hist = re.sub(r'\^[A-Z]{2,3}', '', clean_hist).strip()

            return (
                f"We're here to help with your {intent_label.lower()}. "
                f"Based on verified Apple Support steps:\n\n"
                f"{clean_hist}\n\n"
                f"Make sure to verify your iOS version in Settings > General > About. "
                f"If the issue persists after these steps, reach back out to us or join us in DM: https://apple.co/dm"
            )

        # Default fallback if no historical match
        return (
            f"We'd like to help look into this with you. Which device model are you using, "
            f"and what iOS version is currently installed under Settings > General > About? "
            f"Please let us know if any troubleshooting steps have been tried so far."
        )

    def _call_gemini_api(
        self,
        customer_message: str,
        predicted_intent: str,
        retrieved_examples: List[Dict[str, Any]],
        decision: str,
        reason: str
    ) -> Optional[str]:
        """Calls Gemini REST API with strict grounding constraints."""
        context_str = "\n\n".join([
            f"- Historical Apple Resolution: {ex.get('brand_response')}"
            for ex in retrieved_examples[:2]
        ])

        user_prompt = (
            f"Customer Message: {customer_message}\n"
            f"Identified Intent: {predicted_intent}\n"
            f"Escalation Policy: {decision} (Reason: {reason})\n\n"
            f"Historical Apple Support Resolutions for Grounding:\n{context_str}\n\n"
            f"Draft the Twitter customer support response following Apple Support rules:"
        )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={settings.GEMINI_API_KEY}"
        payload = {
            "contents": [{
                "parts": [
                    {"text": f"{self.system_prompt}\n\n{user_prompt}"}
                ]
            }],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 200
            }
        }
        res = requests.post(url, json=payload, timeout=10)
        if res.status_code == 200:
            data = res.json()
            candidates = data.get("candidates", [])
            if candidates:
                return candidates[0]["content"]["parts"][0]["text"].strip()
        return None
