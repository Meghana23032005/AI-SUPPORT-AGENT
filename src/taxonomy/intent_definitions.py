"""Intent taxonomy definitions and metadata for Apple customer support interactions."""
from enum import Enum
from typing import Dict, List, Any

class IntentCategory(str, Enum):
    BATTERY_POWER_CHARGING = "BATTERY_POWER_CHARGING"
    IOS_UPDATE_SYSTEM_CRASH = "IOS_UPDATE_SYSTEM_CRASH"
    ACCOUNT_ICLOUD_SECURITY = "ACCOUNT_ICLOUD_SECURITY"
    APPSTORE_BILLING_SUBSCRIPTION = "APPSTORE_BILLING_SUBSCRIPTION"
    HARDWARE_SCREEN_AUDIO = "HARDWARE_SCREEN_AUDIO"
    CONNECTIVITY_NETWORK_BLUETOOTH = "CONNECTIVITY_NETWORK_BLUETOOTH"
    GENERAL_INQUIRY_FEEDBACK = "GENERAL_INQUIRY_FEEDBACK"

INTENT_METADATA: Dict[str, Dict[str, Any]] = {
    IntentCategory.BATTERY_POWER_CHARGING.value: {
        "label": "Battery, Power & Charging",
        "description": "Battery health degradation, rapid battery drain, device overheating, charging cable faults, or sudden power shutoffs.",
        "keywords": [
            "battery", "charging", "charge", "drain", "draining", "dies", "charger",
            "cable", "overheat", "overheating", "percent", "battery health", "power off",
            "won't charge", "lightning cable"
        ],
        "default_escalation": False,
        "auto_handle_criteria": "Customer describes battery drain or charging issue that can be diagnosed via Settings > Battery and background app refresh guidance."
    },
    IntentCategory.IOS_UPDATE_SYSTEM_CRASH.value: {
        "label": "iOS Update, Bugs & System Crashes",
        "description": "Operating system updates (e.g. iOS 11), autocorrect/keyboard glitches, frozen UI, boot loops, stuck on Apple logo, app crashes.",
        "keywords": [
            "ios", "update", "updated", "updating", "frozen", "freeze", "freezing",
            "lag", "lagging", "glitch", "boot loop", "apple logo", "keyboard", "autocorrect",
            "crash", "crashes", "crashing", "stuck", "black screen", "reboot"
        ],
        "default_escalation": False,
        "auto_handle_criteria": "Known software bugs, iOS update installation checks (Settings > General > About), or force restart instructions."
    },
    IntentCategory.ACCOUNT_ICLOUD_SECURITY.value: {
        "label": "Apple ID, iCloud & Account Security",
        "description": "Apple ID lockouts, 2-factor authentication codes, forgotten passcode, Activation Lock, iCloud storage full, backup and sync failures.",
        "keywords": [
            "apple id", "icloud", "passcode", "password", "locked", "2fa",
            "verification code", "storage", "backup", "restore backup", "account",
            "disabled", "unlock", "activation lock", "forgot password"
        ],
        "default_escalation": True,
        "auto_handle_criteria": "Requires human specialist or secure iforgot.apple.com authentication protocol due to privacy/identity security rules."
    },
    IntentCategory.APPSTORE_BILLING_SUBSCRIPTION.value: {
        "label": "App Store, Subscriptions & Billing",
        "description": "Unrecognized credit card charges, recurring subscription cancellations, refund requests, iTunes store balance, declined payment methods.",
        "keywords": [
            "app store", "subscription", "charged", "billing", "refund", "receipt",
            "purchase", "purchased", "payment", "cancel subscription", "credit card",
            "itunes", "in-app purchase", "unauthorized charge", "money"
        ],
        "default_escalation": True,
        "auto_handle_criteria": "Financial disputes, unauthorized charges, or refund requests requiring account lookup."
    },
    IntentCategory.HARDWARE_SCREEN_AUDIO.value: {
        "label": "Hardware, Screen & Audio Damage",
        "description": "Physical damage, cracked/shattered screen, unresponsive touch display, microphone/speaker failures, camera issues, liquid/water damage.",
        "keywords": [
            "screen", "cracked", "broken", "shattered", "display", "touch", "speaker",
            "mic", "microphone", "sound", "camera", "volume button", "home button",
            "water damage", "earpiece", "physical damage", "repair", "genius bar"
        ],
        "default_escalation": True,
        "auto_handle_criteria": "Physical hardware repair or component replacement requiring Genius Bar appointment or mail-in repair."
    },
    IntentCategory.CONNECTIVITY_NETWORK_BLUETOOTH.value: {
        "label": "Connectivity, Wi-Fi & Bluetooth",
        "description": "Wi-Fi disconnections, Bluetooth pairing issues (AirPods, car), cellular 'No Service' or search, AirDrop failures, carrier settings.",
        "keywords": [
            "wifi", "wi-fi", "bluetooth", "airpods", "airdrop", "no service",
            "cellular", "lte", "signal", "connect", "connection", "hotspot",
            "pairing", "pair", "disconnecting"
        ],
        "default_escalation": False,
        "auto_handle_criteria": "Standard network reset (Settings > General > Reset > Reset Network Settings) or carrier update checks."
    },
    IntentCategory.GENERAL_INQUIRY_FEEDBACK.value: {
        "label": "General Inquiry, Appointments & Feedback",
        "description": "Store hours, Genius Bar booking assistance, trade-in value questions, compliments, or general customer feedback.",
        "keywords": [
            "help", "question", "recommend", "purchase", "buy", "feedback", "worst",
            "complaint", "hello", "hi", "anyone there", "thanks", "thank you",
            "genius bar", "appointment", "trade in", "store"
        ],
        "default_escalation": False,
        "auto_handle_criteria": "General guidance, appointment booking link, or polite customer service clarification."
    }
}

ALL_INTENTS: List[str] = [i.value for i in IntentCategory]
DEFAULT_INTENT: str = IntentCategory.GENERAL_INQUIRY_FEEDBACK.value
