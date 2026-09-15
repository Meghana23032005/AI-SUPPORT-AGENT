"""Step 2: Reconstruct Apple conversation threads, label intents, and generate leakage-free splits."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.settings import settings
from src.preprocessing.thread_reconstructor import reconstruct_conversations
from src.taxonomy.intent_definitions import IntentCategory, INTENT_METADATA

def assign_intent_from_text(customer_text: str, reply_text: str) -> str:
    """Assigns grounded intent category based on customer issue and Apple resolution."""
    combined = f"{customer_text} {reply_text}".lower()

    # Rule precedence based on specificity:
    # 1. Hardware, Screen & Audio Physical Damage
    hw_kw = [
        "cracked", "shattered", "broken screen", "display", "unresponsive touch",
        "touch screen", "speaker", "mic", "microphone", "sound not working",
        "camera black", "camera blurry", "volume button", "home button",
        "water damage", "dropped phone", "earpiece", "genius bar repair", "physical damage"
    ]
    if any(k in combined for k in hw_kw):
        return IntentCategory.HARDWARE_SCREEN_AUDIO.value

    # 2. App Store, Billing & Subscriptions
    billing_kw = [
        "app store", "subscription", "charged", "billing", "refund", "receipt",
        "unauthorized charge", "purchase", "purchased", "payment method",
        "cancel subscription", "itunes store", "declined", "chargeback", "double charged"
    ]
    if any(k in combined for k in billing_kw):
        return IntentCategory.APPSTORE_BILLING_SUBSCRIPTION.value

    # 3. Account, Apple ID & iCloud Security
    account_kw = [
        "apple id", "icloud", "passcode", "password", "locked", "2fa",
        "two-factor", "verification code", "storage full", "icloud storage",
        "backup", "restore backup", "disabled", "unlock account", "activation lock",
        "forgot passcode", "iforgot"
    ]
    if any(k in combined for k in account_kw):
        return IntentCategory.ACCOUNT_ICLOUD_SECURITY.value

    # 4. Battery, Power & Charging
    battery_kw = [
        "battery", "charging", "charge", "drain", "draining", "dies", "charger",
        "cable", "overheat", "overheating", "percent", "battery health",
        "power off", "won't charge", "lightning cable", "battery life"
    ]
    if any(k in combined for k in battery_kw):
        return IntentCategory.BATTERY_POWER_CHARGING.value

    # 5. Connectivity, Wi-Fi & Bluetooth
    conn_kw = [
        "wifi", "wi-fi", "bluetooth", "airpods", "airdrop", "no service",
        "cellular", "lte", "signal", "hotspot", "pairing", "disconnecting", "carrier settings"
    ]
    if any(k in combined for k in conn_kw):
        return IntentCategory.CONNECTIVITY_NETWORK_BLUETOOTH.value

    # 6. iOS Update, Bugs & System Crashes
    os_kw = [
        "ios", "update", "updated", "updating", "frozen", "freeze", "freezing",
        "lag", "lagging", "glitch", "boot loop", "apple logo", "keyboard",
        "autocorrect", "crash", "crashes", "crashing", "stuck on", "black screen",
        "ios 11", "letter i", "capital i"
    ]
    if any(k in combined for k in os_kw):
        return IntentCategory.IOS_UPDATE_SYSTEM_CRASH.value

    # Default
    return IntentCategory.GENERAL_INQUIRY_FEEDBACK.value

def process_and_split():
    print(f"Loading raw Apple tweets from {settings.RAW_APPLE_CSV}...")
    if not settings.RAW_APPLE_CSV.exists():
        raise FileNotFoundError(f"Raw data missing at {settings.RAW_APPLE_CSV}. Run 01_extract_apple_data.py first.")

    df_raw = pd.read_csv(settings.RAW_APPLE_CSV)
    
    print("Reconstructing multi-turn conversation threads...")
    df_threads = reconstruct_conversations(df_raw, brand_tag="apple")
    print(f"Reconstructed {len(df_threads):,} conversation threads.")

    # Filter threads that have meaningful customer inquiries
    df_threads = df_threads[df_threads['customer_message_clean'].str.strip().str.len() > 5].copy()

    # Assign intents
    print("Assigning empirical intent labels...")
    df_threads['intent'] = df_threads.apply(
        lambda r: assign_intent_from_text(str(r['customer_message_clean']), str(r['brand_response_clean'])),
        axis=1
    )

    settings.PROCESSED_THREADS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_threads.to_csv(settings.PROCESSED_THREADS_CSV, index=False)
    print(f"Saved processed conversation threads to {settings.PROCESSED_THREADS_CSV}")

    # Display intent breakdown
    print("\nIntent Distribution:")
    print(df_threads['intent'].value_counts())

    print("\nConversation Statistics:")
    print(f"Total Conversations: {len(df_threads):,}")
    print(f"Conversations with Brand Response: {df_threads['has_brand_response'].sum():,} ({df_threads['has_brand_response'].mean():.1%})")
    print(f"Multi-turn Conversations: {df_threads['has_multiple_turns'].sum():,} ({df_threads['has_multiple_turns'].mean():.1%})")

    # Conversation-level train/val/test splitting (PREVENTING DATA LEAKAGE)
    print("\nSplitting conversations at CONVERSATION LEVEL (Zero leakage)...")
    np.random.seed(settings.RANDOM_SEED)
    
    # Stratified shuffle by intent
    df_shuffled = df_threads.sample(frac=1.0, random_state=settings.RANDOM_SEED).reset_index(drop=True)
    
    train_dfs, val_dfs, test_dfs = [], [], []
    for intent, group in df_shuffled.groupby('intent'):
        n = len(group)
        n_train = int(n * settings.TRAIN_RATIO)
        n_val = int(n * settings.VAL_RATIO)
        
        train_dfs.append(group.iloc[:n_train])
        val_dfs.append(group.iloc[n_train:n_train + n_val])
        test_dfs.append(group.iloc[n_train + n_val:])

    train_df = pd.concat(train_dfs).sample(frac=1.0, random_state=settings.RANDOM_SEED).reset_index(drop=True)
    val_df = pd.concat(val_dfs).sample(frac=1.0, random_state=settings.RANDOM_SEED).reset_index(drop=True)
    test_df = pd.concat(test_dfs).sample(frac=1.0, random_state=settings.RANDOM_SEED).reset_index(drop=True)

    # Verification: Verify zero conversation ID overlap
    train_ids = set(train_df['conversation_id'])
    val_ids = set(val_df['conversation_id'])
    test_ids = set(test_df['conversation_id'])

    assert len(train_ids.intersection(val_ids)) == 0, "DATA LEAKAGE DETECTED: Train-Val overlap!"
    assert len(train_ids.intersection(test_ids)) == 0, "DATA LEAKAGE DETECTED: Train-Test overlap!"
    assert len(val_ids.intersection(test_ids)) == 0, "DATA LEAKAGE DETECTED: Val-Test overlap!"

    settings.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(settings.TRAIN_SPLIT_CSV, index=False)
    val_df.to_csv(settings.VAL_SPLIT_CSV, index=False)
    test_df.to_csv(settings.TEST_SPLIT_CSV, index=False)

    print("\nData Splits Generated Successfully (Zero Conversation Leakage):")
    print(f"  Train: {len(train_df):,} conversations ({len(train_df)/len(df_threads):.1%})")
    print(f"  Val:   {len(val_df):,} conversations ({len(val_df)/len(df_threads):.1%})")
    print(f"  Test:  {len(test_df):,} conversations ({len(test_df)/len(df_threads):.1%})")

if __name__ == "__main__":
    process_and_split()
