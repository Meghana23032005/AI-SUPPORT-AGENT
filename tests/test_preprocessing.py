"""Unit tests for text preprocessing and conversation reconstruction."""
import pandas as pd
from src.preprocessing.cleaner import clean_tweet_text, extract_agent_signoff
from src.preprocessing.thread_reconstructor import reconstruct_conversations

def test_clean_tweet_text():
    raw = "@118196 @DellCares My laptop won't turn on! https://t.co/xyz123 (1/2) ^VS"
    clean = clean_tweet_text(raw)
    assert "[URL]" in clean
    assert "^VS" not in clean
    assert "(1/2)" not in clean
    assert "[CUSTOMER]" in clean
    assert "@DellCares" in clean

def test_extract_agent_signoff():
    text = "Please try discharging the battery. ^PL"
    signoff = extract_agent_signoff(text)
    assert signoff == "^PL"

    text_none = "Please try restarting."
    assert extract_agent_signoff(text_none) is None

def test_reconstruct_conversations():
    # Synthetic multi-turn thread
    df_sample = pd.DataFrame([
        {
            "tweet_id": 1,
            "author_id": "cust123",
            "inbound": True,
            "created_at": "Tue Oct 31 10:00:00 +0000 2017",
            "text": "@DellCares battery plugged in not charging",
            "in_response_to_tweet_id": None,
            "response_tweet_id": "2"
        },
        {
            "tweet_id": 2,
            "author_id": "DellCares",
            "inbound": False,
            "created_at": "Tue Oct 31 10:05:00 +0000 2017",
            "text": "@cust123 Please remove battery and hold power for 15s. ^VS",
            "in_response_to_tweet_id": 1,
            "response_tweet_id": "3"
        },
        {
            "tweet_id": 3,
            "author_id": "cust123",
            "inbound": True,
            "created_at": "Tue Oct 31 10:15:00 +0000 2017",
            "text": "@DellCares That worked, thank you!",
            "in_response_to_tweet_id": 2,
            "response_tweet_id": None
        }
    ])

    convs = reconstruct_conversations(df_sample)
    assert len(convs) == 1
    row = convs.iloc[0]
    assert row["customer_id"] == "cust123"
    assert bool(row["has_brand_response"]) is True
    assert row["turn_count"] == 3
    assert bool(row["has_multiple_turns"]) is True
