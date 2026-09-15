"""Text cleaning and normalization utilities for Twitter customer support data."""
import re
from typing import Optional

# Regex patterns
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
USER_HANDLE_PATTERN = re.compile(r'@\d{3,8}') # Anonymized customer IDs in twcs like @118196
BRAND_HANDLE_PATTERN = re.compile(r'@(DellCares|Dell|DellSupport)\b', re.IGNORECASE)
OTHER_HANDLE_PATTERN = re.compile(r'@[A-Za-z0-9_]+')
MULTIPART_TWEET_PATTERN = re.compile(r'(\b\d+/\d+\b|\(\d+/\d+\))') # E.g., 1/2, (2/2), 2/5
AGENT_SIGNOFF_PATTERN = re.compile(r'\^[A-Z]{2,3}\b') # Dell agent signatures like ^VS, ^PL, ^SL
WHITESPACE_PATTERN = re.compile(r'\s+')

def clean_tweet_text(
    text: Optional[str],
    normalize_handles: bool = True,
    remove_multipart: bool = True,
    remove_agent_signoffs: bool = True
) -> str:
    """Cleans and standardizes raw tweet text while preserving technical content."""
    if not text or not isinstance(text, str):
        return ""

    cleaned = text

    # Remove URLs or replace with [URL] token
    cleaned = URL_PATTERN.sub('[URL]', cleaned)

    # Normalize agent signatures (^VS, ^PL)
    if remove_agent_signoffs:
        cleaned = AGENT_SIGNOFF_PATTERN.sub('', cleaned)

    # Normalize multi-part markers (1/2, 2/2)
    if remove_multipart:
        cleaned = MULTIPART_TWEET_PATTERN.sub('', cleaned)

    # Normalize handles
    if normalize_handles:
        cleaned = USER_HANDLE_PATTERN.sub('[CUSTOMER]', cleaned)
        cleaned = BRAND_HANDLE_PATTERN.sub('[BRAND]', cleaned)
        cleaned = OTHER_HANDLE_PATTERN.sub('[USER]', cleaned)
        cleaned = cleaned.replace('[BRAND]', '@DellCares')

    # Strip excessive whitespace
    cleaned = WHITESPACE_PATTERN.sub(' ', cleaned).strip()

    return cleaned

def extract_agent_signoff(text: str) -> Optional[str]:
    """Extracts agent signoff code (e.g. ^VS) if present."""
    match = AGENT_SIGNOFF_PATTERN.search(text)
    return match.group(0) if match else None
