"""Reconstructs multi-turn conversation threads from raw Twitter support interactions."""
from typing import Dict, List, Any, Optional, Set
import pandas as pd
from datetime import datetime
from src.preprocessing.cleaner import clean_tweet_text

def reconstruct_conversations(df_tweets: pd.DataFrame, brand_tag: str = "apple") -> pd.DataFrame:
    """
    Reconstructs conversation threads from a DataFrame of tweets.
    Each row in the output represents a unique conversation starting from a customer inquiry.
    Linear time O(N) graph reconstruction.
    """
    df = df_tweets.copy()
    df['tweet_id'] = df['tweet_id'].astype(str)
    
    # Safe handling of in_response_to_tweet_id
    df['in_response_to_tweet_id'] = df['in_response_to_tweet_id'].fillna('').astype(str)
    df['in_response_to_tweet_id'] = df['in_response_to_tweet_id'].apply(
        lambda x: x.split('.')[0] if '.' in x else x
    )

    records = df.to_dict('records')
    tweet_map: Dict[str, Dict[str, Any]] = {}
    parent_to_children: Dict[str, List[str]] = {}
    
    # Pre-clean texts once upfront
    for row in records:
        tid = str(row['tweet_id'])
        raw_text = str(row.get('text', ''))
        cleaned_text = clean_tweet_text(raw_text)
        
        tweet_map[tid] = {
            'tweet_id': tid,
            'author_id': str(row['author_id']),
            'inbound': bool(row['inbound']),
            'created_at': str(row.get('created_at', '')),
            'text_raw': raw_text,
            'text_clean': cleaned_text,
            'in_response_to_tweet_id': str(row.get('in_response_to_tweet_id', '')),
            'response_tweet_id': str(row.get('response_tweet_id', ''))
        }
        parent_id = str(row.get('in_response_to_tweet_id', ''))
        if parent_id and parent_id != '' and parent_id != 'nan':
            parent_to_children.setdefault(parent_id, []).append(tid)

    # Find root customer tweets
    root_tids: List[str] = [
        tid for tid, data in tweet_map.items()
        if data['inbound'] and (not data['in_response_to_tweet_id'] or data['in_response_to_tweet_id'] not in tweet_map)
    ]

    conversations: List[Dict[str, Any]] = []
    globally_visited: Set[str] = set()

    for root_tid in root_tids:
        if root_tid in globally_visited:
            continue

        # BFS traversal
        thread_tids = [root_tid]
        queue = [root_tid]
        globally_visited.add(root_tid)

        while queue:
            curr = queue.pop(0)
            children = parent_to_children.get(curr, [])
            for child in children:
                if child in tweet_map and child not in globally_visited:
                    globally_visited.add(child)
                    queue.append(child)
                    thread_tids.append(child)

        thread_records = [tweet_map[t] for t in thread_tids]

        inbound_tweets = [r for r in thread_records if r['inbound']]
        outbound_tweets = [r for r in thread_records if not r['inbound']]

        if not inbound_tweets:
            continue

        first_inbound = inbound_tweets[0]
        customer_id = first_inbound['author_id']
        first_inbound_raw = first_inbound['text_raw']
        first_inbound_clean = first_inbound['text_clean']

        # Collect brand response
        brand_reply_raw = ""
        brand_reply_clean = ""
        if outbound_tweets:
            brand_reply_raw = " \n".join([r['text_raw'] for r in outbound_tweets[:3]])
            brand_reply_clean = " \n".join([r['text_clean'] for r in outbound_tweets[:3]])

        # Build transcript
        transcript_lines = []
        for r in thread_records:
            speaker = "Customer" if r['inbound'] else "AppleSupport"
            transcript_lines.append(f"{speaker}: {r['text_clean']}")

        full_transcript = "\n".join(transcript_lines)
        turn_count = len(thread_records)
        has_multiple_turns = turn_count > 2 or len(inbound_tweets) > 1

        conversations.append({
            'conversation_id': f"{brand_tag}_conv_{root_tid}",
            'root_tweet_id': root_tid,
            'customer_id': customer_id,
            'created_at': first_inbound['created_at'],
            'customer_message_raw': first_inbound_raw,
            'customer_message_clean': first_inbound_clean,
            'brand_response_raw': brand_reply_raw,
            'brand_response_clean': brand_reply_clean,
            'has_brand_response': bool(brand_reply_clean.strip()),
            'turn_count': turn_count,
            'has_multiple_turns': has_multiple_turns,
            'full_transcript': full_transcript
        })

    df_convs = pd.DataFrame(conversations)
    return df_convs
