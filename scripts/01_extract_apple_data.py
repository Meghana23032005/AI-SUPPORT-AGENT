"""Step 1: Extract AppleSupport tweets from the Kaggle twcs.csv dataset."""
import os
import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.settings import settings

def extract_apple_tweets(source_csv: Path, output_csv: Path, sample_limit: int = 25000) -> pd.DataFrame:
    print(f"Reading dataset from: {source_csv}")
    if not source_csv.exists():
        raise FileNotFoundError(
            f"twcs.csv not found at {source_csv}. Please ensure dataset is downloaded."
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    apple_chunks = []
    total_scanned = 0
    total_apple = 0

    chunksize = 200000
    for i, chunk in enumerate(pd.read_csv(source_csv, chunksize=chunksize)):
        total_scanned += len(chunk)
        # Filter for AppleSupport as author OR customer addressing @AppleSupport
        mask = (chunk['author_id'] == 'AppleSupport') | (
            chunk['text'].str.contains('@AppleSupport', case=False, na=False)
        )
        sub = chunk[mask]
        if len(sub) > 0:
            apple_chunks.append(sub)
            total_apple += len(sub)

        print(f"  Processed {total_scanned:,} rows... Found {total_apple:,} Apple tweets", end="\r")
        if sample_limit and total_apple >= sample_limit:
            print(f"\nReached sample limit of {sample_limit:,} tweets for fast reproducible execution.")
            break

    print("\nExtraction complete. Combining records...")
    df_apple = pd.concat(apple_chunks, ignore_index=True)
    if sample_limit and len(df_apple) > sample_limit:
        df_apple = df_apple.iloc[:sample_limit]

    df_apple.to_csv(output_csv, index=False)
    print(f"Saved {len(df_apple):,} Apple tweets to {output_csv}")
    print(f"  - Inbound (Customer): {len(df_apple[df_apple['inbound']]):,}")
    print(f"  - Outbound (Apple Support): {len(df_apple[~df_apple['inbound']]):,}")
    return df_apple

if __name__ == "__main__":
    extract_apple_tweets(settings.TWCS_CSV, settings.RAW_APPLE_CSV, settings.SAMPLE_TWEET_LIMIT)
