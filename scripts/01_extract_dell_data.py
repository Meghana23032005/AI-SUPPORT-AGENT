"""Step 1: Extract DellCares tweets from the Kaggle twcs.csv dataset."""
import os
import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.settings import settings

def extract_dell_tweets(source_csv: Path, output_csv: Path) -> pd.DataFrame:
    print(f"Reading dataset from: {source_csv}")
    if not source_csv.exists():
        raise FileNotFoundError(
            f"twcs.csv not found at {source_csv}. Please ensure dataset is downloaded."
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    dell_chunks = []
    total_scanned = 0

    chunksize = 200000
    for i, chunk in enumerate(pd.read_csv(source_csv, chunksize=chunksize)):
        total_scanned += len(chunk)
        # Filter for DellCares as author OR customer addressing @DellCares
        mask = (chunk['author_id'] == 'DellCares') | (
            chunk['text'].str.contains('@DellCares', case=False, na=False)
        )
        sub = chunk[mask]
        if len(sub) > 0:
            dell_chunks.append(sub)
        print(f"  Processed {total_scanned:,} rows... Found {sum(len(c) for c in dell_chunks):,} Dell tweets", end="\r")

    print("\nExtraction complete. Combining records...")
    df_dell = pd.concat(dell_chunks, ignore_index=True)
    df_dell.to_csv(output_csv, index=False)
    print(f"Saved {len(df_dell):,} Dell tweets to {output_csv}")
    print(f"  - Inbound (Customer): {len(df_dell[df_dell['inbound']]):,}")
    print(f"  - Outbound (Dell Support): {len(df_dell[~df_dell['inbound']]):,}")
    return df_dell

if __name__ == "__main__":
    extract_dell_tweets(settings.TWCS_CACHE_CSV, settings.RAW_DELL_CSV)
