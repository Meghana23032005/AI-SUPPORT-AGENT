"""Centralized project settings and configurations for Apple Support AI Agent."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

class Settings:
    # Brand Configuration
    SELECTED_BRAND: str = "AppleSupport"
    BRAND_DISPLAY_NAME: str = "Apple Support"

    # Directory Paths
    ROOT_DIR: Path = PROJECT_ROOT
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    SPLITS_DIR: Path = DATA_DIR / "splits"
    GOLDEN_DIR: Path = DATA_DIR / "golden"
    MODELS_DIR: Path = DATA_DIR / "models"
    PROMPTS_DIR: Path = PROJECT_ROOT / "prompts"
    DOCS_DIR: Path = PROJECT_ROOT / "docs"
    APP_DIR: Path = PROJECT_ROOT / "app"

    # Specific Data Files
    RAW_APPLE_CSV: Path = RAW_DATA_DIR / "apple_tweets_raw.csv"
    PROCESSED_THREADS_CSV: Path = PROCESSED_DATA_DIR / "apple_conversation_threads.csv"
    TRAIN_SPLIT_CSV: Path = SPLITS_DIR / "train_conversations.csv"
    VAL_SPLIT_CSV: Path = SPLITS_DIR / "val_conversations.csv"
    TEST_SPLIT_CSV: Path = SPLITS_DIR / "test_conversations.csv"
    GOLDEN_EVAL_CSV: Path = GOLDEN_DIR / "apple_golden_eval_200.csv"

    # Model & Pipeline Parameters
    RANDOM_SEED: int = 42
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15

    # Sub-sample parameters for fast, deterministic reproduction (<10 min)
    SAMPLE_TWEET_LIMIT: int = 25000

    # Intent Classification & Escalation
    INTENT_CONFIDENCE_THRESHOLD: float = 0.60
    TOP_K_RETRIEVAL: int = 3
    MIN_SIMILARITY_THRESHOLD: float = 0.18

    # API Keys (Loaded from environment, never hard-coded)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Twcs Global Path: local workspace twcs.csv prioritized
    LOCAL_TWCS_CSV: Path = PROJECT_ROOT / "twcs.csv"
    KAGGLEHUB_TWCS_CSV: Path = Path(
        os.path.expanduser(
            r"~/.cache/kagglehub/datasets/thoughtvector/customer-support-on-twitter/versions/10/twcs/twcs.csv"
        )
    )

    @property
    def TWCS_CSV(self) -> Path:
        if self.LOCAL_TWCS_CSV.exists():
            return self.LOCAL_TWCS_CSV
        return self.KAGGLEHUB_TWCS_CSV

settings = Settings()
