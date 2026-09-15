"""Step 3: Train baselines and proposed system, index historical conversations, and evaluate on validation split."""
import sys
import pickle
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.settings import settings
from src.classifiers.trivial_classifier import TrivialIntentClassifier
from src.classifiers.ml_classifier import MLIntentClassifier
from src.classifiers.hybrid_classifier import HybridIntentClassifier
from src.retrieval.vector_store import ConversationVectorStore
from src.retrieval.retriever import HistoricalSupportRetriever

def train_and_evaluate():
    print(f"Loading train split from: {settings.TRAIN_SPLIT_CSV}")
    train_df = pd.read_csv(settings.TRAIN_SPLIT_CSV)
    print(f"Loading val split from: {settings.VAL_SPLIT_CSV}")
    val_df = pd.read_csv(settings.VAL_SPLIT_CSV)

    train_texts = train_df['customer_message_clean'].fillna('').tolist()
    train_labels = train_df['intent'].tolist()

    val_texts = val_df['customer_message_clean'].fillna('').tolist()
    val_labels = val_df['intent'].tolist()

    # 1. Baseline 1: Trivial Majority Classifier
    print("\n--- Training Baseline 1: Trivial Majority Classifier ---")
    trivial_clf = TrivialIntentClassifier()
    trivial_clf.fit(train_texts, train_labels)
    trivial_preds = [p["intent"] for p in trivial_clf.predict_batch(val_texts)]
    acc_trivial = accuracy_score(val_labels, trivial_preds)
    f1_trivial = f1_score(val_labels, trivial_preds, average="macro", zero_division=0)
    print(f"Baseline 1 Validation Accuracy: {acc_trivial:.4f} | Macro F1: {f1_trivial:.4f}")

    # 2. Baseline 2: Simple ML Classifier (TF-IDF + ComplementNB)
    print("\n--- Training Baseline 2: Simple ML Classifier (TF-IDF + ComplementNB) ---")
    ml_clf = MLIntentClassifier(alpha=0.5)
    ml_clf.fit(train_texts, train_labels)
    ml_preds = [p["intent"] for p in ml_clf.predict_batch(val_texts)]
    acc_ml = accuracy_score(val_labels, ml_preds)
    f1_ml = f1_score(val_labels, ml_preds, average="macro", zero_division=0)
    print(f"Baseline 2 Validation Accuracy: {acc_ml:.4f} | Macro F1: {f1_ml:.4f}")

    # 3. Proposed System: Hybrid Intent Classifier
    print("\n--- Training Proposed System: Hybrid Intent Classifier ---")
    hybrid_clf = HybridIntentClassifier(alpha=0.2)
    hybrid_clf.fit(train_texts, train_labels)
    hybrid_preds = [p["intent"] for p in hybrid_clf.predict_batch(val_texts)]
    acc_hybrid = accuracy_score(val_labels, hybrid_preds)
    f1_hybrid = f1_score(val_labels, hybrid_preds, average="macro", zero_division=0)
    print(f"Proposed System Validation Accuracy: {acc_hybrid:.4f} | Macro F1: {f1_hybrid:.4f}")

    print("\nDetailed Proposed System Classification Report (Validation):")
    print(classification_report(val_labels, hybrid_preds, zero_division=0))

    # 4. Build Historical Knowledge Vector Store
    print("\n--- Indexing Historical Apple Conversations (Train Split Only - Zero Leakage) ---")
    train_conv_records = train_df.to_dict('records')
    vector_store = ConversationVectorStore()
    vector_store.index_conversations(train_conv_records)
    print(f"Successfully indexed {len(vector_store.documents):,} historical support conversations.")

    # Test retrieval on sample val queries
    retriever = HistoricalSupportRetriever(vector_store)
    test_queries = [
        "My iPhone battery drains from 100% to 20% in an hour after updating iOS",
        "My screen is shattered and touch display is totally unresponsive",
        "My Apple ID is locked and I cannot get the 2FA verification code",
        "I was charged twice for an App Store subscription and need a refund",
        "Wi-Fi and Bluetooth disconnect constantly on my iPhone"
    ]
    print("\nSample Retrieval Spot-Check:")
    for q in test_queries:
        res = retriever.retrieve(q, top_k=1)
        sim = res[0]['similarity_score'] if res else 0.0
        print(f"Query: '{q}' -> Top Match Score: {sim:.2f} | Grounded Intent: {res[0]['intent'] if res else 'None'}")

    # Save trained artifacts
    models_dir = settings.DATA_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    with open(models_dir / "trivial_classifier.pkl", "wb") as f:
        pickle.dump(trivial_clf, f)
    with open(models_dir / "ml_classifier.pkl", "wb") as f:
        pickle.dump(ml_clf, f)
    with open(models_dir / "hybrid_classifier.pkl", "wb") as f:
        pickle.dump(hybrid_clf, f)
    with open(models_dir / "vector_store.pkl", "wb") as f:
        pickle.dump(vector_store, f)

    print(f"\nSaved trained models and vector store index to {models_dir}")

if __name__ == "__main__":
    train_and_evaluate()
