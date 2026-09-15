"""End-to-end integration test for AppleSupportAgent pipeline."""
import pickle
from config.settings import settings
from src.agent import AppleSupportAgent
from src.retrieval.retriever import HistoricalSupportRetriever
from src.generation.escalation import EscalationDecisionEngine
from src.generation.generator import GroundedReplyGenerator

def test_full_agent_pipeline():
    models_dir = settings.DATA_DIR / "models"
    with open(models_dir / "hybrid_classifier.pkl", "rb") as f:
        hybrid_clf = pickle.load(f)
    with open(models_dir / "vector_store.pkl", "rb") as f:
        vector_store = pickle.load(f)

    agent = AppleSupportAgent(
        classifier=hybrid_clf,
        retriever=HistoricalSupportRetriever(vector_store),
        escalation_engine=EscalationDecisionEngine(),
        generator=GroundedReplyGenerator()
    )

    query = "My iPhone battery is draining so fast after updating iOS, drops from 100% to 20% in an hour."
    result = agent.process_message(query)

    assert "intent" in result
    assert "confidence" in result
    assert "decision" in result
    assert "escalation_reason" in result
    assert "evidence" in result
    assert "generated_response" in result
    assert "retrieved_examples" in result
    assert len(result["generated_response"]) > 20
    assert "apple" in result["generated_response"].lower() or "dm" in result["generated_response"].lower()
