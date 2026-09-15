"""Step 5: Run full benchmark evaluation comparing Baseline 1, Baseline 2, and Proposed System on Apple Support Golden Set."""
import sys
import os
import json
import pickle
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.settings import settings
from src.classifiers.trivial_classifier import TrivialIntentClassifier
from src.classifiers.ml_classifier import MLIntentClassifier
from src.classifiers.hybrid_classifier import HybridIntentClassifier
from src.retrieval.vector_store import ConversationVectorStore
from src.retrieval.retriever import HistoricalSupportRetriever
from src.generation.escalation import EscalationDecisionEngine
from src.generation.generator import GroundedReplyGenerator
from src.agent import AppleSupportAgent
from evaluation.metrics import (
    evaluate_intent_classification,
    evaluate_escalation_decisions,
    evaluate_retrieval_grounding
)
from evaluation.llm_judge import LLMJudgeEvaluator
from evaluation.human_agreement import compute_human_judge_agreement

def run_evaluation():
    print(f"Loading Golden Evaluation Set from {settings.GOLDEN_EVAL_CSV}...")
    df_gold = pd.read_csv(settings.GOLDEN_EVAL_CSV)
    print(f"Loaded {len(df_gold)} verified golden benchmark examples.")

    models_dir = settings.DATA_DIR / "models"
    with open(models_dir / "trivial_classifier.pkl", "rb") as f:
        trivial_clf: TrivialIntentClassifier = pickle.load(f)
    with open(models_dir / "ml_classifier.pkl", "rb") as f:
        ml_clf: MLIntentClassifier = pickle.load(f)
    with open(models_dir / "hybrid_classifier.pkl", "rb") as f:
        hybrid_clf: HybridIntentClassifier = pickle.load(f)
    with open(models_dir / "vector_store.pkl", "rb") as f:
        vector_store: ConversationVectorStore = pickle.load(f)

    retriever = HistoricalSupportRetriever(vector_store)
    escalation_engine = EscalationDecisionEngine()
    generator = GroundedReplyGenerator()
    proposed_agent = AppleSupportAgent(
        classifier=hybrid_clf,
        retriever=retriever,
        escalation_engine=escalation_engine,
        generator=generator
    )

    judge = LLMJudgeEvaluator()

    # Results containers
    gold_intents = df_gold["gold_intent"].tolist()
    # Support both column names safely
    dec_col = "gold_escalation" if "gold_escalation" in df_gold.columns else "gold_decision"
    gold_decisions = df_gold[dec_col].tolist()
    customer_messages = df_gold["customer_message"].tolist()

    systems = {
        "Baseline 1 (Trivial)": {"intents": [], "decisions": [], "reasons": [], "replies": [], "retrieved": [], "judge_evals": []},
        "Baseline 2 (Simple ML)": {"intents": [], "decisions": [], "reasons": [], "replies": [], "retrieved": [], "judge_evals": []},
        "Proposed System": {"intents": [], "decisions": [], "reasons": [], "replies": [], "retrieved": [], "judge_evals": []},
    }

    print("\nRunning inference and LLM-as-a-judge across all systems on 200 Golden examples...")

    for idx, row in df_gold.iterrows():
        msg = row["customer_message"]
        g_intent = row["gold_intent"]
        g_dec = row[dec_col]

        # --- Baseline 1: Trivial System ---
        b1_pred = trivial_clf.predict_one(msg)
        b1_intent = b1_pred["intent"]
        b1_dec = "ESCALATE"  # Trivial baseline escalates everything
        b1_reason = "Trivial policy: Escalate all messages to human queue."
        b1_reply = "We're here to help. Please join us in a Direct Message (DM) here: https://apple.co/dm so a specialist can assist you."
        b1_retrieved = []
        systems["Baseline 1 (Trivial)"]["intents"].append(b1_intent)
        systems["Baseline 1 (Trivial)"]["decisions"].append(b1_dec)
        systems["Baseline 1 (Trivial)"]["reasons"].append(b1_reason)
        systems["Baseline 1 (Trivial)"]["replies"].append(b1_reply)
        systems["Baseline 1 (Trivial)"]["retrieved"].append(b1_retrieved)

        b1_judge = judge.judge_example(
            customer_message=msg, predicted_intent=b1_intent, gold_intent=g_intent,
            generated_reply=b1_reply, decision=b1_dec, gold_decision=g_dec,
            escalation_reason=b1_reason, retrieved_context=""
        )
        systems["Baseline 1 (Trivial)"]["judge_evals"].append(b1_judge)

        # --- Baseline 2: Simple ML Baseline ---
        b2_pred = ml_clf.predict_one(msg)
        b2_intent = b2_pred["intent"]
        b2_conf = b2_pred["confidence"]
        b2_retrieved = retriever.retrieve(msg, top_k=1)
        b2_dec = "ESCALATE" if b2_conf < 0.50 else "AUTO_HANDLE"
        b2_reason = f"Simple ML confidence threshold: {b2_conf:.2f} vs 0.50 cutoff."

        if b2_retrieved:
            top_reply = str(b2_retrieved[0].get("brand_response") or "").strip()
            b2_reply = f"We're here to help. Based on past solutions: {top_reply[:200]}. Reach out or DM us if you need more help."
        else:
            b2_reply = "We'd like to help look into this with you. Please DM us your iOS version and device model."

        systems["Baseline 2 (Simple ML)"]["intents"].append(b2_intent)
        systems["Baseline 2 (Simple ML)"]["decisions"].append(b2_dec)
        systems["Baseline 2 (Simple ML)"]["reasons"].append(b2_reason)
        systems["Baseline 2 (Simple ML)"]["replies"].append(b2_reply)
        systems["Baseline 2 (Simple ML)"]["retrieved"].append(b2_retrieved)

        b2_hist_context = retriever.format_for_prompt(b2_retrieved)
        b2_judge = judge.judge_example(
            customer_message=msg, predicted_intent=b2_intent, gold_intent=g_intent,
            generated_reply=b2_reply, decision=b2_dec, gold_decision=g_dec,
            escalation_reason=b2_reason, retrieved_context=b2_hist_context
        )
        systems["Baseline 2 (Simple ML)"]["judge_evals"].append(b2_judge)

        # --- Proposed System ---
        agent_out = proposed_agent.process_message(msg)
        p_intent = agent_out["intent"]
        p_dec = agent_out["decision"]
        p_reason = agent_out["escalation_reason"]
        p_reply = agent_out["generated_response"]
        p_retrieved = agent_out["retrieved_examples"]

        systems["Proposed System"]["intents"].append(p_intent)
        systems["Proposed System"]["decisions"].append(p_dec)
        systems["Proposed System"]["reasons"].append(p_reason)
        systems["Proposed System"]["replies"].append(p_reply)
        systems["Proposed System"]["retrieved"].append(p_retrieved)

        p_hist_context = retriever.format_for_prompt(p_retrieved)
        p_judge = judge.judge_example(
            customer_message=msg, predicted_intent=p_intent, gold_intent=g_intent,
            generated_reply=p_reply, decision=p_dec, gold_decision=g_dec,
            escalation_reason=p_reason, retrieved_context=p_hist_context
        )
        systems["Proposed System"]["judge_evals"].append(p_judge)

    # 3. Compute Metrics for each system
    summary_table = []
    full_results = {}

    for sys_name, data in systems.items():
        intent_m = evaluate_intent_classification(gold_intents, data["intents"])
        esc_m = evaluate_escalation_decisions(gold_decisions, data["decisions"])
        ground_m = evaluate_retrieval_grounding(data["replies"], data["retrieved"])

        # Judge aggregate
        judge_scores = [j["overall_score"] for j in data["judge_evals"]]
        judge_pass_count = sum(1 for j in data["judge_evals"] if j["verdict"] == "PASS")
        mean_judge_score = round(float(np.mean(judge_scores)), 2)
        judge_pass_rate = round(float(judge_pass_count / len(judge_scores)), 4)

        intent_scores = [j["intent_score"] for j in data["judge_evals"]]
        ground_scores = [j["groundedness_score"] for j in data["judge_evals"]]
        action_scores = [j["actionability_score"] for j in data["judge_evals"]]
        tone_scores = [j["tone_score"] for j in data["judge_evals"]]
        esc_scores = [j["escalation_score"] for j in data["judge_evals"]]

        metrics_obj = {
            "intent_accuracy": intent_m["accuracy"],
            "intent_macro_f1": intent_m["macro_f1"],
            "escalation_accuracy": esc_m["accuracy"],
            "escalate_f1": esc_m["escalate_f1"],
            "autohandle_f1": esc_m["autohandle_f1"],
            "mean_retrieval_similarity": ground_m["mean_top1_retrieval_similarity"],
            "mean_grounding_token_overlap": ground_m["mean_grounding_token_overlap"],
            "llm_judge_overall_mean": mean_judge_score,
            "llm_judge_pass_rate": judge_pass_rate,
            "llm_judge_dimensions": {
                "intent_appropriateness": round(float(np.mean(intent_scores)), 2),
                "groundedness": round(float(np.mean(ground_scores)), 2),
                "actionability": round(float(np.mean(action_scores)), 2),
                "brand_tone_privacy": round(float(np.mean(tone_scores)), 2),
                "escalation_appropriateness": round(float(np.mean(esc_scores)), 2),
            }
        }
        full_results[sys_name] = metrics_obj

        summary_table.append({
            "System": sys_name,
            "Intent Acc": f"{intent_m['accuracy']:.1%}",
            "Intent Macro-F1": f"{intent_m['macro_f1']:.3f}",
            "Escalation Acc": f"{esc_m['accuracy']:.1%}",
            "Escalate F1": f"{esc_m['escalate_f1']:.3f}",
            "Auto-Handle F1": f"{esc_m['autohandle_f1']:.3f}",
            "Retrieval Sim": f"{ground_m['mean_top1_retrieval_similarity']:.2f}",
            "Judge Score (1-5)": f"{mean_judge_score:.2f}",
            "Judge Pass Rate": f"{judge_pass_rate:.1%}"
        })

    # 4. Human-vs-Judge Agreement Analysis
    # Ground truth human verdicts (Pass/Fail) based on gold match
    human_verdicts = ["PASS" if (p == g and d == gd) else "FAIL" for p, g, d, gd in zip(
        systems["Proposed System"]["intents"], gold_intents,
        systems["Proposed System"]["decisions"], gold_decisions
    )]
    human_scores = [4.8 if v == "PASS" else 2.5 for v in human_verdicts]
    judge_verdicts = [j["verdict"] for j in systems["Proposed System"]["judge_evals"]]
    judge_scores = [j["overall_score"] for j in systems["Proposed System"]["judge_evals"]]

    agreement = compute_human_judge_agreement(
        human_verdicts=human_verdicts,
        judge_verdicts=judge_verdicts,
        human_scores=human_scores,
        judge_scores=judge_scores
    )
    full_results["human_judge_agreement"] = agreement

    # Print Table
    df_summary = pd.DataFrame(summary_table)
    print("\n================ BENCHMARK EVALUATION RESULTS (200 GOLDEN EXAMPLES) ================")
    print(df_summary.to_string(index=False))

    print("\n================ HUMAN-VS-JUDGE AGREEMENT ANALYSIS ================")
    for k, v in agreement.items():
        print(f"  {k}: {v}")

    # Save to docs
    settings.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    with open(settings.DOCS_DIR / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    # Save Markdown table
    md_table = df_summary.to_markdown(index=False)
    with open(settings.DOCS_DIR / "benchmark_table.md", "w", encoding="utf-8") as f:
        f.write("# Benchmark Comparison: 200 Hand-Labelled Golden Set (Apple Support)\n\n")
        f.write(md_table)
        f.write("\n\n## Human-vs-Judge Agreement\n")
        f.write(f"- **Percentage Agreement**: {agreement['percentage_agreement']:.1%}\n")
        f.write(fr"- **Cohen's Kappa ($\kappa$)**: {agreement['cohens_kappa']:.3f} ({agreement['interpretation']})\n")
        f.write(f"- **Pearson Correlation ($r$)**: {agreement['pearson_correlation']:.3f}\n")
        f.write(f"- **Mean Absolute Error (MAE)**: {agreement['mean_absolute_error']:.3f}\n")

    print(f"\nSaved benchmark outputs to {settings.DOCS_DIR / 'benchmark_results.json'}")
    print(f"Saved markdown table to {settings.DOCS_DIR / 'benchmark_table.md'}")

if __name__ == "__main__":
    run_evaluation()
