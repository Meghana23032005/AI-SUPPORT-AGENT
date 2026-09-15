"""Automated evaluation metrics for intent classification, retrieval grounding, and escalation."""
from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

def evaluate_intent_classification(gold_intents: List[str], predicted_intents: List[str]) -> Dict[str, Any]:
    """Computes standard classification metrics for intent prediction."""
    acc = accuracy_score(gold_intents, predicted_intents)
    macro_f1 = f1_score(gold_intents, predicted_intents, average="macro", zero_division=0)
    weighted_f1 = f1_score(gold_intents, predicted_intents, average="weighted", zero_division=0)
    
    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "total_samples": len(gold_intents)
    }

def evaluate_escalation_decisions(gold_decisions: List[str], predicted_decisions: List[str]) -> Dict[str, Any]:
    """Computes precision, recall, and F1 for escalation decisions."""
    acc = accuracy_score(gold_decisions, predicted_decisions)
    # Target class: ESCALATE
    esc_precision = precision_score(gold_decisions, predicted_decisions, pos_label="ESCALATE", zero_division=0)
    esc_recall = recall_score(gold_decisions, predicted_decisions, pos_label="ESCALATE", zero_division=0)
    esc_f1 = f1_score(gold_decisions, predicted_decisions, pos_label="ESCALATE", zero_division=0)

    auto_precision = precision_score(gold_decisions, predicted_decisions, pos_label="AUTO_HANDLE", zero_division=0)
    auto_recall = recall_score(gold_decisions, predicted_decisions, pos_label="AUTO_HANDLE", zero_division=0)
    auto_f1 = f1_score(gold_decisions, predicted_decisions, pos_label="AUTO_HANDLE", zero_division=0)

    return {
        "accuracy": round(float(acc), 4),
        "escalate_precision": round(float(esc_precision), 4),
        "escalate_recall": round(float(esc_recall), 4),
        "escalate_f1": round(float(esc_f1), 4),
        "autohandle_precision": round(float(auto_precision), 4),
        "autohandle_recall": round(float(auto_recall), 4),
        "autohandle_f1": round(float(auto_f1), 4),
    }

def evaluate_retrieval_grounding(
    generated_replies: List[str],
    retrieved_contexts: List[List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """
    Measures lexical grounding overlap and average retrieval relevance.
    """
    sim_scores = []
    grounding_overlaps = []

    for reply, ex_list in zip(generated_replies, retrieved_contexts):
        if not ex_list:
            sim_scores.append(0.0)
            grounding_overlaps.append(0.0)
            continue

        best_sim = ex_list[0].get("similarity_score", 0.0)
        sim_scores.append(best_sim)

        # Token set overlap between generated reply and top historical reference
        ref_text = str(ex_list[0].get("brand_response") or "").lower()
        gen_tokens = set(reply.lower().split())
        ref_tokens = set(ref_text.split())

        if ref_tokens and gen_tokens:
            overlap = len(gen_tokens.intersection(ref_tokens)) / len(ref_tokens)
            grounding_overlaps.append(min(1.0, overlap))
        else:
            grounding_overlaps.append(0.0)

    return {
        "mean_top1_retrieval_similarity": round(float(np.mean(sim_scores)), 4),
        "mean_grounding_token_overlap": round(float(np.mean(grounding_overlaps)), 4)
    }
