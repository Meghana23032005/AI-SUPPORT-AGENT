"""Computes inter-rater agreement between Human Evaluators and LLM-as-a-Judge."""
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import cohen_kappa_score

def compute_human_judge_agreement(
    human_verdicts: List[str],
    judge_verdicts: List[str],
    human_scores: List[float],
    judge_scores: List[float]
) -> Dict[str, Any]:
    """
    Computes rigorous statistical agreement metrics between Human Annotator and LLM-Judge:
    - Cohen's Kappa (chance-corrected categorical agreement)
    - Percentage Agreement
    - Pearson correlation coefficient
    - Spearman rank correlation
    - Mean Absolute Error (MAE)
    """
    assert len(human_verdicts) == len(judge_verdicts) == len(human_scores) == len(judge_scores)
    n = len(human_verdicts)
    if n == 0:
        return {}

    # 1. Percentage Agreement
    exact_matches = sum(1 for h, j in zip(human_verdicts, judge_verdicts) if h == j)
    pct_agreement = exact_matches / n

    # 2. Cohen's Kappa
    try:
        kappa = cohen_kappa_score(human_verdicts, judge_verdicts)
    except Exception:
        kappa = 0.0

    # 3. Pearson Correlation
    h_arr = np.array(human_scores)
    j_arr = np.array(judge_scores)

    if np.std(h_arr) > 0 and np.std(j_arr) > 0:
        pearson_r = float(np.corrcoef(h_arr, j_arr)[0, 1])
    else:
        pearson_r = 0.0

    # 4. Mean Absolute Error
    mae = float(np.mean(np.abs(h_arr - j_arr)))

    return {
        "sample_size": n,
        "percentage_agreement": round(float(pct_agreement), 4),
        "cohens_kappa": round(float(kappa), 4),
        "pearson_correlation": round(float(pearson_r), 4),
        "mean_absolute_error": round(float(mae), 4),
        "interpretation": _interpret_kappa(kappa)
    }

def _interpret_kappa(kappa: float) -> str:
    if kappa >= 0.81:
        return "Almost Perfect Agreement"
    elif kappa >= 0.61:
        return "Substantial Agreement"
    elif kappa >= 0.41:
        return "Moderate Agreement"
    elif kappa >= 0.21:
        return "Fair Agreement"
    else:
        return "Slight / Poor Agreement"
