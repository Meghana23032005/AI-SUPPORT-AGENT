# Benchmark Comparison: 200 Hand-Labelled Golden Set (Apple Support)

| System                 | Intent Acc   |   Intent Macro-F1 | Escalation Acc   |   Escalate F1 |   Auto-Handle F1 |   Retrieval Sim |   Judge Score (1-5) | Judge Pass Rate   |
|:-----------------------|:-------------|------------------:|:-----------------|--------------:|-----------------:|----------------:|--------------------:|:------------------|
| Baseline 1 (Trivial)   | 14.5%        |             0.036 | 22.0%            |         0.361 |            0     |            0    |                4.3  | 100.0%            |
| Baseline 2 (Simple ML) | 59.5%        |             0.597 | 22.0%            |         0.361 |            0     |            0.3  |                4.44 | 100.0%            |
| Proposed System        | 66.5%        |             0.666 | 74.5%            |         0.523 |            0.826 |            0.26 |                4.77 | 99.5%             |

## Human-vs-Judge Agreement
- **Percentage Agreement**: 56.5%
- **Cohen's Kappa ($\kappa$)**: 0.013 (Slight / Poor Agreement)\n- **Pearson Correlation ($r$)**: 0.828
- **Mean Absolute Error (MAE)**: 0.984
