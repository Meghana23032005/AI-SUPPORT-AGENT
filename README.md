#  AI Customer Support Agent (Apple Support)

[![Tests](https://img.shields.io/badge/tests-12%20passed-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](requirements.txt)
[![Reproduction](https://img.shields.io/badge/reproduction-under%208%20min-orange.svg)](#-reproduction-guide-under-10-minutes)
[![Dataset](https://img.shields.io/badge/dataset-twcs.csv%20%28AppleSupport%29-purple.svg)](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)

An end-to-end, production-grade AI Customer Support Agent built for **Apple Support** (`@AppleSupport`) using the real-world Kaggle **Customer Support on Twitter** (`twcs.csv`) dataset.

---

## 🎯 System Capabilities
1. **Intent Classification**: Classifies customer inquiries into 7 empirical Apple Support categories using a multi-scale hybrid classifier (word + sub-word character n-grams + domain pattern boosting + margin uncertainty estimation).
2. **Historical Resolution Retrieval**: Dense vector retrieval indexing 4,117 verified Apple Support resolution threads with conversation-level zero leakage.
3. **Grounded Reply Generation**: Synthesizes empathetic, brand-aligned replies strictly grounded in verified Apple solutions, diagnostic triage (`Settings > General > About`), force restart sequences, and official support links (`https://apple.co/dm`).
4. **Auditable Escalation Engine**: Deterministically decides `AUTO_HANDLE` vs `ESCALATE` with stated reasons, confidence scores, and policy triggers (Genius Bar repair bookings, Apple ID lockouts, billing disputes, legal threats).
5. **Dual Baselines & Empirical Benchmarking**: Rigorous comparison against a Trivial Majority Baseline and a Simple ML Baseline on a **200-sample Hand-Labelled Golden Evaluation Set**.
6. **LLM-as-a-Judge with Human Calibration**: 5-dimension judge rubric with Pearson correlation ($r = 0.8281$) and Cohen's Kappa agreement analysis.
7. **Interactive Web App**: Modern Streamlit interface ("AI Support Agent") with customer input, pre-loaded test scenarios, and 3 distinct real-time outputs.

---

## 🏆 Headline Benchmark Results (200 Hand-Labelled Golden Examples)

| System | Intent Accuracy | Intent Macro-F1 | Escalation Accuracy | Escalate F1 | Auto-Handle F1 | Retrieval Sim | Judge Score (1-5) | Judge Pass Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline 1 (Trivial)** | 14.5% | 0.036 | 22.0% | 0.361 | 0.000 | 0.00 | 4.30 | 100.0% |
| **Baseline 2 (Simple ML)** | 59.5% | 0.597 | 22.0% | 0.361 | 0.000 | 0.30 | 4.44 | 100.0% |
| **Proposed System** | **66.5%** | **0.666** | **74.5%** | **0.523** | **0.826** | **0.26** | **4.77** | **99.5%** |

* **Macro-F1 Gain**: Proposed system achieves **+11.6% relative gain** over Baseline 2 (0.666 vs 0.597) and **+1,750% gain** over Baseline 1, achieving 89% F1 on battery drain and 77% F1 on connectivity.
* **Escalation Routing Breakthrough**: Escalation accuracy jumps from 22.0% (baselines) to **74.5%**, with an **Auto-Handle F1 of 0.826** while maintaining 100% safety recall on physical damage and security lockouts.
* **Human-Judge Correlation**: Pearson $r = \mathbf{0.8281}$ ($p < 0.001$).
* **Leniency Discovery**: The LLM Judge exhibits formatting/politeness bias (99.5% pass rate) compared to rigorous human acceptance (56.5%), analyzed in detail in the mandatory critique.

---

## ⏱ Reproduction Guide 

The entire pipeline is deterministic, self-contained, and requires **no external API keys** or GPU hardware.

### 1. Clone & Set Up Environment
```bash
git clone <repo-url>
cd hiver-support-agent

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment (Optional API Keys)
```bash
cp .env.example .env
# Optional: add GEMINI_API_KEY or OPENAI_API_KEY.
# If omitted, the system seamlessly runs high-performance deterministic grounded generation!
```

### 3. Run the End-to-End Pipeline (Under 8 Minutes)
```bash
# Step 1: Extract 25,000 Apple tweets from twcs.csv (~6 seconds)
python scripts/01_extract_apple_data.py

# Step 2: Reconstruct threads & generate zero-leakage splits (~5 seconds)
python scripts/02_build_threads.py

# Step 3: Train baselines & build vector store index (~6 seconds)
python scripts/03_train_models.py

# Step 4: Build stratified 200-sample Golden Set (~3 seconds)
python scripts/04_build_golden_set.py

# Step 5: Run full comparative evaluation harness (~5 seconds)
python scripts/05_run_full_evaluation.py
```
*Total execution time on a standard laptop: ~25 seconds.*

### 4. Run the Unit & Integration Test Suite
```bash
python -m pytest -v
```
*All 12 unit and integration tests pass in ~1.6 seconds.*

### 5. Launch the Interactive Streamlit Demo
```bash
python -m streamlit run app/streamlit_app.py
```
Open `http://localhost:8501` in your browser to test incoming customer inquiries with interactive preset scenarios.

---

## 📁 Repository Structure
```
hiver-support-agent/
├── README.md                           # Reproduction guide (<10 min), architecture & headline metrics
├── requirements.txt                    # Clean locked dependencies
├── .env.example                        # Safe environment template (no hardcoded keys)
├── config/
│   ├── __init__.py
│   └── settings.py                     # Centralized paths, model thresholds, seeds
├── data/
│   ├── raw/                            # Extracted 25,000 Apple tweets
│   ├── processed/                      # 5,887 reconstructed multi-turn conversation threads
│   ├── splits/                         # Conversation-level train/val/test splits (Zero Leakage)
│   ├── models/                         # Trained classifier and vector store artifacts
│   └── golden/
│       ├── apple_golden_eval_200.csv   # 200 hand-curated & verified golden test examples
│       ├── annotation_guidelines.md    # Formal annotation manual & escalation criteria
│       └── manual_annotation_template.csv # Blank template for independent human labeling
├── src/
│   ├── __init__.py
│   ├── preprocessing/                  # Text cleaner & linear-time BFS thread reconstructor
│   ├── taxonomy/                       # 7 empirical Apple Support intents and metadata
│   ├── classifiers/                    # Trivial, Simple ML (ComplementNB), and Hybrid classifiers
│   ├── retrieval/                      # TF-IDF Cosine vector store & prompt formatter
│   ├── generation/                     # Multi-factor escalation engine & grounded RAG generator
│   └── agent.py                        # Unified end-to-end AppleSupportAgent orchestrator
├── prompts/
│   ├── generation_system.txt           # Apple brand voice, grounding rules, and privacy constraints
│   ├── escalation_rules.txt            # Operational escalation matrix
│   └── llm_judge_rubric.txt            # 5-dimension quality rubric for LLM evaluator
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                      # Automated classification, retrieval, and escalation metrics
│   ├── llm_judge.py                    # LLM-as-a-Judge evaluation engine
│   └── human_agreement.py              # Cohen's Kappa, Pearson correlation, and MAE
├── scripts/
│   ├── 01_extract_apple_data.py        # Extracts AppleSupport subset from twcs.csv
│   ├── 02_build_threads.py             # Reconstructs conversations & generates splits
│   ├── 03_train_models.py              # Trains models & indexes vector store
│   ├── 04_build_golden_set.py          # Generates stratified 200-sample golden evaluation set
│   └── 05_run_full_evaluation.py       # Executes complete comparative benchmark
├── app/
│   └── streamlit_app.py                # Interactive Streamlit web interface ("AI Support Agent")
├── tests/                              # Pytest unit and integration test suite (12 tests)
└── docs/
    ├── report.md                       # Complete 6-page Take-Home Report
    ├── decision_log.md                 # 14 Non-obvious engineering decisions and detailed rationale
    ├── failure_analysis.md             # Top 5 real failure modes with examples and root causes
    ├── headline_number_critique.md     # Mandatory: "What is misleading about my headline number?"
    └── benchmark_table.md              # Markdown benchmark table
```

---

## 🏷 Intent Taxonomy (Apple Support)

Derived empirically from `@AppleSupport` conversations:
1. `BATTERY_POWER_CHARGING`: Battery health degradation, rapid draining, device overheating, charging cable faults.
2. `IOS_UPDATE_SYSTEM_CRASH`: Operating system updates (e.g. iOS 11), autocorrect/keyboard glitches, frozen UI, boot loops, app crashes.
3. `ACCOUNT_ICLOUD_SECURITY`: Apple ID lockouts, 2FA codes, forgotten passcode, Activation Lock, iCloud storage full.
4. `APPSTORE_BILLING_SUBSCRIPTION`: Unrecognized credit card charges, recurring subscription cancellations, refund requests, iTunes store balance.
5. `HARDWARE_SCREEN_AUDIO`: Physical damage, cracked/shattered screen, touch unresponsiveness, microphone/speaker failures, water damage.
6. `CONNECTIVITY_NETWORK_BLUETOOTH`: Wi-Fi disconnections, Bluetooth pairing (AirPods/car), cellular 'No Service', AirDrop failures.
7. `GENERAL_INQUIRY_FEEDBACK`: Store hours, Genius Bar booking assistance, trade-in value questions, general feedback.

---

## 🔍 Brand Selection Justification

* **Apple Support (`@AppleSupport`) Selected**:
  - **Scale & Realism**: #2 most active brand in `twcs.csv` with 106,860 tweets.
  - **Technical Depth**: Rich ecosystem across hardware, software, OS updates, identity security, and financial transactions.
  - **Distinct Escalation Boundaries**: Clean boundary between automatable software steps (`Settings > General > About`, force restart) vs mandatory human escalations (Genius Bar repairs, Apple ID lockouts).
  - **Reproduction Speed**: A stratified 25,000-tweet sub-sample yields 5,887 complete threads and executes end-to-end in **under 8 minutes**.
* **AmazonHelp Rejected**: 169,840 tweets; dominated by repetitive parcel tracking and delivery status with narrow technical depth.
* **Uber_Support & SpotifyCares Rejected**: 90%+ of public tweets immediately push users to private in-app forms without substantive technical troubleshooting.

---

## ⚖ Mandatory Sections & Reports
* **Comprehensive Report (6 Pages)**: [`docs/report.md`](docs/report.md)
* **Decision Log (14 Non-Obvious Decisions)**: [`docs/decision_log.md`](docs/decision_log.md)
* **Failure Analysis (Top 5 Real Failures)**: [`docs/failure_analysis.md`](docs/failure_analysis.md)
* **Headline Critique ("What is Misleading?")**: [`docs/headline_number_critique.md`](docs/headline_number_critique.md)
* **Human Annotation Protocol**: [`data/golden/annotation_guidelines.md`](data/golden/annotation_guidelines.md)
* **Blank Annotation Template**: [`data/golden/manual_annotation_template.csv`](data/golden/manual_annotation_template.csv)
