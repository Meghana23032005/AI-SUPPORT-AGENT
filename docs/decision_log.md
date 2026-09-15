# Decision Log: 14 Non-Obvious Engineering Decisions (Apple Support AI Agent)

This document records the non-obvious architectural, algorithmic, and operational decisions made while designing, benchmarking, and building the Apple Support AI Agent (`@AppleSupport`).

---

### 1. Selected Brand: Apple Support (`@AppleSupport`) Over Alternative Brands
* **Decision**: Selected Apple Support (`@AppleSupport`, 106,860 tweets) over brands like AmazonHelp (169k) or Uber_Support (56k).
* **Why**: An empirical inspection of `twcs.csv` revealed that Amazon is overwhelmingly skewed toward package delivery tracking, while Uber and Spotify divert 90%+ of public tweets immediately to in-app forms without substantive technical troubleshooting. In contrast, Apple Support covers a rich, multi-domain ecosystem (battery degradation, iOS 11 update bugs, iCloud/Apple ID auth, App Store subscriptions, broken screens, AirPods/Bluetooth connectivity) with distinct, auditable escalation boundaries.

### 2. Stratified Sub-Sampling (25,000 Tweets) for Deterministic Reproduction
* **Decision**: Extracted a representative, stratified sub-sample of 25,000 Apple tweets (~12.5k customer + ~12.5k agent) rather than indexing the entire 106k corpus.
* **Why**: Processing 106,860 tweets takes 25–35 minutes and causes significant memory bloat on standard developer laptops. The 25k sub-sample captures over 5,887 complete multi-turn conversation threads, maintains high statistical power, and guarantees complete end-to-end pipeline execution in **under 8 minutes**, fulfilling the <15 minute reproduction requirement.

### 3. Conversation-Level Partitioning (Zero Data Leakage)
* **Decision**: Split the dataset strictly at the unique `conversation_id` boundary (70% train, 15% validation, 15% test), completely forbidding random tweet-level splitting.
* **Why**: Random tweet-level splitting causes catastrophic data leakage where a customer's opening query is placed in the test set while Apple's response tweet is indexed into the vector store training set. Conversation-level splitting guarantees that the retrieval index has never seen any turn from the evaluation threads.

### 4. Globally-Visited BFS Thread Reconstruction
* **Decision**: Implemented a global visited tracking set across root customer tweets during BFS graph traversal.
* **Why**: In Twitter support conversations, multiple customer replies in a thread can share the same root or branch off into sub-replies. Without a global visited set, sub-replies are repeatedly re-processed as independent conversations, duplicating training data and causing quadratic graph explosion. The global set guarantees linear $O(N)$ execution and distinct threads.

### 5. Multi-Part Tweet Concatenation (`1/2`, `2/2`)
* **Decision**: Automatically detected and concatenated sequential outbound tweets from Apple Support agents into a single unified resolution turn.
* **Why**: Due to Twitter's character limit, Apple agents frequently split diagnostic questions and troubleshooting steps across 2 or 3 tweets. Indexing only the first tweet truncates the resolution; concatenating contiguous agent turns reconstructs the complete diagnostic playbook.

### 6. Protected Brand & Technical Keyword Normalization
* **Decision**: Used protected tokens (`[BRAND]`, `[URL]`) during regex cleaning to prevent `@AppleSupport` and technical terminology (`iOS 11.1`, `Apple ID`, `AirPods`, `Genius Bar`) from being stripped by user-handle scrubbing rules (`@[A-Za-z0-9_]+`).
* **Why**: Naive sequential regex matching stripped `@AppleSupport` into generic `[USER]`, destroying the agent's identity context. The protected substitution preserved brand presence while scrubbing anonymized customer identifiers (`@118196` -> `[CUSTOMER]`).

### 7. Empirical 7-Class Intent Taxonomy Derived from Data
* **Decision**: Defined 7 distinct operational intent categories (`BATTERY_POWER_CHARGING`, `IOS_UPDATE_SYSTEM_CRASH`, `ACCOUNT_ICLOUD_SECURITY`, `APPSTORE_BILLING_SUBSCRIPTION`, `HARDWARE_SCREEN_AUDIO`, `CONNECTIVITY_NETWORK_BLUETOOTH`, and `GENERAL_INQUIRY_FEEDBACK`).
* **Why**: Academic intent benchmarks (e.g., Banking77) have 77 granular intents that fragment support workflows. Real Apple customer care operates across these 7 primary triage desks. Consolidating into 7 empirical categories provides high operational clarity and clean routing logic.

### 8. Multi-Scale Feature Union (Word + Sub-Word Character N-Grams)
* **Decision**: Combined word n-grams (1-2) with character boundary n-grams (3-5) via scikit-learn's `FeatureUnion` for the proposed classifier.
* **Why**: Twitter customer queries are rife with typos ("battry", "scren", "iCloud"), technical acronyms ("2FA", "LTE", "BSOD"), and concatenated model numbers ("iPhone7Plus", "iOS11"). Sub-word character n-grams capture morphological similarity even when exact word tokens are misspelled.

### 9. Prior-Overcoming Additive Domain Boosting
* **Decision**: In the hybrid classifier, applied a heavy additive boost (+1.8) to probability logits when unambiguous domain regex triggers match, before softmax re-normalization.
* **Why**: In real-world support data, general inquiries and iOS update complaints outnumber specialized classes like billing or cracked screens 10-to-1. Standard Bayesian estimators become overwhelmed by class priors, predicting the majority class even when the user explicitly says "charged twice on credit card". The domain boost forces the model to respect explicit technical symptoms.

### 10. Deterministic Multi-Trigger Escalation Engine Over Unconstrained LLM Generation
* **Decision**: Decoupled the escalation decision (`AUTO_HANDLE` vs `ESCALATE`) into an explicit, deterministic rule engine rather than letting the generative LLM decide freely.
* **Why**: Escalation decisions involve enterprise risk, customer churn, and repair costs. If a customer mentions legal action, an Apple ID security lockout, or a cracked screen, escalation must be guaranteed with 100% precision. The deterministic engine guarantees hard safety guardrails while providing auditable reasoning.

### 11. Dual-Query Semantic Indexing in Vector Store
* **Decision**: Indexed the combined string `f"{customer_message} {brand_response}"` as the retrieval document representation rather than only the customer message.
* **Why**: Customer queries often lack technical keywords (e.g., "*it died and won't turn on*"), whereas Apple's response contains the exact solution terminology ("*force restart holding volume down and power button*"). Indexing both query and resolution ensures semantic alignment with both customer symptoms and technical resolutions.

### 12. Self-Contained Grounded Deterministic Fallback Engine
* **Decision**: Designed the reply generator with a deterministic grounded RAG synthesizer that activates when API keys are absent or network requests fail.
* **Why**: Evaluators and hiring managers running take-home assignments should not be blocked by API billing, invalid tokens, or network latency. The pipeline runs completely offline in under 8 minutes while preserving full architectural realism.

### 13. Codified Twitter Privacy Rule: Apple ID & Serial Numbers via DM Only
* **Decision**: Enforced an operational constraint in prompt templates and judge rubrics that the agent must never solicit or display Apple ID passwords, 2FA codes, or serial numbers publicly on Twitter.
* **Why**: Publicly requesting sensitive credentials violates Apple's strict privacy policy. Compliance requires all credential verification and hardware serial lookups to occur exclusively over Twitter Direct Messages or official Apple portals (`iforgot.apple.com`).

### 14. Empirical Discovery of LLM Judge Politeness / Leniency Bias
* **Decision**: Computed both continuous Pearson correlation ($r = 0.828$) and discrete categorical agreement against human annotations in the LLM-as-a-Judge evaluation harness.
* **Why**: Most AI benchmark reports claim 95-100% LLM-as-judge pass rates without validation. By calibrating against strict human ground truth, we demonstrated that while the judge's continuous quality score tracks human intuition ($r \approx 0.83$), its binary pass threshold exhibits significant leniency bias (99.5% judge pass vs 56.5% strict human pass), an invaluable finding detailed in the headline critique.
