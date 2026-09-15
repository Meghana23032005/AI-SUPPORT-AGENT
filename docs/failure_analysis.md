# Top 5 Failure Modes: Empirical Analysis, Real Examples & Root Hypotheses (Apple Support)

This document details the top 5 failure modes observed during empirical testing and benchmark evaluation on the 200-example Golden Evaluation Set for the Apple Support AI Agent (`@AppleSupport`).

---

### Failure Mode 1: Sub-Intent Dominance in Compound iOS Update & Battery Inquiries
* **Real Customer Example**:
  > *"Ever since updating to iOS 11.1 yesterday, my iPhone 7 battery drops from 100% to 20% in an hour and the phone is boiling hot."*
* **Gold Label**: `BATTERY_POWER_CHARGING` | `AUTO_HANDLE`
* **Model Prediction**: `IOS_UPDATE_SYSTEM_CRASH` (Confidence: 0.62)
* **Root Cause & Hypothesis**:
  Customers on Twitter frequently frame their symptoms with a temporal trigger clause ("*Ever since updating to iOS 11.1 yesterday...*"). Because the tokens "updating" and "ios 11.1" have high term frequency and strong weights in the training set, the statistical classifier prioritized the temporal preface over the actual underlying symptom clause ("*battery drops*", "*boiling hot*"). While the advice for both involves diagnostic triage, misclassifying the intent degrades downstream vector retrieval relevance.
* **Mitigation**: Implement dependency parsing or sentence segmentation that separates the temporal context ("*updating to iOS 11*") from the predicate symptom ("*battery drops*", "*boiling hot*"), weighting the predicate more heavily.

---

### Failure Mode 2: Multi-Intent Collision Between Hardware Screen Crack and Financial / Warranty Claims
* **Real Customer Example**:
  > *"I dropped my phone and the screen is completely shattered. I bought AppleCare+ last month, how do I get a refund or free replacement screen?"*
* **Gold Label**: `HARDWARE_SCREEN_AUDIO` | `ESCALATE`
* **Model Prediction**: `APPSTORE_BILLING_SUBSCRIPTION` (Confidence: 0.54)
* **Root Cause & Hypothesis**:
  The customer presented a multi-intent inquiry combining physical hardware destruction ("*dropped phone*", "*screen shattered*") with warranty financial questions ("*bought AppleCare+*", "*refund*"). In a single-label classification architecture, dense financial vocabulary competed with hardware damage keywords. In this instance, the classifier leaned toward billing, creating an unsafe scenario where a billing automated link could have been provided rather than booking a physical repair.
* **Mitigation**: Introduce multi-label classification and enforce a **Highest Operational Risk Precedence Rule**: if *any* physical damage or hardware hazard is detected, the pipeline automatically routes to `HARDWARE_SCREEN_AUDIO` and forces human/Genius Bar escalation.

---

### Failure Mode 3: Lexical Vector Store Collision on Apple ID Passcode vs Routine iCloud Storage FAQs
* **Real Customer Example**:
  > *"My Apple ID account is locked and disabled because I forgot my passcode, how do I recover my data?"*
* **Gold Intent**: `ACCOUNT_ICLOUD_SECURITY` | `ESCALATE` (Account Recovery)
* **Retrieved Historical Response**:
  > *"You can manage and free up your iCloud storage by going to Settings > [Your Name] > iCloud > Manage Storage to delete old device backups."*
* **Root Cause & Hypothesis**:
  TF-IDF vector retrieval relies on lexical token overlap. The tokens "apple id", "account", "data", and "storage" have high co-occurrence across routine, benign iCloud storage conversations. The vector store matched the dominant storage management documentation rather than the critical account recovery workflow (`iforgot.apple.com`).
* **Mitigation**: Replace unigram TF-IDF retrieval with dense semantic bi-encoders (e.g. `all-MiniLM-L6-v2`) fine-tuned with contrastive loss, combined with intent-partitioned index filtering (sub-indexing Account Recovery separately from Cloud Storage).

---

### Failure Mode 4: LLM Judge Politeness / Formatting Bias Toward Inappropriate Diagnostic Deflection
* **Generated Reply Example**:
  > *"We're here to help! Could you check your current iOS version under Settings > General > About? If your screen is unresponsive, meet us in DM here: https://apple.co/dm. ^AppleSupport"*
* **Human Score**: 2.5/5 (FAIL: Ineffective; instructing a user whose touchscreen is shattered/unresponsive to navigate into the Settings app).
* **LLM Judge Score**: 4.4/5 (PASS).
* **Root Cause & Hypothesis**:
  LLM evaluators suffer from a well-documented **politeness, formatting, and surface-feature bias**. The generated reply was grammatically flawless, empathetic, and contained standard Apple links. The LLM judge rewarded these surface stylistic attributes, failing to identify the physical logical contradiction: a user with an unresponsive, broken touchscreen cannot open the Settings app to check their iOS version.
* **Mitigation**: Add explicit negative assertion tests in the judge prompt: penalize replies that suggest on-device screen navigation to customers reporting broken, unresponsive, or black screens (-2.5 deduction).

---

### Failure Mode 5: Hyperbolic Customer Frustration Triggering False Escalation on Self-Service Inquiries
* **Real Customer Example**:
  > *"Your new iOS update is absolute garbage, battery dies every 2 hours! How do I turn on Low Power Mode?"*
* **Gold Label**: `BATTERY_POWER_CHARGING` | `AUTO_HANDLE`
* **Model Prediction**: `ESCALATE` (Trigger: "absolute garbage" sentiment trigger)
* **Root Cause & Hypothesis**:
  The escalation policy engine includes sensitivity rules designed to catch extreme customer anger and churn risk. Here, the customer used hyperbolic rhetorical venting ("*absolute garbage*") while asking a routine, readily automatable query ("*How do I turn on Low Power Mode?*"). The keyword safety net treated this as an urgent human escalation, driving up human specialist queue loads unnecessarily.
* **Mitigation**: Deploy a calibrated sentiment-intensity model rather than binary keyword triggers, differentiating between venting about an iOS battery feature versus genuine threats of litigation, fraud claims, or executive escalation.
