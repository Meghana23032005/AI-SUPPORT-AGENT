# Golden Evaluation Set Annotation Guidelines (Apple Support)

This document formalizes the sampling, intent taxonomy definitions, escalation criteria, and reply quality standards for constructing and evaluating the **200-sample Golden Evaluation Set** for `@AppleSupport`.

---

## 1. Sampling Protocol

To ensure rigorous, unbiased benchmarking, the 200 evaluation examples were constructed via **stratified sampling** strictly from the held-out conversation splits (**Test Split: 15%** and boundary cases from **Validation Split: 15%**), guaranteeing **zero data leakage** from the training split.

- **Total Sample Size**: Exactly 200 customer support interactions.
- **Strata Selection**: Balanced across the 7 empirical Apple Support intent categories (~28 samples per category).
- **Inclusion Criteria**:
  - Customer message must contain at least 15 characters of substantive text.
  - Excludes raw URLs, empty handles, or automated promotional bots.
  - Multi-turn interactions prioritized to ensure conversational realism.

---

## 2. Intent Classification Taxonomy

Annotators must classify incoming customer messages into exactly one of the following mutually exclusive categories:

| Intent Category | Operational Scope | Canonical Examples |
| :--- | :--- | :--- |
| `BATTERY_POWER_CHARGING` | Battery drain, overheating, charger cable defects, battery health degradation. | *"My iPhone 7 battery drops from 100% to 20% in 30 mins after updating."* |
| `IOS_UPDATE_SYSTEM_CRASH` | Operating system bugs, keyboard autocorrect glitches (iOS 11 'I' bug), frozen UI, boot loops. | *"After updating to iOS 11.1 my keyboard keeps replacing the letter 'I' with a question mark symbol."* |
| `ACCOUNT_ICLOUD_SECURITY` | Apple ID locked, 2FA code delivery failures, forgotten passcodes, Activation Lock, iCloud storage. | *"My Apple ID is disabled for security reasons and I can't receive verification codes."* |
| `APPSTORE_BILLING_SUBSCRIPTION` | Unrecognized charges, recurring subscriptions, refund requests, iTunes store balance. | *"I was billed $9.99 for a subscription I cancelled last week. How do I get a refund?"* |
| `HARDWARE_SCREEN_AUDIO` | Cracked screen, unresponsive touch display, microphone/speaker failures, water damage. | *"I dropped my phone and the screen is completely shattered. Touch does not work at all."* |
| `CONNECTIVITY_NETWORK_BLUETOOTH` | Wi-Fi drops, Bluetooth pairing (AirPods/car), cellular 'No Service', AirDrop failures. | *"My iPhone won't connect to my home Wi-Fi after restarting my router, keep getting incorrect password."* |
| `GENERAL_INQUIRY_FEEDBACK` | Store hours, Genius Bar booking assistance, trade-in value questions, compliments, or general feedback. | *"Can I walk into the Covent Garden Apple Store without an appointment to get my phone checked?"* |

---

## 3. Escalation Policy Criteria (`AUTO_HANDLE` vs `ESCALATE`)

The AI Agent must determine whether an inquiry can be safely resolved autonomously or requires routing to a human specialist.

### Conditions for `AUTO_HANDLE`
The inquiry can be auto-handled if:
1. **Self-Service Playbook Exists**: The issue is resolvable through known software troubleshooting (force restart combination, checking `Settings > Battery`, toggling Airplane Mode, or `Settings > General > Reset > Reset Network Settings`).
2. **Official Documentation Available**: The issue can be resolved by pointing to an official Apple Support article (`support.apple.com` or `apple.co/...`).
3. **No Private Authentication Required**: Resolution does not require accessing private Apple ID purchase records, credit card ledgers, or hardware diagnostics.

### Conditions for `ESCALATE`
The inquiry MUST be escalated to a human specialist if:
1. **Physical Hardware Damage**: Broken screens, swollen batteries, cracked glass, or liquid spills requiring physical inspection at an Apple Store Genius Bar or mail-in repair depot.
2. **Account Security Lockout**: Locked Apple ID, 2FA recovery, or Activation Lock requiring identity verification via `iforgot.apple.com` or Apple Security.
3. **Financial & Billing Disputes**: Requests for refunds, disputed in-app charges, or unauthorized credit card transactions requiring purchase history lookup.
4. **Safety Hazards**: Extreme thermal overheating, sparks, smoke, or swollen casing.
5. **Legal Threats & Severe Churn**: Explicit threats of litigation, consumer protection complaints, or persistent demands to speak with a human manager.
6. **Ambiguity / Low Confidence**: Classification confidence < 0.60 or confidence margin between top 2 intents < 0.15.

---

## 4. Grounded Reply Quality Rubric (LLM-as-a-Judge & Human Evaluation)

Each generated response is evaluated on a 1–5 scale across 5 core dimensions:

1. **Intent Appropriateness (1-5)**: Does the reply accurately address the customer's specific problem domain?
2. **Groundedness & Faithfulness (1-5)**: Are all facts, settings paths, and procedures grounded in verified Apple Support documentation? (Zero hallucinated iOS settings).
3. **Technical Actionability (1-5)**: Are troubleshooting steps clear, ordered, and directly executable by a customer?
4. **Brand Tone & Privacy Compliance (1-5)**: Is the reply empathetic, calm, and professional? Does it strictly protect user privacy by directing serial numbers and Apple ID credentials to DM?
5. **Escalation Appropriateness (1-5)**: Was the routing decision correct, and is the stated reason logically sound?

**Pass Threshold**: Overall average score $\ge 3.8 / 5.0$ with no individual dimension scoring $1$.
