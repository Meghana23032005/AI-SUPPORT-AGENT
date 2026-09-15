"""Streamlit Demo Application for Apple Support AI Agent."""
import sys
from pathlib import Path
import pickle
import streamlit as st
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.settings import settings
from src.retrieval.retriever import HistoricalSupportRetriever
from src.generation.escalation import EscalationDecisionEngine
from src.generation.generator import GroundedReplyGenerator
from src.agent import AppleSupportAgent
from src.taxonomy.intent_definitions import INTENT_METADATA

# Page config
st.set_page_config(
    page_title="AI Support Agent | Apple Support",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Apple-inspired clean aesthetic)
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1D1D1F;
        margin-bottom: 0.1rem;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #86868B;
        margin-bottom: 1.5rem;
    }
    .status-badge-auto {
        background-color: #E8F5E9;
        color: #1B5E20;
        border: 1px solid #4CAF50;
        padding: 7px 16px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        font-size: 0.95rem;
    }
    .status-badge-esc {
        background-color: #FFEBEE;
        color: #B71C1C;
        border: 1px solid #E53935;
        padding: 7px 16px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        font-size: 0.95rem;
    }
    .reply-card {
        background-color: #F5F5F7;
        border-left: 4px solid #0071E3;
        padding: 18px;
        border-radius: 8px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-size: 0.98rem;
        line-height: 1.6;
        white-space: pre-wrap;
        margin-top: 10px;
        color: #1D1D1F;
    }
    .evidence-item {
        background: #F2F2F7;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 0.88rem;
        margin-bottom: 6px;
        border-left: 3px solid #8E8E93;
    }
    .stButton>button {
        background-color: #0071E3;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0077ED;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_agent():
    """Loads model artifacts and initializes the agent."""
    models_dir = settings.DATA_DIR / "models"
    with open(models_dir / "hybrid_classifier.pkl", "rb") as f:
        hybrid_clf = pickle.load(f)
    with open(models_dir / "vector_store.pkl", "rb") as f:
        vector_store = pickle.load(f)

    retriever = HistoricalSupportRetriever(vector_store)
    escalation_engine = EscalationDecisionEngine()
    generator = GroundedReplyGenerator()

    return AppleSupportAgent(
        classifier=hybrid_clf,
        retriever=retriever,
        escalation_engine=escalation_engine,
        generator=generator
    )

agent = load_agent()

# Header
st.markdown('<div class="main-header">AI Support Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Apple Support (@AppleSupport) • Intent Classification, Grounded Reply Synthesis & Auditable Escalation Routing</div>', unsafe_allow_html=True)

# Sidebar: Preset Examples & Brand Info
st.sidebar.title("Customer Test Scenarios")
st.sidebar.markdown("**Brand**: Apple Support (`@AppleSupport`)")
st.sidebar.markdown("**Grounding Knowledge Base**: 4,117 Verified Threads")

presets = {
    "Select a pre-loaded query...": "",
    "Battery Rapid Drain (Auto-Handle)": "My iPhone 7 battery is draining super fast after updating iOS, drops from 100% to 20% in 1 hour while idle. Gets really hot too.",
    "iOS 11 Keyboard Autocorrect Bug (Auto-Handle)": "Why does my iPhone keep changing the capital letter 'I' into an 'A' with a question mark symbol when I type in messages?",
    "Wi-Fi & Bluetooth Disconnecting (Auto-Handle)": "My Wi-Fi keeps disconnecting every few minutes and my AirPods won't pair with my phone anymore.",
    "Cracked Screen & Unresponsive Touch (Escalate)": "I accidentally dropped my iPhone on concrete, the screen is completely cracked and touch is not responding. How do I get it fixed?",
    "Apple ID Locked & 2FA Failure (Escalate)": "My Apple ID is locked for security reasons and I am not receiving the two factor authentication code on my trusted number. I am completely locked out.",
    "Unauthorized App Store Charge (Escalate)": "I was just charged $49.99 on my credit card for an App Store subscription I never authorized. I want an immediate refund.",
    "Legal Class Action Threat (Escalate)": "Your update bricked my phone and support ignored me for two weeks. This is fraud. I am consulting my attorney to file a class action lawsuit."
}

selected_preset = st.sidebar.selectbox("Test Pre-loaded Scenarios:", list(presets.keys()))
default_text = presets[selected_preset] if selected_preset != "Select a pre-loaded query..." else ""

# Input Section: Customer enters their message
customer_input = st.text_area(
    "Enter Customer Message:",
    value=default_text,
    height=110,
    placeholder="Type incoming customer tweet or inquiry to @AppleSupport..."
)

col_btn, col_blank = st.columns([1, 4])
with col_btn:
    run_button = st.button("Process Message", type="primary", use_container_width=True)

if run_button and customer_input.strip():
    with st.spinner("Processing through Apple Support AI Pipeline..."):
        result = agent.process_message(customer_input.strip())

    st.markdown("---")

    # 3 Required Output Components
    col1, col2 = st.columns([1, 1])

    with col1:
        # 1. Intent Classification
        st.subheader("1. Intent Classification")
        intent_cat = result["intent"]
        intent_meta = INTENT_METADATA.get(intent_cat, {})
        st.markdown(f"**Identified Intent:** `{intent_cat}`")
        st.caption(intent_meta.get("description", ""))

        conf = result["confidence"]
        st.progress(min(1.0, conf), text=f"Confidence: {conf:.1%}")

        if result.get("is_ambiguous"):
            st.warning("Prediction Ambiguity: Top two intent classes have a narrow margin.")

        st.markdown("")
        # 3. Escalation Decision (with Stated Reason)
        st.subheader("2. Escalation Decision")
        decision = result["decision"]
        if decision == "AUTO_HANDLE":
            st.markdown(f'<div class="status-badge-auto">✔ AUTO_HANDLE</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="status-badge-esc">⚠ ESCALATE TO HUMAN SPECIALIST</div>', unsafe_allow_html=True)

        st.markdown(f"**Stated Reason:**\n> {result['escalation_reason']}")

        st.markdown("**Policy Triggers & Evidence:**")
        for ev in result["evidence"]:
            st.markdown(f'<div class="evidence-item">• {ev}</div>', unsafe_allow_html=True)

    with col2:
        # 2. Grounded Reply
        st.subheader("3. Grounded Draft Reply")
        st.markdown(f'<div class="reply-card">{result["generated_response"]}</div>', unsafe_allow_html=True)
        st.caption(f"Synthesis Engine: `{result['generation_method']}` | Grounded in Historical Resolutions: `True`")

        # Probability Breakdown
        with st.expander("View Intent Probability Distribution"):
            probs = result.get("all_probabilities", {})
            if probs:
                df_probs = pd.DataFrame(list(probs.items()), columns=["Intent", "Probability"])
                df_probs = df_probs.sort_values(by="Probability", ascending=True)
                st.bar_chart(df_probs.set_index("Intent"))

    # Historical Retrieval Grounding Section
    st.markdown("---")
    st.subheader("4. Retrieved Historical Apple Resolutions (Vector Store)")
    retrieved = result["retrieved_examples"]

    if retrieved:
        for i, ex in enumerate(retrieved, 1):
            with st.expander(f"Historical Match #{i} | Relevance: {ex['similarity_score']:.2f} | Category: {ex['intent']}"):
                st.markdown(f"**Past Customer Query:**\n*{ex['customer_message']}*")
                st.markdown(f"**Verified Apple Support Resolution:**\n{ex['brand_response']}")
    else:
        st.info("No historical matches exceeded the minimum similarity threshold.")

st.sidebar.markdown("---")
st.sidebar.caption("AI Support Agent for Apple Support • Kaggle Customer Support on Twitter Dataset")
