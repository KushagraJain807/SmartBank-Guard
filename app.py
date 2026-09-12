from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SmartBank Guard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "fraud_model.joblib"
META_PATH = ROOT / "models" / "metadata.json"
REPORT_PATH = ROOT / "reports" / "model_comparison.csv"
TEST_PATH = ROOT / "reports" / "test_set.csv"

THRESHOLD_PATH = ROOT / "reports" / "threshold_analysis.csv"
CONFUSION_PATH = ROOT / "reports" / "confusion_matrix.png"
ROC_PATH = ROOT / "reports" / "roc_curve.png"
FEATURE_IMPORTANCE_PATH = ROOT / "reports" / "feature_importance.csv"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    /* Hero */
    .hero {
        padding: 2rem 2.2rem;
        border-radius: 20px;
        background: linear-gradient(
            135deg,
            rgba(30, 60, 114, 0.98),
            rgba(42, 82, 152, 0.98)
        );
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.20);
    }

    .hero h1 {
        color: white;
        font-size: 2.8rem;
        margin-bottom: 0.3rem;
        font-weight: 750;
    }

    .hero p {
        color: #dce7ff;
        font-size: 1.05rem;
        margin-bottom: 0;
    }

    /* Section titles */
    .section-title {
        font-size: 1.45rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 0.9rem;
    }

    /* Cards */
    .metric-card {
        padding: 1.2rem;
        border-radius: 15px;
        border: 1px solid rgba(128,128,128,0.25);
        background: rgba(128,128,128,0.06);
        text-align: center;
        min-height: 120px;
    }

    .metric-title {
        font-size: 0.9rem;
        opacity: 0.75;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin-top: 0.4rem;
    }

    /* Info cards */
    .info-card {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.20);
        background: rgba(128,128,128,0.05);
        margin-bottom: 0.8rem;
    }

    /* Result card */
    .result-card {
        padding: 1.3rem;
        border-radius: 15px;
        border: 1px solid rgba(128,128,128,0.25);
        background: rgba(128,128,128,0.05);
    }

    /* Footer */
    .footer {
        text-align: center;
        opacity: 0.6;
        font-size: 0.85rem;
        padding-top: 2rem;
        padding-bottom: 1rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128,128,128,0.15);
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
    }

    /* Progress */
    div[data-testid="stProgressBar"] {
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = [
    MODEL_PATH,
    META_PATH,
    REPORT_PATH,
    TEST_PATH
]

missing_files = [
    str(file.name)
    for file in required_files
    if not file.exists()
]

if missing_files:
    st.error(
        "The following required files are missing:\n\n"
        + "\n".join(f"- {file}" for file in missing_files)
    )
    st.stop()


# ============================================================
# LOAD MODEL + METADATA
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    with open(META_PATH, "r") as f:
        return json.load(f)


@st.cache_data
def load_reports():
    results = pd.read_csv(REPORT_PATH)
    test_data = pd.read_csv(TEST_PATH)
    return results, test_data


model = load_model()
metadata = load_metadata()
results, test_data = load_reports()

features = metadata["features"]
threshold = float(metadata.get("threshold", 0.50))
best_model = metadata.get("best_model", "XGBoost")

X_test = test_data.drop(columns=["Class"])
y_test = test_data["Class"]


# ============================================================
# SESSION STATE
# ============================================================

if "transaction" not in st.session_state:
    st.session_state.transaction = None

if "demo_type" not in st.session_state:
    st.session_state.demo_type = None

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🏦 SmartBank Guard</h1>
        <p>
            AI-Powered Transaction Fraud Detection & Risk Scoring
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write(
    """
    **SmartBank Guard** is an educational machine-learning prototype
    that analyzes transaction data and estimates the probability of
    fraudulent activity.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Transaction Analysis")

st.sidebar.markdown(
    """
    ### 🎯 Quick Demo

    Load a real transaction from the held-out test dataset
    and use it for an instant demonstration.
    """
)

demo_col1, demo_col2 = st.sidebar.columns(2)

with demo_col1:

    if st.button("✅ Legit", width="stretch"):

        legitimate = test_data[test_data["Class"] == 0]

        sample = legitimate.sample(
            n=1,
            random_state=None
        )

        st.session_state.transaction = (
            sample.drop(columns=["Class"])
            .iloc[0]
            .to_dict()
        )

        st.session_state.demo_type = "Legitimate"
        st.session_state.analysis_result = None


with demo_col2:

    if st.button("🚨 Fraud", width="stretch"):

        fraud = test_data[test_data["Class"] == 1]

        sample = fraud.sample(
            n=1,
            random_state=None
        )

        st.session_state.transaction = (
            sample.drop(columns=["Class"])
            .iloc[0]
            .to_dict()
        )

        st.session_state.demo_type = "Fraud"
        st.session_state.analysis_result = None


st.sidebar.divider()

st.sidebar.subheader("💳 Transaction Details")


# ============================================================
# TRANSACTION INPUT
# ============================================================

values = {}

for feature in features:

    if st.session_state.transaction is not None:

        default_value = float(
            st.session_state.transaction.get(feature, 0.0)
        )

    else:

        default_value = (
            100.0
            if feature.lower() == "amount"
            else 0.0
        )

    if feature.lower() == "amount":

        values[feature] = st.sidebar.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=default_value,
            step=10.0,
            format="%.2f"
        )

    else:

        values[feature] = st.sidebar.number_input(
            feature,
            value=default_value,
            format="%.6f"
        )


transaction = pd.DataFrame(
    [values],
    columns=features
)


# ============================================================
# MAIN ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-title">🔍 Transaction Risk Assessment</div>',
    unsafe_allow_html=True
)

if st.session_state.demo_type:

    st.info(
        f"🎯 **{st.session_state.demo_type} demo transaction loaded "
        "from the held-out test dataset.**"
    )

col1, col2 = st.columns([2, 1])

with col1:

    analyze = st.button(
        "🔎 Analyze Transaction",
        type="primary",
        width="stretch"
    )

with col2:

    st.metric(
        "Selected Model",
        best_model
    )


# ============================================================
# PREDICTION
# ============================================================

if analyze:

    probability = float(
        model.predict_proba(transaction)[0, 1]
    )

    prediction = int(
        probability >= threshold
    )

    percentage = probability * 100

    if percentage < 20:
        risk = "LOW"

    elif percentage < 50:
        risk = "MEDIUM"

    elif percentage < 80:
        risk = "HIGH"

    else:
        risk = "CRITICAL"

    decision = "FLAG" if prediction else "PASS"

    st.session_state.analysis_result = {
        "probability": probability,
        "percentage": percentage,
        "prediction": prediction,
        "risk": risk,
        "decision": decision
    }


# ============================================================
# DISPLAY ANALYSIS RESULT
# ============================================================

if st.session_state.analysis_result is not None:

    result = st.session_state.analysis_result

    st.markdown(
        '<div class="section-title">📊 Analysis Result</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "💰 Fraud Probability",
            f"{result['percentage']:.2f}%"
        )

    with c2:

        st.metric(
            "⚠️ Risk Level",
            result["risk"]
        )

    with c3:

        st.metric(
            "🎯 Model Decision",
            result["decision"]
        )

    st.markdown("### Fraud Probability")

    st.progress(
        min(result["probability"], 1.0)
    )

    st.caption(
        f"Prediction threshold: {threshold:.0%}"
    )

    if result["prediction"]:

        st.error(
            """
            🚨 **TRANSACTION FLAGGED**

            The model estimates elevated fraud risk.
            The transaction should be reviewed.
            """
        )

    else:

        st.success(
            """
            ✅ **TRANSACTION PASSED**

            The model currently classifies this transaction
            as likely legitimate.
            """
        )

    if result["percentage"] >= threshold * 100:

        st.info(
            f"""
            **Why was this transaction flagged?**

            The estimated fraud probability is
            **{result['percentage']:.2f}%**, which is above the
            configured **{threshold:.0%} decision threshold**.
            """
        )

    else:

        st.info(
            f"""
            **Why did this transaction pass?**

            The estimated fraud probability is
            **{result['percentage']:.2f}%**, which is below the
            configured **{threshold:.0%} decision threshold**.
            """
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">🤖 Model Performance</div>',
    unsafe_allow_html=True
)

if best_model in results["model"].values:

    row = results[
        results["model"] == best_model
    ].iloc[0]

    p1, p2, p3, p4 = st.columns(4)

    with p1:

        st.metric(
            "Precision",
            f"{row['precision']:.3f}"
        )

    with p2:

        st.metric(
            "Recall",
            f"{row['recall']:.3f}"
        )

    with p3:

        st.metric(
            "F1-Score",
            f"{row['f1']:.3f}"
        )

    with p4:

        st.metric(
            "ROC-AUC",
            f"{row['roc_auc']:.3f}"
        )

st.caption(
    "Metrics are calculated on the held-out test dataset."
)


# ============================================================
# MODEL COMPARISON
# ============================================================

st.markdown(
    '<div class="section-title">📈 Model Comparison</div>',
    unsafe_allow_html=True
)

display_results = results.copy()

numeric_columns = [
    "precision",
    "recall",
    "f1",
    "roc_auc"
]

for column in numeric_columns:

    display_results[column] = (
        display_results[column].round(4)
    )

st.dataframe(
    display_results,
    width="stretch",
    hide_index=True
)

chart_data = results.set_index("model")[
    ["precision", "recall", "f1", "roc_auc"]
]

st.bar_chart(chart_data)


# ============================================================
# DECISION THRESHOLD ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-title">⚖️ Decision Threshold Analysis</div>',
    unsafe_allow_html=True
)

if THRESHOLD_PATH.exists():

    threshold_data = pd.read_csv(
        THRESHOLD_PATH
    )

    st.write(
        """
        The decision threshold determines when a transaction is
        classified as fraudulent.

        Lower thresholds increase fraud detection sensitivity but
        may generate more false alerts, while higher thresholds
        improve precision at the cost of potentially missing
        fraudulent transactions.
        """
    )

    best_threshold_row = threshold_data.loc[
        threshold_data["f1"].idxmax()
    ]

    t1, t2, t3, t4 = st.columns(4)

    with t1:

        st.metric(
            "Selected Threshold",
            f"{best_threshold_row['threshold']:.0%}"
        )

    with t2:

        st.metric(
            "Precision",
            f"{best_threshold_row['precision']:.3f}"
        )

    with t3:

        st.metric(
            "Recall",
            f"{best_threshold_row['recall']:.3f}"
        )

    with t4:

        st.metric(
            "F1-Score",
            f"{best_threshold_row['f1']:.3f}"
        )

    st.markdown("### Precision vs Recall vs F1")

    threshold_chart = threshold_data.set_index(
        "threshold"
    )[[
        "precision",
        "recall",
        "f1"
    ]]

    st.line_chart(
        threshold_chart
    )

    st.success(
        f"""
        **Selected operating threshold:
        {best_threshold_row['threshold']:.0%}**

        This threshold produced the highest F1-score of
        **{best_threshold_row['f1']:.4f}** on the held-out test set.
        """
    )

else:

    st.info(
        "Run `python src/threshold_analysis.py` to generate "
        "threshold analysis results."
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

st.markdown(
    '<div class="section-title">🧩 Model Evaluation</div>',
    unsafe_allow_html=True
)

evaluation_col1, evaluation_col2 = st.columns(2)

with evaluation_col1:

    if CONFUSION_PATH.exists():

        st.image(
            str(CONFUSION_PATH),
            caption="Confusion Matrix",
            width="stretch"
        )

    else:

        st.info(
            "Confusion matrix image not found."
        )


with evaluation_col2:

    if ROC_PATH.exists():

        st.image(
            str(ROC_PATH),
            caption="ROC Curve",
            width="stretch"
        )

    else:

        st.info(
            "ROC curve image not found."
        )


# ============================================================
# MODEL EXPLAINABILITY
# ============================================================

st.markdown(
    '<div class="section-title">🧠 Model Explainability</div>',
    unsafe_allow_html=True
)

if FEATURE_IMPORTANCE_PATH.exists():

    feature_importance = pd.read_csv(
        FEATURE_IMPORTANCE_PATH
    )

    st.write(
        """
        Feature importance shows which input variables the
        XGBoost model relied on most when making predictions.

        Higher importance indicates greater contribution to the
        model's overall decision-making across the dataset.
        """
    )

    # --------------------------------------------------------
    # TOP FEATURES
    # --------------------------------------------------------

    top_features = feature_importance.head(10).copy()

    st.markdown("### Top 10 Important Features")

    st.dataframe(
        top_features,
        width="stretch",
        hide_index=True
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE CHART
    # --------------------------------------------------------

    st.markdown("### Feature Importance Ranking")

    feature_chart = (
        top_features
        .set_index("feature")["importance"]
        .sort_values()
    )

    st.bar_chart(
        feature_chart
    )

    # --------------------------------------------------------
    # KEY OBSERVATION
    # --------------------------------------------------------

    if len(feature_importance) >= 2:

        top_feature = feature_importance.iloc[0]
        second_feature = feature_importance.iloc[1]

        combined_importance = (
            top_feature["importance"]
            + second_feature["importance"]
        )

        st.info(
            f"""
            **Key observation:**

            **{top_feature['feature']}** is the most important
            feature with an importance of
            **{top_feature['importance']:.4f}**.

            **{second_feature['feature']}** is the second most
            important feature with an importance of
            **{second_feature['importance']:.4f}**.

            Together, these two features contribute approximately
            **{combined_importance:.1%}** of the model's total
            feature importance.
            """
        )

    st.caption(
        "Note: V1–V28 are anonymized transformed variables from "
        "the public credit-card fraud dataset, so their individual "
        "real-world meanings are not directly interpretable."
    )

else:

    st.info(
        "Run `python src/feature_importance.py` to generate "
        "feature importance results."
    )


# ============================================================
# DATASET INFORMATION
# ============================================================

with st.expander("🔬 About the Dataset"):

    total_transactions = len(test_data)

    fraud_transactions = int(
        test_data["Class"].sum()
    )

    legitimate_transactions = (
        total_transactions - fraud_transactions
    )

    d1, d2, d3 = st.columns(3)

    with d1:

        st.metric(
            "Test Transactions",
            f"{total_transactions:,}"
        )

    with d2:

        st.metric(
            "Legitimate",
            f"{legitimate_transactions:,}"
        )

    with d3:

        st.metric(
            "Fraud",
            f"{fraud_transactions:,}"
        )

    st.markdown(
        """
        The dataset contains anonymized transaction features.

        `V1`–`V28` are anonymized numerical variables. They do not
        directly represent human-readable attributes such as city,
        merchant name, card type or transaction category.

        `Class = 0` represents legitimate transactions.

        `Class = 1` represents fraudulent transactions.
        """
    )


# ============================================================
# PROJECT ARCHITECTURE
# ============================================================

with st.expander("🏗️ Project Architecture"):

    st.code(
        """
                 Transaction Data
                        │
                        ▼
                  Data Loading
                        │
                        ▼
                  Preprocessing
                        │
                        ▼
                 Feature Scaling
                        │
                        ▼
              ┌─────────────────────┐
              │   Machine Learning  │
              │                     │
              │ Logistic Regression │
              │ Random Forest       │
              │ XGBoost             │
              └──────────┬──────────┘
                         │
                         ▼
                Fraud Probability
                         │
                         ▼
                Risk Classification
                         │
                         ▼
                 Streamlit Dashboard
        """,
        language="text"
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.warning(
    """
    **Educational Prototype**

    SmartBank Guard is a student machine-learning project using
    a public anonymized dataset. It does not use HSBC customer data
    and should not be used for real financial decision-making.

    Production banking systems would require additional security,
    monitoring, explainability, governance and human-review controls.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        SmartBank Guard • AI/ML & FinTech Project<br>
        Python • Scikit-learn • XGBoost • Pandas • Streamlit • Git
    </div>
    """,
    unsafe_allow_html=True
)