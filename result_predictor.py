import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Result Prediction",
    page_icon="🎓",
    layout="wide"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #2a2f45);
        border: 1px solid #3a3f5c;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px;
    }
    .metric-card h2 { color: #7c83fd; font-size: 2rem; margin: 0; }
    .metric-card p  { color: #a0a5c0; font-size: 0.85rem; margin: 4px 0 0; }
    .result-pass {
        background: linear-gradient(135deg, #0d3320, #145a32);
        border: 2px solid #27ae60;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
    }
    .result-fail {
        background: linear-gradient(135deg, #3b0d0d, #5a1414);
        border: 2px solid #e74c3c;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
    }
    .result-pass h1, .result-fail h1 { font-size: 3rem; margin: 0; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; padding: 10px 24px; }
</style>
""", unsafe_allow_html=True)

# ─── Data Generation ─────────────────────────────────────────────────────────
@st.cache_data
def generate_data(n=500, seed=42):
    np.random.seed(seed)
    hours       = np.random.uniform(1, 12, n)
    attendance  = np.random.uniform(40, 100, n)
    prev_marks  = np.random.uniform(30, 100, n)
    assignments = np.random.randint(0, 11, n)
    sleep       = np.random.uniform(4, 10, n)

    # Realistic marks formula
    marks = (
        hours       * 3.5 +
        attendance  * 0.3 +
        prev_marks  * 0.25 +
        assignments * 1.5 +
        sleep       * 0.5 +
        np.random.normal(0, 5, n)
    )
    marks = np.clip(marks, 0, 100)
    result = (marks >= 50).astype(int)

    df = pd.DataFrame({
        "Hours_Study":      hours,
        "Attendance":       attendance,
        "Previous_Marks":   prev_marks,
        "Assignments_Done": assignments,
        "Sleep_Hours":      sleep,
        "Marks":            marks,
        "Result":           result
    })
    return df

# ─── Model Training ──────────────────────────────────────────────────────────
@st.cache_resource
def train_model(df):
    features = ["Hours_Study", "Attendance", "Previous_Marks",
                "Assignments_Done", "Sleep_Hours"]
    X = df[features].values
    y = df["Result"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)
    metrics = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "F1 Score":  f1_score(y_test, y_pred)
    }
    cm = confusion_matrix(y_test, y_pred)
    return model, scaler, metrics, cm, y_test, y_pred

# ─── Load Data & Model ───────────────────────────────────────────────────────
df    = generate_data()
model, scaler, metrics, cm, y_test, y_pred = train_model(df)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("# 🎓 Student Result Prediction System")
st.markdown("*Machine Learning Web Application — VaultSphere AI Technologies Pvt. Ltd.*")
st.divider()

# ─── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    ["🎯 Predict Result", "📊 Data Analysis", "📈 Model Performance"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT RESULT
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Enter Student Details")
    st.markdown("Adjust the sliders below and click **Predict** to get the result.")

    col1, col2 = st.columns(2)
    with col1:
        hours       = st.slider("📚 Study Hours (per day)",     1.0, 12.0, 6.0, 0.5)
        attendance  = st.slider("🏫 Attendance (%)",           40.0, 100.0, 75.0, 1.0)
        prev_marks  = st.slider("📝 Previous Exam Marks",      30.0, 100.0, 60.0, 1.0)
    with col2:
        assignments = st.slider("📋 Assignments Completed",    0, 10, 5)
        sleep       = st.slider("😴 Sleep Hours (per day)",    4.0, 10.0, 7.0, 0.5)

    st.markdown("")
    if st.button("🔮 Predict Result", use_container_width=True, type="primary"):
        input_arr  = np.array([[hours, attendance, prev_marks, assignments, sleep]])
        input_sc   = scaler.transform(input_arr)
        prediction = model.predict(input_sc)[0]
        prob       = model.predict_proba(input_sc)[0]

        st.markdown("")
        if prediction == 1:
            confidence = prob[1] * 100
            st.markdown(f"""
            <div class="result-pass">
                <h1>✅ PASS</h1>
                <h3 style="color:#2ecc71; margin:8px 0;">Confidence: {confidence:.1f}%</h3>
                <p style="color:#a0c8a0;">The student is likely to pass the examination.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            confidence = prob[0] * 100
            st.markdown(f"""
            <div class="result-fail">
                <h1>❌ FAIL</h1>
                <h3 style="color:#e74c3c; margin:8px 0;">Confidence: {confidence:.1f}%</h3>
                <p style="color:#c8a0a0;">The student is at risk of failing. Intervention recommended.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")
        st.markdown("#### 📋 Input Summary")
        summary = pd.DataFrame({
            "Parameter":    ["Study Hours", "Attendance (%)", "Previous Marks",
                             "Assignments Done", "Sleep Hours"],
            "Value":        [hours, f"{attendance}%", prev_marks, assignments, sleep],
            "Contribution": ["High", "Medium", "High", "Medium", "Low"]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATA ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Dataset Insights")

    # Summary stats
    total    = len(df)
    passed   = int(df["Result"].sum())
    failed   = total - passed
    pass_rate = passed / total * 100

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, color in zip(
        [c1, c2, c3, c4],
        ["Total Students", "Passed", "Failed", "Pass Rate"],
        [total, passed, failed, f"{pass_rate:.1f}%"],
        ["#7c83fd", "#27ae60", "#e74c3c", "#f39c12"]
    ):
        col.markdown(f"""
        <div class="metric-card">
            <h2 style="color:{color};">{value}</h2>
            <p>{label}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.patch.set_facecolor("#0f1117")
    palette = {0: "#e74c3c", 1: "#27ae60"}
    colors  = df["Result"].map(palette)

    # 1. Study Hours vs Marks
    ax = axes[0, 0]
    ax.set_facecolor("#1e2130")
    ax.scatter(df["Hours_Study"], df["Marks"], c=colors, alpha=0.6, s=20, edgecolors="none")
    ax.set_xlabel("Study Hours", color="white")
    ax.set_ylabel("Marks", color="white")
    ax.set_title("Study Hours vs Marks", color="white", fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_edgecolor("#3a3f5c")
    ax.legend(handles=[
        plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='#27ae60', label='Pass'),
        plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='#e74c3c', label='Fail')
    ], facecolor="#1e2130", labelcolor="white")

    # 2. Attendance vs Marks
    ax = axes[0, 1]
    ax.set_facecolor("#1e2130")
    ax.scatter(df["Attendance"], df["Marks"], c=colors, alpha=0.6, s=20, edgecolors="none")
    ax.set_xlabel("Attendance (%)", color="white")
    ax.set_ylabel("Marks", color="white")
    ax.set_title("Attendance vs Marks", color="white", fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_edgecolor("#3a3f5c")

    # 3. Pass/Fail Pie Chart
    ax = axes[1, 0]
    ax.set_facecolor("#1e2130")
    ax.pie(
        [passed, failed],
        labels=["Pass", "Fail"],
        colors=["#27ae60", "#e74c3c"],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"color": "white"}
    )
    ax.set_title("Pass / Fail Distribution", color="white", fontweight="bold")

    # 4. Marks Distribution Histogram
    ax = axes[1, 1]
    ax.set_facecolor("#1e2130")
    ax.hist(df["Marks"], bins=25, color="#7c83fd", edgecolor="#0f1117", alpha=0.85)
    ax.axvline(50, color="#f39c12", linestyle="--", linewidth=2, label="Pass Mark (50)")
    ax.set_xlabel("Marks", color="white")
    ax.set_ylabel("Number of Students", color="white")
    ax.set_title("Marks Distribution", color="white", fontweight="bold")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1e2130", labelcolor="white")
    for spine in ax.spines.values(): spine.set_edgecolor("#3a3f5c")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Model Performance Metrics")

    metric_descriptions = {
        "Accuracy":  "% of students correctly classified overall",
        "Precision": "When model says PASS, how often it is right",
        "Recall":    "Of all students who passed, how many were caught",
        "F1 Score":  "Harmonic mean of Precision and Recall"
    }

    cols = st.columns(4)
    for col, (name, val) in zip(cols, metrics.items()):
        col.markdown(f"""
        <div class="metric-card">
            <h2>{val*100:.1f}%</h2>
            <p><b>{name}</b></p>
            <p style="font-size:0.75rem;">{metric_descriptions[name]}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    col_cm, col_fi = st.columns(2)

    # Confusion Matrix
    with col_cm:
        st.markdown("#### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#1e2130")
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fail", "Pass"],
            yticklabels=["Fail", "Pass"],
            ax=ax, linewidths=0.5,
            annot_kws={"size": 14, "color": "white"}
        )
        ax.set_xlabel("Predicted", color="white")
        ax.set_ylabel("Actual", color="white")
        ax.tick_params(colors="white")
        ax.set_title("Confusion Matrix", color="white", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Feature Importance
    with col_fi:
        st.markdown("#### Feature Importance")
        features = ["Study Hours", "Attendance", "Prev Marks",
                    "Assignments", "Sleep Hours"]
        importance = np.abs(model.coef_[0])
        importance = importance / importance.sum() * 100

        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#1e2130")
        bar_colors = ["#7c83fd", "#27ae60", "#f39c12", "#e74c3c", "#1abc9c"]
        bars = ax.barh(features, importance, color=bar_colors, edgecolor="none")
        ax.set_xlabel("Importance (%)", color="white")
        ax.set_title("Feature Importance (Coef. Magnitude)", color="white", fontweight="bold")
        ax.tick_params(colors="white")
        for spine in ax.spines.values(): spine.set_edgecolor("#3a3f5c")
        for bar, val in zip(bars, importance):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}%", va="center", color="white", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Metric explanations
    st.markdown("")
    st.markdown("#### 📖 What These Metrics Mean")
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        st.info("**Accuracy** — Out of all 100 test students, how many did the model get right? An accuracy of 90% means 90 correct predictions out of 100.")
        st.info("**Precision** — When the model predicts PASS, how trustworthy is that prediction? High precision = fewer false alarms.")
    with exp_col2:
        st.info("**Recall** — Of all students who actually passed, how many did the model correctly identify? High recall = fewer missed students.")
        st.info("**F1 Score** — A single balanced score combining Precision and Recall. Best used when classes are slightly imbalanced.")

# ─── Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center; color:#555; font-size:0.8rem;'>"
    "Developed by Kambala Neeha Kiran | VaultSphere AI Technologies Pvt. Ltd. | June 2026"
    "</p>",
    unsafe_allow_html=True
)
