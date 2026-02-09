import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Decision Fatigue Predictor",
    page_icon="🧠",
    layout="centered",
)

# ── Load model & encoders ───────────────────────────────────────────────────
MODEL_DIR = os.path.dirname(__file__)
checkpoint_path = os.path.join(MODEL_DIR, "fatigue_behavioral.joblib")
best_model_path = os.path.join(MODEL_DIR, "fatigue_behavioral_best_model.joblib")


@st.cache_resource
def load_model():
    """Load the saved model checkpoint."""
    if os.path.exists(checkpoint_path):
        checkpoint = joblib.load(checkpoint_path)
        return checkpoint["model"]
    elif os.path.exists(best_model_path):
        return joblib.load(best_model_path)
    else:
        st.error("No model file found! Train the model first.")
        st.stop()


model = load_model()

# ── Encoding maps (match LabelEncoder alphabetical order) ───────────────────
TIME_OF_DAY_MAP = {"Afternoon": 0, "Evening": 1, "Morning": 2, "Night": 3}
FATIGUE_LEVEL_MAP = {"High": 0, "Low": 1, "Moderate": 2}
RECOMMENDATION_MAP = {0: "✅ Continue", 1: "🐢 Slow Down", 2: "🛑 Take Break"}

# ── App header ──────────────────────────────────────────────────────────────
st.title("🧠 Decision Fatigue Predictor")
st.markdown(
    "Enter your current state and the app will predict whether you should "
    "**Continue**, **Slow Down**, or **Take a Break**."
)
st.divider()

# ── Sidebar – quick info ────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        "This app uses a **KNN classifier** trained on the "
        "[Human Decision Fatigue Behavioral Dataset]"
        "(https://www.kaggle.com/datasets/sonalshinde123/human-decision-fatigue-behavioral-dataset) "
        "to recommend whether you should continue working, slow down, or take a break."
    )
    st.divider()
    st.markdown("**Model:** KNeighborsClassifier (tuned via RandomizedSearchCV)")

# ── Input form ──────────────────────────────────────────────────────────────
with st.form("prediction_form"):
    st.subheader("📝 Your Current State")

    col1, col2 = st.columns(2)

    with col1:
        hours_awake = st.number_input(
            "Hours Awake", min_value=0, max_value=30, value=8, step=1,
            help="How many hours have you been awake?"
        )
        decisions_made = st.number_input(
            "Decisions Made", min_value=0, max_value=200, value=30, step=1,
            help="Total number of decisions made today."
        )
        task_switches = st.number_input(
            "Task Switches", min_value=0, max_value=50, value=10, step=1,
            help="Number of times you switched between tasks."
        )
        avg_decision_time = st.number_input(
            "Avg Decision Time (sec)", min_value=0.0, max_value=15.0,
            value=2.5, step=0.1, format="%.2f",
            help="Average time you take to make a decision."
        )
        sleep_hours = st.number_input(
            "Sleep Hours Last Night", min_value=0.0, max_value=14.0,
            value=7.0, step=0.1, format="%.1f",
            help="How many hours did you sleep last night?"
        )
        time_of_day = st.selectbox(
            "Time of Day",
            options=list(TIME_OF_DAY_MAP.keys()),
            index=1,
            help="Current time of day."
        )

    with col2:
        caffeine_cups = st.number_input(
            "Caffeine Intake (Cups)", min_value=0, max_value=10, value=1, step=1,
            help="Number of cups of coffee / caffeinated drinks today."
        )
        stress_level = st.slider(
            "Stress Level (1-10)", min_value=1.0, max_value=10.0,
            value=3.0, step=0.1,
            help="Your current stress level on a 1-10 scale."
        )
        error_rate = st.number_input(
            "Error Rate", min_value=0.0, max_value=1.0,
            value=0.0, step=0.01, format="%.3f",
            help="Proportion of errors in recent decisions (0 = none, 1 = all)."
        )
        cognitive_load = st.number_input(
            "Cognitive Load Score", min_value=0.0, max_value=10.0,
            value=3.0, step=0.1, format="%.1f",
            help="Current cognitive load score."
        )
        decision_fatigue_score = st.slider(
            "Decision Fatigue Score", min_value=0.0, max_value=100.0,
            value=30.0, step=0.1,
            help="Your self-assessed decision fatigue (0-100)."
        )
        fatigue_level = st.selectbox(
            "Fatigue Level",
            options=list(FATIGUE_LEVEL_MAP.keys()),
            index=1,
            help="Your overall fatigue level."
        )

    submitted = st.form_submit_button(
        "🔮 Predict Recommendation", use_container_width=True
    )

# ── Prediction ──────────────────────────────────────────────────────────────
if submitted:
    # Build feature array in the same column order as training
    features = np.array([[
        hours_awake,
        decisions_made,
        task_switches,
        avg_decision_time,
        sleep_hours,
        TIME_OF_DAY_MAP[time_of_day],
        caffeine_cups,
        stress_level,
        error_rate,
        cognitive_load,
        decision_fatigue_score,
        FATIGUE_LEVEL_MAP[fatigue_level],
    ]])

    prediction = model.predict(features)[0]
    label = RECOMMENDATION_MAP.get(prediction, f"Unknown ({prediction})")

    st.divider()
    st.subheader("🎯 Recommendation")

    # Show result with colored styling
    if prediction == 0:
        st.success(f"**{label}** — You're doing fine. Keep going!")
    elif prediction == 1:
        st.warning(f"**{label}** — Consider reducing your pace.")
    else:
        st.error(f"**{label}** — You need a break. Step away and recharge!")

    # Show input summary
    with st.expander("📊 Input Summary"):
        input_df = pd.DataFrame({
            "Feature": [
                "Hours Awake", "Decisions Made", "Task Switches",
                "Avg Decision Time (sec)", "Sleep Hours Last Night",
                "Time of Day", "Caffeine Intake (Cups)", "Stress Level",
                "Error Rate", "Cognitive Load Score",
                "Decision Fatigue Score", "Fatigue Level",
            ],
            "Value": [
                hours_awake, decisions_made, task_switches,
                avg_decision_time, sleep_hours,
                time_of_day, caffeine_cups, stress_level,
                error_rate, cognitive_load,
                decision_fatigue_score, fatigue_level,
            ],
        })
        st.dataframe(input_df, use_container_width=True, hide_index=True)