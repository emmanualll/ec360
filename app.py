import streamlit as st
import pandas as pd
import engine  # Imports functions from engine.py
import traceback

# --- Page Configuration ---
st.set_page_config(
    page_title="EC-360 Dashboard",
    page_icon="🧭",
    layout="wide"
)

# --- Title ---
st.title("🧭 EC-360 — The Empathy Layer of AI")

# --- Instructions ---
st.subheader("How to Run")
st.code("streamlit run app.py --server.port=3000 --server.address=0.0.0.0", language="bash")
st.markdown("---")

# --- Helper Function for Explanation ---
def format_explanation(exp_list):
    """Formats the LIME explanation list into readable markdown."""
    explanation_md = ""
    for feature, weight in exp_list:
        color = "green" if weight > 0 else "red"
        explanation_md += f"- **{feature}**: <span style='color:{color};'>{weight:.4f}</span><br>"
    return explanation_md

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Controls")
    sensitive_attribute = st.selectbox(
        "Choose a sensitive attribute:",
        options=["sex", "race"],
        index=0
    )
    run_button = st.button("Run Fairness Check")

# --- Main Dashboard ---
if run_button:
    try:
        with st.spinner("Running analysis... This may take a moment."):
            # 1. Load Data
            df = engine.load_data()

            # 2. Preprocess
            X, y, sensitive = engine.preprocess(df, sensitive_col=sensitive_attribute)

            # 3. Train and Evaluate
            model, X_train, X_test, y_test, sp, eo, trust = engine.train_and_evaluate(
                X, y, sensitive
            )

        st.success("Fairness check complete!")

        # --- Display Metrics ---
        st.header("Fairness & Trust Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Statistical Parity Difference", f"{sp:.4f}")
        col2.metric("Equalized Odds Difference", f"{eo:.4f}")
        col3.metric("Trust Score", f"{trust:.2f}/100")

        st.progress(int(trust))
        st.markdown(
            """
        - **Statistical Parity (SP)**: Measures the difference in the probability of a favorable outcome (e.g., high income) between different groups. A value near 0 is ideal.
        - **Equalized Odds (EO)**: Measures if the model predicts positive outcomes (True Positives) and negative outcomes (True Negatives) at equal rates for both groups. A value near 0 is ideal.
        - **Trust Score**: Our custom metric (100 - (abs(SP) + abs(EO)) * 50), penalizing deviations from 0 in both fairness metrics.
        """
        )
        st.markdown("---")

        # --- Display Explanation ---
        st.header("Local Prediction Explanation (LIME)")
        
        # Get a random instance from the test set
        instance_index = X_test.sample(1).index
        instance_df = X_test.loc[instance_index]
        true_label = y_test.loc[instance_index].values[0]
        pred_label = model.predict(instance_df)[0]
        
        # Get explanation
        explanation_list = engine.explain(model, X_train, instance_df)
        
        st.subheader("Explanation for a Random Test Case")
        
        # Display instance info
        st.markdown(f"**True Income:** `{'<=50K' if true_label == 0 else '>50K'}`")
        st.markdown(f"**Predicted Income:** `{'<=50K' if pred_label == 0 else '>50K'}`")
        
        # Display formatted explanation
        st.markdown(
            f"""
            <div style="border: 1px solid #ddd; border-radius: 5px; padding: 15px; background-color: #f9f9f9;">
            <p>Features contributing to the prediction (Green = For, Red = Against '>50K'):</p>
            {format_explanation(explanation_list)}
            </div>
            """,
            unsafe_allow_html=True
        )
        
        with st.expander("View Raw Instance Data"):
            st.dataframe(instance_df)


    except FileNotFoundError:
        st.error(f"Error: Data file not found. Make sure '{engine.DATA_PATH}' exists.")
    except Exception as e:
        st.error("An error occurred during the analysis.")
        st.exception(e)
else:
    st.info("Select a sensitive attribute and click 'Run Fairness Check' to start the analysis.")