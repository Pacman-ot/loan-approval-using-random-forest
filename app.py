from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "loan_approval_rf_pipeline.joblib"


@st.cache_resource
def load_model(model_path: str):
    if not Path(model_path).exists():
        return None
    return joblib.load(model_path)


def predict_loan(model, customer_data: dict):
    sample = pd.DataFrame([customer_data])
    pred_class = int(model.predict(sample)[0])
    pred_prob = float(model.predict_proba(sample)[0, 1])
    decision = "Approved" if pred_class == 1 else "Rejected"
    return decision, pred_prob


def get_feature_importances(model):
    try:
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]

        # Prefer sklearn-generated transformed names to avoid relying on transformer order.
        all_features = preprocessor.get_feature_names_out().tolist()

        imp = classifier.feature_importances_
        imp_df = pd.DataFrame({"feature": all_features, "importance": imp})
        return imp_df.sort_values("importance", ascending=False)
    except Exception:
        return None


st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
)

st.title("Intelligent Loan Approval Prediction")
st.caption("Random Forest model for fast and consistent loan decision support")

model = load_model(MODEL_PATH)

if model is None:
    st.error(
        f"Model file not found: {MODEL_PATH.name}. "
        "Run the notebook to train and save the model first."
    )
    st.stop()

col1, col2 = st.columns([1.2, 1.0])

with col1:
    st.subheader("Applicant Details")

    age = st.slider("Age", min_value=18, max_value=100, value=35)
    income = st.number_input("Annual Income", min_value=0.0, value=78000.0, step=1000.0)
    credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=720)
    loan_amount = st.number_input("Loan Amount", min_value=1000.0, value=150000.0, step=5000.0)
    loan_term = st.selectbox("Loan Term (months)", options=[12, 24, 36, 48, 60, 72, 84], index=4)
    employment_years = st.slider("Employment Years", min_value=0, max_value=60, value=8)
    existing_debt = st.number_input("Existing Debt", min_value=0.0, value=12000.0, step=1000.0)
    employment_type = st.selectbox(
        "Employment Type",
        options=["salaried", "self_employed", "contract", "unemployed"],
        index=0,
    )

    threshold = st.slider("Approval Threshold", min_value=0.0, max_value=1.0, value=0.50, step=0.01)

    submitted = st.button("Predict Loan Decision", type="primary")

with col2:
    st.subheader("Prediction Result")

    if submitted:
        customer = {
            "age": age,
            "income": income,
            "credit_score": credit_score,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "employment_years": employment_years,
            "existing_debt": existing_debt,
            "employment_type": employment_type,
        }

        try:
            _, approval_prob = predict_loan(model, customer)
        except Exception as ex:
            st.error("Prediction failed. Verify the saved model matches the expected feature schema.")
            st.exception(ex)
            st.stop()

        decision = "Approved" if approval_prob >= threshold else "Rejected"

        if decision == "Approved":
            st.success(f"Decision: {decision}")
        else:
            st.error(f"Decision: {decision}")

        st.metric("Approval Probability", f"{approval_prob * 100:.2f}%")
        st.progress(float(approval_prob))

        st.write("Model-based probability compared against your selected threshold.")
        st.write(f"Threshold: {threshold:.2f}")

    else:
        st.info("Enter applicant data and click Predict Loan Decision.")

st.divider()
st.subheader("Model Insight: Feature Importance")
importance_df = get_feature_importances(model)

if importance_df is not None:
    st.dataframe(importance_df.head(10), use_container_width=True)
    st.bar_chart(importance_df.head(10).set_index("feature"))
else:
    st.warning("Feature importance is not available for this model pipeline.")

st.caption("Note: This tool is for decision support and should complement policy and compliance checks.")
