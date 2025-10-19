import io
import os
import joblib
import pandas as pd
import streamlit as st
from datetime import datetime
from src.schema import MANDATORY_FOR_APP
from src.utils import basic_clean_description, ensure_columns, coerce_types

st.set_page_config(page_title="Smart Expense Classifier", page_icon="💸", layout="centered")

MODEL_PATH = os.path.join("models", "expense_classifier.joblib")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model not found. Please run:  `python -m src.train`")
        st.stop()
    return joblib.load(MODEL_PATH)

def predict_categories(model, df: pd.DataFrame):
    work = df.copy()
    work["clean_text"] = work["description"].apply(basic_clean_description)
    preds = model.predict(work["clean_text"])
    work["predicted_category"] = preds
    return work

def example_template():
    return pd.DataFrame({
        "date": ["2025-10-01", "2025-10-02"],
        "amount": [250.0, 1499.0],
        "description": ["Zomato order 9999", "Myntra fashion sale"],
    })

st.title("💸 Smart Expense Classifier")
st.caption("Upload a CSV (`date, amount, description`) → auto-categorize → analyze → download.")

with st.expander("Need a sample CSV format?"):
    st.download_button(
        "Download example.csv",
        data=example_template().to_csv(index=False).encode("utf-8"),
        file_name="example.csv",
        mime="text/csv"
    )

uploaded = st.file_uploader("Upload your transactions CSV", type=["csv"])
model = load_model()

if uploaded:
    try:
        df = pd.read_csv(uploaded)
        df = ensure_columns(df, MANDATORY_FOR_APP)
        df = coerce_types(df)
        df = df.dropna(subset=["amount", "description"], how="any")
        if df.empty:
            st.warning("No valid rows after cleaning.")
        else:
            out = predict_categories(model, df)

            st.subheader("Preview")
            st.dataframe(out.head(20), use_container_width=True)

            st.subheader("Category split")
            st.bar_chart(out["predicted_category"].value_counts().sort_values(ascending=False))

            st.subheader("Spend by category")
            st.bar_chart(out.groupby("predicted_category")["amount"].sum().sort_values(ascending=False))

            buf = io.StringIO()
            out.to_csv(buf, index=False)
            st.download_button(
                "⬇️ Download categorized CSV",
                data=buf.getvalue().encode("utf-8"),
                file_name=f"categorized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Failed to process file: {e}")
else:
    st.info("Upload a CSV to get started.")
