
import streamlit as st
import pandas as pd
import joblib
import os
from scipy.io import arff

# =====================================
# LOAD TRAINED MODEL AND DATASET
# =====================================

MODEL_PATH = "model/best_phishing_model.joblib"
DATASET_PATH = "dataset/Training Dataset.arff"

st.set_page_config(
    page_title="Phishing Website Detection",
    page_icon="🔒",
    layout="wide"
)

st.title("Phishing Website Detection System")

st.write(
    "Enter website feature values to predict "
    "whether a website is phishing or legitimate."
)

if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found. Run train_model.py first.")
    st.stop()

if not os.path.exists(DATASET_PATH):
    st.error("Dataset file not found.")
    st.stop()

model = joblib.load(MODEL_PATH)

# Load dataset to get exact feature names and values
data, meta = arff.loadarff(DATASET_PATH)

df = pd.DataFrame(data)

for col in df.columns:
    df[col] = df[col].apply(
        lambda x: x.decode("utf-8")
        if isinstance(x, bytes) else x
    )

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

feature_names = [
    col for col in df.columns
    if col != "Result"
]

# =====================================
# USER INPUT FORM
# =====================================

st.subheader("Enter Website Features")

st.info(
    "Select the feature values using the dataset's "
    "available values."
)

with st.form("prediction_form"):

    user_input = {}

    columns = st.columns(3)

    for i, feature in enumerate(feature_names):

        available_values = sorted(
            df[feature].dropna().unique().tolist()
        )

        with columns[i % 3]:

            user_input[feature] = st.selectbox(
                feature,
                options=available_values,
                key=feature
            )

    submitted = st.form_submit_button(
        "Predict Website"
    )

# =====================================
# PREDICTION
# =====================================

if submitted:

    input_df = pd.DataFrame(
        [user_input],
        columns=feature_names
    )

    prediction = model.predict(input_df)[0]

    st.subheader("Prediction Result")

    if prediction == -1:

        st.error("Potentially Phishing Website")

        st.warning(
            "The entered website features indicate "
            "a potentially phishing website."
        )

    elif prediction == 1:

        st.success("Likely Legitimate Website")

        st.info(
            "The entered website features indicate "
            "a likely legitimate website."
        )

    else:

        st.write("Unknown Prediction Result")

# =====================================
# END
# =====================================

