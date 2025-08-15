import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- App Configuration ---
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="👋",
    layout="wide"
)

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data-set.csv")
MODEL_PATH = os.path.join(BASE_DIR, "logistic_churn_model.pkl")

# --- Load the pre-trained model ---
@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error(f"Model file not found at `{MODEL_PATH}`. Upload it and restart the app.")
        return None

model = load_model()

# --- Load raw dataset ---
@st.cache_data
def load_raw_data():
    try:
        df = pd.read_csv(DATA_PATH)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        return df
    except FileNotFoundError:
        st.error(f"Dataset file not found at `{DATA_PATH}`. Upload it and restart the app.")
        return pd.DataFrame()

df_raw = load_raw_data()

# --- Title & Description ---
st.title("👋 Customer Churn Prediction App")
st.markdown("""
This app predicts whether a customer is likely to **churn** (cancel subscription) 
based on their information and services.
""")

# --- Calculate model performance metrics ---
@st.cache_data
def get_model_metrics(df):
    """Only cache metrics calculation — model loaded separately to avoid unhashable object error."""
    if df.empty:
        return None

    # Load model inside function to avoid passing unhashable object
    local_model = joblib.load(MODEL_PATH)

    df_clean = df.dropna()
    X = df_clean.drop("Churn", axis=1)
    y = df_clean["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

    categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
    X_aligned = X_encoded.reindex(columns=local_model.feature_names_in_, fill_value=0)

    y_pred = local_model.predict(X_aligned)

    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred)
    }

metrics = get_model_metrics(df_raw)

# --- Sidebar: Model Info ---
with st.sidebar:
    st.markdown("### 📊 Model Info")
    if metrics:
        st.write(f"**Accuracy:** {metrics['accuracy']*100:.2f}%")
        st.write(f"**Precision:** {metrics['precision']*100:.2f}%")
        st.write(f"**Recall:** {metrics['recall']*100:.2f}%")
        st.write(f"**F1-Score:** {metrics['f1']*100:.2f}%")
    else:
        st.write("Metrics unavailable")

    if st.button("Load Sample Data"):
        st.session_state.sample_loaded = True
    else:
        st.session_state.sample_loaded = False

# --- Prediction Form ---
if model is not None and not df_raw.empty:
    with st.sidebar.form(key='prediction_form'):
        st.subheader("Customer Details")

        # Customer Info
        gender = st.selectbox("Gender", options=df_raw['gender'].unique())
        senior_citizen = st.selectbox("Senior Citizen", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        partner = st.selectbox("Partner", options=df_raw['Partner'].unique())
        dependents = st.selectbox("Dependents", options=df_raw['Dependents'].unique())

        # Account Info
        tenure = st.slider("Tenure (months)", 0, 72, 24)
        contract = st.selectbox("Contract", options=df_raw['Contract'].unique())
        paperless_billing = st.selectbox("Paperless Billing", options=df_raw['PaperlessBilling'].unique())
        payment_method = st.selectbox("Payment Method", options=df_raw['PaymentMethod'].unique())
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 55.0, 0.05)
        total_charges = st.slider("Total Charges ($)", 18.0, 9000.0, 1500.0, 0.05)

        # Services
        phone_service = st.selectbox("Phone Service", options=df_raw['PhoneService'].unique())
        multiple_lines = st.selectbox("Multiple Lines", options=df_raw['MultipleLines'].unique())
        internet_service = st.selectbox("Internet Service", options=df_raw['InternetService'].unique())
        online_security = st.selectbox("Online Security", options=df_raw['OnlineSecurity'].unique())
        online_backup = st.selectbox("Online Backup", options=df_raw['OnlineBackup'].unique())
        device_protection = st.selectbox("Device Protection", options=df_raw['DeviceProtection'].unique())
        tech_support = st.selectbox("Tech Support", options=df_raw['TechSupport'].unique())
        streaming_tv = st.selectbox("Streaming TV", options=df_raw['StreamingTV'].unique())
        streaming_movies = st.selectbox("Streaming Movies", options=df_raw['StreamingMovies'].unique())

        submit_button = st.form_submit_button(label='Predict Churn')

    # --- Prediction Logic ---
    if submit_button:
        input_dict = {
            'gender': gender, 'SeniorCitizen': senior_citizen, 'Partner': partner, 'Dependents': dependents,
            'tenure': tenure, 'PhoneService': phone_service, 'MultipleLines': multiple_lines,
            'InternetService': internet_service, 'OnlineSecurity': online_security, 'OnlineBackup': online_backup,
            'DeviceProtection': device_protection, 'TechSupport': tech_support, 'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies, 'Contract': contract, 'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method, 'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
        }

        input_df = pd.DataFrame([input_dict])
        categorical_cols = [col for col in input_df.columns if input_df[col].dtype == 'object']
        input_df_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=False)
        input_data_aligned = input_df_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

        try:
            prediction = model.predict(input_data_aligned)[0]
            prediction_proba = model.predict_proba(input_data_aligned)[0]

            st.subheader("Prediction Result")
            if prediction == 1:
                st.error(f"🚨 This customer is likely to churn.\nConfidence: {prediction_proba[1]*100:.2f}%")
            else:
                st.success(f"✅ This customer is likely to stay.\nConfidence: {prediction_proba[0]*100:.2f}%")

            # Probability bar
            st.markdown("### Probability")
            st.progress(prediction_proba[1] if prediction == 1 else prediction_proba[0])

        except Exception as e:
            st.error(f"Prediction error: {e}")
else:
    st.warning("Model or dataset not loaded. Please upload the files and restart the app.")
