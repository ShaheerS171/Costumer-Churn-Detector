import streamlit as st
import pandas as pd
import joblib
import os

# --- App Configuration ---
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="👋",
    layout="wide"
)

# --- Paths ---
# Ensures the app can find files regardless of where it's run from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data-set.csv")
MODEL_PATH = os.path.join(BASE_DIR, "logistic_churn_model.pkl")

# --- Load the pre-trained model ---
@st.cache_resource
def load_model():
    """Loads the saved logistic regression model from the file."""
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {MODEL_PATH}. Please ensure the model is saved correctly.")
        return None

model = load_model()

# --- Load raw dataset for UI options ---
@st.cache_data
def load_raw_data():
    """Loads the raw dataset to populate UI select boxes with original category names."""
    df_raw = pd.read_csv(DATA_PATH)
    # Perform minimal cleaning just for UI options
    df_raw['TotalCharges'] = pd.to_numeric(df_raw['TotalCharges'], errors='coerce')
    return df_raw

df_raw = load_raw_data()

# --- STREAMLIT APP LAYOUT ---
st.title("👋 Customer Churn Prediction App")
st.markdown("""
This application predicts whether a customer is likely to churn (cancel their subscription) based on their information and subscribed services. 
Customer churn is a critical metric for subscription-based businesses, and predicting it can help companies take proactive steps to retain customers.
""")

st.header("Methodology and Code Explanation")
with st.expander("Click here to see how this app was made"):
    st.markdown("""
    This app is built upon a standard data science workflow.
    
    **1. Offline Model Training (in a Jupyter Notebook):**
    - **Data Cleaning:** The dataset was cleaned by dropping `customerID` and handling missing values in `TotalCharges`.
    - **Preprocessing:** Categorical features (like 'Contract') were converted into a numerical format using **One-Hot Encoding** (`pd.get_dummies`). This creates separate binary (0/1) columns for each category.
    - **Model Selection:** A **Logistic Regression** classifier was trained.
    - **Saving the Model:** The trained model was saved to `logistic_churn_model.pkl` using `joblib`. This file contains both the trained logic and the list of feature names it expects, in the correct order.

    **2. Streamlit App (This Web App):**
    - **Loading the Model:** The app loads the pre-trained `logistic_churn_model.pkl` file.
    - **User Input:** Interactive widgets in the sidebar collect new customer data.
    - **Prediction:** When you click 'Predict', the app:
        1. Creates a DataFrame from your inputs.
        2. Applies **One-Hot Encoding** to the categorical inputs to match the model's training format.
        3. **Aligns the columns** of this new data with the exact list of features the model was trained on, filling any missing columns with 0.
        4. Feeds the correctly formatted data into the model to generate a prediction.
    """)

st.header("Churn Prediction")
st.sidebar.title("Customer Details")
st.sidebar.markdown("Enter the customer's details to get a churn prediction.")

# --- Prediction Form ---
if model is not None:
    with st.sidebar.form(key='prediction_form'):
        st.subheader("User Info")
        gender = st.selectbox("Gender", options=df_raw['gender'].unique())
        senior_citizen = st.selectbox("Senior Citizen", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        partner = st.selectbox("Partner", options=df_raw['Partner'].unique())
        dependents = st.selectbox("Dependents", options=df_raw['Dependents'].unique())

        st.subheader("Account Information")
        tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=24)
        contract = st.selectbox("Contract", options=df_raw['Contract'].unique())
        paperless_billing = st.selectbox("Paperless Billing", options=df_raw['PaperlessBilling'].unique())
        payment_method = st.selectbox("Payment Method", options=df_raw['PaymentMethod'].unique())
        monthly_charges = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=55.0, step=0.05)
        total_charges = st.slider("Total Charges ($)", min_value=18.0, max_value=9000.0, value=1500.0, step=0.05)

        st.subheader("Subscribed Services")
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

    if submit_button:
        # Create a dictionary of the inputs
        input_dict = {
            'gender': gender, 'SeniorCitizen': senior_citizen, 'Partner': partner, 'Dependents': dependents,
            'tenure': tenure, 'PhoneService': phone_service, 'MultipleLines': multiple_lines,
            'InternetService': internet_service, 'OnlineSecurity': online_security, 'OnlineBackup': online_backup,
            'DeviceProtection': device_protection, 'TechSupport': tech_support, 'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies, 'Contract': contract, 'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method, 'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
        }
        
        # Create a DataFrame from the dictionary
        input_df = pd.DataFrame([input_dict])

        # --- FIX: Convert categorical columns to one-hot encoding ---
        # Identify categorical and numerical columns
        categorical_cols = [col for col in input_df.columns if input_df[col].dtype == 'object']
        numerical_cols = [col for col in input_df.columns if input_df[col].dtype != 'object']

        # Apply one-hot encoding
        input_df_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=False)

        # Get the feature list the model was trained on
        model_features = model.feature_names_in_

        # Align the input DataFrame with the model's feature list
        # This adds missing columns (and fills them with 0) and ensures the correct order
        input_data_aligned = input_df_encoded.reindex(columns=model_features, fill_value=0)
        
        try:
            # Make prediction
            prediction = model.predict(input_data_aligned)[0]
            prediction_proba = model.predict_proba(input_data_aligned)[0]

            st.subheader("Prediction Result")
            if prediction == 1:
                st.error("This customer is likely to **Churn**.")
                st.write(f"Confidence: **{prediction_proba[1]*100:.2f}%**")
            else:
                st.success("This customer is likely to **Stay**.")
                st.write(f"Confidence: **{prediction_proba[0]*100:.2f}%**")
            
            st.progress(max(prediction_proba))
        
        except Exception as e:
            st.error("An error occurred during prediction.")
            st.error(f"Details: {e}")

else:
    st.warning("Model could not be loaded. Prediction is unavailable.")
