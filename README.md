# 📊 Customer Churn Prediction App  

An interactive **Streamlit** web application that predicts whether a customer is likely to churn based on their demographics, account information, and subscribed services.  

This project uses a **Logistic Regression** model trained on a telecom customer dataset to help businesses take proactive steps toward **customer retention**.  

---

## 🚀 Features  
- **Interactive Input Form**: Enter customer details via dropdowns, sliders, and selectors.  
- **Real-Time Predictions**: Instantly see churn predictions and probability scores.  
- **User-Friendly Interface**: Simple, intuitive layout built with Streamlit.  
- **Data Preprocessing**:  
  - Handles missing values  
  - Label encoding for categorical features  
  - Scaling and transformation ready for machine learning  
- **Model Integration**: Pre-trained Logistic Regression model loaded via **joblib**.  

---

## 🛠️ Tech Stack  
- **Python**  
- **Pandas** & **NumPy** – Data manipulation and preprocessing  
- **Scikit-learn** – Logistic Regression model training & evaluation  
- **Streamlit** – Web application framework  
- **Joblib** – Model serialization & loading  

---

## 📂 Project Structure  
```
📦 churn-prediction-app
 ┣ 📜 app.py                # Streamlit application code
 ┣ 📜 logistic_churn_model.pkl  # Pre-trained Logistic Regression model
 ┣ 📜 data-set.csv          # Dataset used for training
 ┣ 📜 README.md             # Project documentation
 ┗ 📜 requirements.txt      # Python dependencies
```

---

## ⚙️ Installation & Setup  

1️⃣ **Clone the repository**  
```bash
git clone https://github.com/your-username/churn-prediction-app.git
cd churn-prediction-app
```

2️⃣ **Install dependencies**  
```bash
pip install -r requirements.txt
```

3️⃣ **Run the app**  
```bash
streamlit run app.py
```

---

## 📊 Model Information  
- **Algorithm**: Logistic Regression  
- **Target Variable**: `Churn` (0 = No Churn, 1 = Churn)  
- **Evaluation Metrics**:  
  - Accuracy: ~81%  
  - Precision/Recall trade-off optimized for recall (to catch most churn cases)  

---

## 📸 Screenshots  
| Input Form | Prediction Output |
|------------|------------------|
| ![Form Screenshot](images/form.png) | ![Prediction Screenshot](images/output.png) |

---

## 📌 Use Case  
This app is ideal for **telecom companies, subscription-based services, and customer success teams** who want to predict customer churn and reduce attrition.  

---

 
