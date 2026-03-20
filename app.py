import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(page_title="Healthcare XAI", layout="wide")
st.title("🩺 Advanced Heart Disease Predictor (Explainable AI)")
st.markdown("---")

# 2. Load Real Dataset (Free Source)
@st.cache_data
def load_data():
    # Using the standard UCI Heart Disease dataset
    url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/heart.csv"
    return pd.read_csv(url)

df = load_data()

# 3. Train the Model (Background Process)
# Features: Age, Sex, CP, Trestbps, Chol, Fbs, Restecg, Thalach, Exang, Oldpeak, Slope, Ca, Thal
X = df.drop('target', axis=1)
y = df['target']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 4. Sidebar Inputs for User
st.sidebar.header("📝 Input Patient Data")
def user_input_features():
    age = st.sidebar.slider('Age', 20, 80, 50)
    sex = st.sidebar.selectbox('Sex (1=Male, 0=Female)', [1, 0])
    cp = st.sidebar.slider('Chest Pain Type (0-3)', 0, 3, 1)
    trestbps = st.sidebar.slider('Resting Blood Pressure', 90, 200, 120)
    chol = st.sidebar.slider('Serum Cholestoral (mg/dl)', 120, 564, 240)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)', [0, 1])
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved', 70, 202, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina (1=Yes, 0=No)', [0, 1])
    oldpeak = st.sidebar.slider('ST Depression (Oldpeak)', 0.0, 6.2, 1.0)
    ca = st.sidebar.slider('Number of Major Vessels (0-3)', 0, 3, 0)
    
    # Matching the 13 features of the dataset
    data = {'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
            'fbs': fbs, 'restecg': 1, 'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak,
            'slope': 1, 'ca': ca, 'thal': 2}
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# 5. Prediction Logic
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# 6. UI Layout: Results and Explainability
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📊 Prediction Results")
    risk_score = prediction_proba[0][1] * 100
    
    if prediction[0] == 1:
        st.error(f"**Status: HIGH RISK**")
    else:
        st.success(f"**Status: LOW RISK**")
    
    st.metric(label="Risk Probability", value=f"{risk_score:.2f}%")
    st.progress(risk_score / 100)

with col2:
    st.subheader("🔍 Why this prediction? (XAI)")
    
    # Initialize SHAP Explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(input_df)

    # Handle SHAP multi-output (pick index 1 for 'Disease')
    if len(shap_values.values.shape) == 3:
        exp = shap.Explanation(
            values=shap_values.values[0, :, 1], 
            base_values=shap_values.base_values[0, 1], 
            data=input_df.iloc[0], 
            feature_names=X.columns
        )
    else:
        exp = shap_values[0]

    # Create Plot
    fig, ax = plt.subplots()
    shap.plots.bar(exp, show=False)
    plt.title("Feature Contribution to Risk")
    st.pyplot(plt.gcf())

st.divider()
st.info("""
    **Project Info:** This application uses a **Random Forest Classifier** trained on the UCI Heart Disease dataset. 
    The **SHAP (SHapley Additive exPlanations)** values shown in the bar chart explain how much each specific health 
    factor increased (positive bar) or decreased (negative bar) the patient's risk score.
""")