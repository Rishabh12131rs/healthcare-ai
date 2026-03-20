import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(page_title="Universal Health AI", layout="wide")
st.title("🏥 Universal AI Healthcare Diagnostic Portal")
st.markdown("---")

# 2. Sidebar - Service Selection
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100)
st.sidebar.title("Select Service")
app_mode = st.sidebar.selectbox("What is your health concern?", 
    ["Heart Health Analysis", "Diabetes Risk Screening", "General Symptom Checker"])

# --- DATA ENGINES ---
@st.cache_data
def load_heart_data():
    return pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/heart.csv")

@st.cache_data
def load_diabetes_data():
    # Standard Pima Indians Diabetes Dataset
    return pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv", 
                      names=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DPF', 'Age', 'Outcome'])

# --- MODULE 1: HEART HEALTH ---
if app_mode == "Heart Health Analysis":
    st.header("🫀 Advanced Heart Diagnostic")
    df = load_heart_data()
    X = df.drop('target', axis=1)
    y = df['target']
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
    
    st.sidebar.subheader("Input Heart Vitals")
    age = st.sidebar.slider('Age', 20, 80, 50)
    cp = st.sidebar.slider('Chest Pain Type (0-3)', 0, 3, 1)
    trestbps = st.sidebar.slider('Resting Blood Pressure', 90, 200, 120)
    chol = st.sidebar.slider('Cholesterol', 120, 564, 240)
    
    input_df = pd.DataFrame({'age': age, 'sex': 1, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
                             'fbs': 0, 'restecg': 1, 'thalach': 150, 'exang': 0, 'oldpeak': 1.0,
                             'slope': 1, 'ca': 0, 'thal': 2}, index=[0])
    
    if st.button("Run Heart Analysis"):
        prob = model.predict_proba(input_df)[0][1] * 100
        st.metric("Heart Risk Probability", f"{prob:.2f}%")
        st.progress(prob/100)
        
        # XAI Explanation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(input_df)
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values[0,:,1] if len(shap_values.values.shape)==3 else shap_values[0], show=False)
        st.pyplot(plt.gcf())

# --- MODULE 2: DIABETES SCREENING ---
elif app_mode == "Diabetes Risk Screening":
    st.header("🩸 Diabetes Risk Assessment")
    df = load_diabetes_data()
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
    
    st.sidebar.subheader("Input Diabetes Markers")
    glucose = st.sidebar.slider('Glucose Level', 0, 200, 120)
    bmi = st.sidebar.slider('BMI (Body Mass Index)', 10.0, 50.0, 25.0)
    age_d = st.sidebar.slider('Age', 1, 100, 30)
    bp = st.sidebar.slider('Blood Pressure', 40, 140, 80)
    
    input_df = pd.DataFrame({'Pregnancies': 2, 'Glucose': glucose, 'BloodPressure': bp, 
                             'SkinThickness': 20, 'Insulin': 80, 'BMI': bmi, 
                             'DPF': 0.5, 'Age': age_d}, index=[0])
    
    if st.button("Run Diabetes Screening"):
        prob = model.predict_proba(input_df)[0][1] * 100
        st.write(f"### Result: {'High Risk' if prob > 50 else 'Low Risk'}")
        st.metric("Diabetes Risk Score", f"{prob:.2f}%")
        
        st.subheader("Why this score?")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(input_df)
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values[0,:,1] if len(shap_values.values.shape)==3 else shap_values[0], show=False)
        st.pyplot(plt.gcf())

# --- MODULE 3: SYMPTOM CHECKER ---
elif app_mode == "General Symptom Checker":
    st.header("🔍 Intelligent Symptom Analysis")
    st.write("Enter your symptoms below for a preliminary AI assessment.")
    user_input = st.text_area("Example: I have a persistent headache, high fever, and body aches.")
    
    if st.button("Analyze Symptoms"):
        if "fever" in user_input.lower() and "headache" in user_input.lower():
            st.warning("Assessment: Symptoms may indicate a viral infection. Please check your temperature.")
        elif "chest" in user_input.lower() or "breath" in user_input.lower():
            st.error("Urgent: Please use the 'Heart Health Analysis' module and consult a doctor immediately.")
        else:
            st.info("Assessment: Please provide more specific symptoms or use one of our specialized modules.")

st.markdown("---")
st.caption("⚠️ Disclaimer: This is an AI project for educational purposes and not a substitute for professional medical advice.")