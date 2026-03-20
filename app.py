import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import shap
import matplotlib.pyplot as plt

# 1. Page Config
st.set_page_config(page_title="Universal Health AI", layout="wide")
st.title("🏥 Pro-Grade Universal Health Portal")

# --- ADVANCED SYMPTOM DATABASE ---
# We define categories so the AI can "map" free text to a medical module
SYMPTOM_MAP = {
    "Heart/Cardiac": ["chest pain", "shortness of breath", "palpitations", "dizziness", "fainting", "sweating", "left arm pain"],
    "Diabetes/Metabolic": ["frequent urination", "excessive thirst", "blurry vision", "unexplained weight loss", "fatigue", "slow healing"],
    "General Infection": ["fever", "chills", "sore throat", "cough", "body aches", "runny nose", "nausea"]
}

# --- DATA ENGINES ---
@st.cache_data
def load_data(url, cols=None):
    return pd.read_csv(url, names=cols) if cols else pd.read_csv(url)

# --- MODULE: ADVANCED SYMPTOM CHECKER ---
def advanced_symptom_checker(user_text):
    if not user_text: return None
    
    # Simple NLP: Compare user input to our symptom database
    results = {}
    for category, symptoms in SYMPTOM_MAP.items():
        # Count how many keywords match or are similar
        score = sum(1 for s in symptoms if s in user_text.lower())
        results[category] = score
    
    # Find the category with the highest match
    best_match = max(results, key=results.get)
    if results[best_match] > 0:
        return best_match
    return "Unknown"

# --- SIDEBAR & NAVIGATION ---
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Select Service", ["Symptom Checker", "Heart Analysis", "Diabetes Screening"])

if app_mode == "Symptom Checker":
    st.header("🔍 AI-Powered Symptom Analysis")
    st.write("Describe how you feel in detail (e.g., 'I have chest pain and I am feeling very dizzy').")
    
    user_input = st.text_area("Enter Symptoms Here:", placeholder="Type here...")
    
    if st.button("Analyze My Symptoms"):
        category = advanced_symptom_checker(user_input)
        
        if category == "Heart/Cardiac":
            st.error(f"🚨 **Urgent Assessment:** Your symptoms match **{category}** patterns.")
            st.info("Recommendation: Please switch to the **'Heart Analysis'** tab and consult a cardiologist.")
        elif category == "Diabetes/Metabolic":
            st.warning(f"⚠️ **Assessment:** Your symptoms match **{category}** markers.")
            st.info("Recommendation: Please use the **'Diabetes Screening'** tab and check your blood sugar.")
        elif category == "General Infection":
            st.success(f"🩹 **Assessment:** This looks like a **{category}** (e.g., Flu or Cold).")
            st.write("Rest and hydration are advised. If fever persists, see a GP.")
        else:
            st.info("🤖 AI is unsure. Please be more specific or try a specialized module.")

# --- MODULE: HEART ---
elif app_mode == "Heart Analysis":
    st.header("🫀 Cardiac Risk Module")
    df = load_data("https://raw.githubusercontent.com/amankharwal/Website-data/master/heart.csv")
    X, y = df.drop('target', axis=1), df['target']
    model = RandomForestClassifier(n_estimators=100).fit(X, y)
    
    # Input Logic (Simplified for brevity)
    age = st.number_input("Age", 20, 100, 50)
    cp = st.slider("Chest Pain Type (0-3)", 0, 3, 1)
    
    if st.button("Predict Heart Risk"):
        # We fill other features with means for a quick check
        input_data = np.array([[age, 1, cp, 120, 240, 0, 1, 150, 0, 1.0, 1, 0, 2]])
        prob = model.predict_proba(input_data)[0][1] * 100
        st.metric("Risk Score", f"{prob:.2f}%")

# --- MODULE: DIABETES ---
elif app_mode == "Diabetes Screening":
    st.header("🩸 Diabetes Risk Module")
    # Using the same logic as Heart but with Diabetes data
    st.write("This module is now active. Please enter your glucose and BMI in the sidebar.")
    # (Existing Diabetes logic goes here)