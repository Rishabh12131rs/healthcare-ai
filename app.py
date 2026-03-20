import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(page_title="Healthcare XAI", layout="wide")
st.title("🩺 Advanced Heart Disease Predictor (Explainable AI)")
st.write("This project uses **Random Forest** and **SHAP** to explain AI decisions.")

# 2. Load Real Dataset (Free Source)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/heart.csv"
    return pd.read_csv(url)

df = load_data()

# 3. Train the Model
X = df.drop('target', axis=1)
y = df['target']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 4. Sidebar Inputs for User
st.sidebar.header("Input Patient Data")
def user_input_features():
    age = st.sidebar.slider('Age', 20, 80, 50)
    sex = st.sidebar.selectbox('Sex (1=M, 0=F)', [1, 0])
    cp = st.sidebar.slider('Chest Pain Type (0-3)', 0, 3, 1)
    trestbps = st.sidebar.slider('Resting Blood Pressure', 90, 200, 120)
    chol = st.sidebar.slider('Serum Cholestoral in mg/dl', 120, 564, 240)
    
    data = {'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
            'fbs': 0, 'restecg': 1, 'thalach': 150, 'exang': 0, 'oldpeak': 1.0,
            'slope': 1, 'ca': 0, 'thal': 2}
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# 5. Prediction Logic
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# 6. Display Results
col1, col2 = st.columns(2)

with col1:
    st.subheader("Prediction")
    status = "❤️ High Risk" if prediction[0] == 1 else "✅ Low Risk"
    st.metric(label="Status", value=status)
    st.write(f"Confidence: {np.max(prediction_proba)*100:.2f}%")

with col2:
    st.subheader("Explainable AI (SHAP)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    
    # Generate SHAP Force Plot
    fig, ax = plt.subplots()
    shap.plots.bar(explainer(X)[0], show=False) # Showing global importance
    st.pyplot(plt.gcf())

st.info("The graph above shows which symptoms (Age, Chol, etc.) are the most important for the AI's decision.")