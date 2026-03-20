import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Page Setup
st.set_page_config(page_title="Pro Healthcare XAI", layout="wide")
st.title("🏥 Pro-Grade Heart Disease Diagnostic Dashboard")
st.markdown("---")

# 2. Data Engine
@st.cache_data
def get_data():
    url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/heart.csv"
    df = pd.read_csv(url)
    return df

df = get_data()
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Sidebar: Model Selection & Inputs
st.sidebar.header("⚙️ Settings")
model_type = st.sidebar.selectbox("Select Algorithm", ["Random Forest (Advanced)", "Logistic Regression (Basic)"])

st.sidebar.header("📝 Patient Vitals")
def get_user_input():
    age = st.sidebar.slider('Age', 20, 80, 50)
    sex = st.sidebar.selectbox('Sex (1=M, 0=F)', [1, 0])
    cp = st.sidebar.slider('Chest Pain Type (0-3)', 0, 3, 1)
    trestbps = st.sidebar.slider('Resting BP', 90, 200, 120)
    chol = st.sidebar.slider('Cholesterol', 120, 564, 240)
    thalach = st.sidebar.slider('Max Heart Rate', 70, 202, 150)
    ca = st.sidebar.slider('Major Vessels (0-3)', 0, 3, 0)
    oldpeak = st.sidebar.slider('ST Depression', 0.0, 6.0, 1.0)
    
    data = {'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
            'fbs': 0, 'restecg': 1, 'thalach': thalach, 'exang': 0, 'oldpeak': oldpeak,
            'slope': 1, 'ca': ca, 'thal': 2}
    return pd.DataFrame(data, index=[0])

input_df = get_user_input()

# 4. Model Training Logic
if model_type == "Random Forest (Advanced)":
    model = RandomForestClassifier(n_estimators=100, random_state=42)
else:
    model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# 5. UI Tabs
tab1, tab2, tab3 = st.tabs(["🎯 Patient Diagnosis", "📊 Model Performance", "🔍 Global AI Logic"])

with tab1:
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.subheader("Results")
        prob = model.predict_proba(input_df)[0][1] * 100
        if prob > 50:
            st.error(f"**HIGH RISK** ({prob:.1f}%)")
        else:
            st.success(f"**LOW RISK** ({prob:.1f}%)")
        st.progress(prob/100)
    
    with col2:
        st.subheader("Local Explanation (Why this patient?)")
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(input_df)
        fig, ax = plt.subplots()
        # Handle different SHAP output formats for different models
        if len(shap_values.values.shape) == 3: # Random Forest
            shap.plots.bar(shap_values[0,:,1], show=False)
        else: # Logistic Regression
            shap.plots.bar(shap_values[0], show=False)
        st.pyplot(plt.gcf())

with tab2:
    st.subheader("Model Evaluation Metrics")
    c1, c2 = st.columns(2)
    c1.metric("Model Accuracy", f"{acc*100:.2f}%")
    
    st.write("---")
    st.subheader("Confusion Matrix")
    # This shows how many times the AI was right vs wrong
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig_cm)

with tab3:
    st.subheader("Global Feature Importance")
    st.write("This chart shows which factors are most important across all 300+ patients in the dataset.")
    # Global SHAP Summary
    explainer_gen = shap.Explainer(model, X_train)
    shap_values_gen = explainer_gen(X_test)
    fig_gen, ax_gen = plt.subplots()
    if len(shap_values_gen.values.shape) == 3:
        shap.plots.beeswarm(shap_values_gen[:,:,1], show=False)
    else:
        shap.plots.beeswarm(shap_values_gen, show=False)
    st.pyplot(plt.gcf())