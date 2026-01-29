import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# --- DATA LOADING & TRAINING ---
@st.cache_data
def load_data():
    return pd.read_csv('heart_disease_data.csv')

@st.cache_resource
def train_models(df):
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_acc = accuracy_score(y_test, lr_model.predict(X_test))
    
    # Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
    
    return lr_model, rf_model, lr_acc, rf_acc, X_train

heart_df = load_data()
lr_model, rf_model, lr_acc, rf_acc, X_train = train_models(heart_df)

# --- SIDEBAR INPUTS ---
st.sidebar.header('Patient Data')

def user_input_features():
    age = st.sidebar.slider('Age', 1, 100, 1)
    sex = st.sidebar.selectbox('Sex', ['Female', 'Male'])
    cp = st.sidebar.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
    trestbps = st.sidebar.slider('Resting Blood Pressure', 94, 200, 120)
    chol = st.sidebar.slider('Cholesterol', 126, 564, 200)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', ['False', 'True'])
    restecg = st.sidebar.selectbox('Resting ECG Results', ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
    thalach = st.sidebar.slider('Max Heart Rate Achieved', 71, 202, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina', ['No', 'Yes'])
    oldpeak = st.sidebar.slider('ST Depression Induced by Exercise', 0.0, 6.2, 1.0)
    slope = st.sidebar.selectbox('Slope of the Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'])
    ca = st.sidebar.slider('Major Vessels Colored by Flourosopy', 0, 4, 0)
    thal = st.sidebar.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversable Defect', 'Unknown'])

    # Mapping inputs to model format
    sex_map = {'Female': 0, 'Male': 1}
    cp_map = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}
    fbs_map = {'False': 0, 'True': 1}
    restecg_map = {'Normal': 0, 'ST-T Wave Abnormality': 1, 'Left Ventricular Hypertrophy': 2}
    exang_map = {'No': 0, 'Yes': 1}
    slope_map = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
    thal_map = {'Normal': 1, 'Fixed Defect': 2, 'Reversable Defect': 3, 'Unknown': 0} # Adjust based on dataset specifics usually 1,2,3 or 0,1,2,3

    data = {
        'age': age,
        'sex': sex_map[sex],
        'cp': cp_map[cp],
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs_map[fbs],
        'restecg': restecg_map[restecg],
        'thalach': thalach,
        'exang': exang_map[exang],
        'oldpeak': oldpeak,
        'slope': slope_map[slope],
        'ca': ca,
        'thal': thal_map[thal]
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- MAIN APP LAYOUT ---
st.title("Heart Disease Prediction Web App")
try:
    image = Image.open('human-heart-image-free-png.webp')
    st.image(image, width=300)
except Exception as e:
    st.warning("Image not found, skipping.")

tab1, tab2, tab3 = st.tabs(["Prediction", "Data Analysis", "Model Comparison"])

with tab1:
    st.subheader("User Input Parameters")
    st.write(input_df)

    if st.button('Predict Heart Disease'):
        # Using Random Forest as default for better accuracy usually, or we can let user choose.
        # Let's use Random Forest here as the "Advanced" one.
        prediction = rf_model.predict(input_df)
        prediction_proba = rf_model.predict_proba(input_df)

        if prediction[0] == 0:
            st.success(f'Healthy Heart (Probability: {prediction_proba[0][0]:.2f})')
        else:
            st.error(f'Heart Disease Detected (Probability: {prediction_proba[0][1]:.2f})')
            st.warning("Please consult a doctor for further checkups.")

with tab2:
    st.subheader("Data Analysis")
    
    # Correlation Heatmap
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(heart_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

    # Histograms
    st.write("### Age Distribution by Heart Disease Status")
    fig2, ax2 = plt.subplots()
    sns.histplot(data=heart_df, x='age', hue='target', kde=True, element="step", ax=ax2)
    st.pyplot(fig2)

with tab3:
    st.subheader("Model Perfromance Comparison")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Logistic Regression Accuracy", value=f"{lr_acc:.2%}")
    with col2:
        st.metric(label="Random Forest Accuracy", value=f"{rf_acc:.2%}")

    st.write("Random Forest typically performs better on complex datasets as it captures non-linear relationships.")

