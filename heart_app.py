import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# Custom CSS for UI
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        background-color: #e74c3c;
        color: white;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 10px;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
    }
    .disease { background-color: #ffebee; color: #c62828; }
    .healthy { background-color: #e8f5e9; color: #2e7d32; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        pipeline = joblib.load("heart_pipeline.pkl")
        return pipeline
    except FileNotFoundError:
        st.error("Model file not found.")
        return None

model = load_model()

st.title("Heart Disease Prediction")
st.markdown("Predict heart disease risk using Machine Learning")
st.markdown("---")

if model is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Personal Information")
        age = st.slider("Age", 20, 100, 50)

        sex = st.selectbox("Sex", ["Male", "Female"])

        st.subheader("Heart Metrics")

        cp = st.selectbox(
            "Chest Pain Type",
            ["typical angina", "atypical angina", "non-anginal", "asymptomatic"]
        )

        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
        chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)

        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])

        restecg = st.selectbox(
            "Resting ECG",
            ["normal", "st-t abnormality", "lv hypertrophy"]
        )

    with col2:
        thalach = st.slider("Max Heart Rate", 60, 220, 150)

        exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])

        oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0, 0.1)

        slope = st.selectbox(
            "Slope of Peak Exercise ST",
            ["upsloping", "flat", "downsloping"]
        )

        ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])

        thal = st.selectbox(
            "Thal",
            ["normal", "fixed defect", "reversable defect"]
        )

    st.markdown("---")

if st.button("Predict Heart Disease"):


    fbs_val = 1 if fbs == "Yes" else 0
    exang_val = 1 if exang == "Yes" else 0

    input_data = {
        "id": 0,  

        "age": age,
        "sex": sex,
        "dataset": "Cleveland",  # fixed value

        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs_val,
        "restecg": restecg,

        "thalch": thalach,  # corrected column name

        "exang": exang_val,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    input_df = pd.DataFrame([input_data])

    # 🔥 Pipeline handles everything
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    classes = model.classes_

    disease_index = list(classes).index(1)
    healthy_index = list(classes).index(0)

    disease_prob = proba[disease_index]
    healthy_prob = proba[healthy_index]
    

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
     if disease_prob > 0.5:
        st.markdown('<div class="prediction-box disease">DISEASE RISK</div>', unsafe_allow_html=True)
     else:
        st.markdown('<div class="prediction-box healthy">HEALTHY</div>', unsafe_allow_html=True)

    with col2:
        st.metric("Confidence", f"{max(proba) * 100:.1f}%")

    with col3:
        risk = "High" if disease_prob > 0.7 else "Medium" if disease_prob > 0.4 else "Low"
        st.metric("Risk Level", risk)


        # Probability chart
        fig = go.Figure(data=[
            go.Bar(name='Healthy', x=['Probability'], y=[healthy_prob], marker_color='#2ecc71'),
            go.Bar(name='Disease', x=['Probability'], y=[disease_prob], marker_color='#e74c3c')
        ])

        fig.update_layout(
            title='Prediction Probabilities',
            yaxis_title='Probability',
            barmode='group',
            height=350
        )

        st.plotly_chart(fig, use_container_width=True)

        # Result message
        if prediction == 1:
            st.error("**High Risk Detected:** Consult a cardiologist immediately. Maintain healthy lifestyle, monitor vitals.")
        else:
            st.success("**Low Risk:** Continue healthy habits - exercise regularly, balanced diet, regular checkups.")

        # Input summary
        with st.expander("Input Summary"):
            st.write(input_data)

        # Sidebar
    with st.sidebar:
        st.header("Model Info")
        st.info("""
        **Logistic Regression**
        - Accuracy: ~85%
        - Training: 303 patients
        - Features: 13 attributes
        """)

        st.header("Key Risk Factors")
        st.markdown("""
        - Chest pain type
        - Age & Gender
        - Blood pressure
        - Cholesterol levels
        - Max heart rate
        - Exercise angina
        """)

        st.markdown("---")
        st.caption("This is for educational purposes only")
        st.caption("Not a substitute for professional medical advice")
else:
    st.stop()



print(proba)
print(model.classes_)