import pandas as pd
import numpy as np
import streamlit as st
import pickle
import base64

# Load the ML model
model = pickle.load(open('Heart_prediction_model.pkl', 'rb'))
data = pd.read_csv('heart_disease_data.csv')

# Function to set background image
def set_background(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
    
    bg_image = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
    }}
    </style>
    """
    st.markdown(bg_image, unsafe_allow_html=True)

set_background("background.jpg")

# Load mean and std values
with open('mean_std_values.pkl', 'rb') as f:
    mean_std_values = pickle.load(f)

# UI Header
st.header('Heart Disease Predictor')

def main():
    st.title('Heart Disease Prediction')

    # Gender selection
    gender_options = {'Male': 1, 'Female': 0}
    gender = st.selectbox('Choose Gender', list(gender_options.keys()))
    gender = gender_options[gender]

    # Input Features
    age = int(st.number_input("Enter the age", min_value=1, max_value=120, value=25))

    cp_options = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic']
    cp = st.selectbox('Chest Pain Type', cp_options)
    cp_num = cp_options.index(cp)

    trestbps = st.number_input('Resting Blood Pressure', 90, 200, 120)
    chol = st.number_input('Cholesterol', 100, 600, 250)

    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['False', 'True'])
    fbs_num = 1 if fbs == 'True' else 0

    restecg_options = ['Normal', 'ST-T Abnormality', 'Left Ventricular Hypertrophy']
    restecg = st.selectbox('Resting ECG Results', restecg_options)
    restecg_num = restecg_options.index(restecg)

    thalach = st.number_input('Max Heart Rate Achieved', 70, 220, 150)

    exang = st.selectbox('Exercise Induced Angina', ['No', 'Yes'])
    exang_num = 1 if exang == 'Yes' else 0

    oldpeak = st.number_input('ST Depression', 0.0, 6.2, 1.0)

    slope_mapping = {'Upsloping': 1, 'Flat': 2, 'Downsloping': 3}
    slope = st.selectbox('Slope of ST Segment', list(slope_mapping.keys()))
    slope_num = slope_mapping[slope]

    ca = st.number_input('Number of Major Vessels', 0, 4, 1)

    thal_mapping = {'Normal': 1, 'Fixed Defect': 2, 'Reversible Defect': 3}
    thal = st.selectbox('Thalassemia', list(thal_mapping.keys()))
    thal_num = thal_mapping[thal]

    # Prediction
    if st.button('Predict'):
        user_input = pd.DataFrame({
            'age': [age], 'sex': [gender], 'cp': [cp_num], 'trestbps': [trestbps],
            'chol': [chol], 'fbs': [fbs_num], 'restecg': [restecg_num], 'thalach': [thalach],
            'exang': [exang_num], 'oldpeak': [oldpeak], 'slope': [slope_num], 'ca': [ca], 'thal': [thal_num]
        })

        user_input = (user_input - mean_std_values['mean']) / mean_std_values['std']

        prediction = model.predict(user_input)
        confidence = model.predict_proba(user_input)[0][prediction[0]]

        if prediction[0] == 1:
            st.error(f"❌ **High Risk of Heart Disease**\n\n**Confidence:** {round(confidence * 100, 2)}%")
        else:
            st.success(f"✅ **Low Risk of Heart Disease**\n\n**Confidence:** {round(confidence * 100, 2)}%")

if __name__ == '__main__':
    main()





# input these are value check  Positive
# age = 65
# sex = 1  # Male
# cp = 3  # Asymptomatic (most concerning chest pain type)
# trestbps = 180  # High blood pressure
# chol = 300  # High cholesterol
# fbs = 1  # Fasting blood sugar > 120 mg/dl
# restecg = 2  # Left Ventricular Hypertrophy (bad ECG result)
# thalach = 100  # Low maximum heart rate
# exang = 1  # Exercise-induced angina
# oldpeak = 3.5  # High ST depression (severe heart condition)
# slope = 2  # Flat ST segment (sign of risk)
# ca = 3  # 3 major vessels blocked
# thal = 2  # Fixed defect (serious heart issue)



#Negative

# age = 30
# sex = 0  # Female
# cp = 1  # Atypical Angina (not very risky)
# trestbps = 120  # Normal blood pressure
# chol = 180  # Normal cholesterol level
# fbs = 0  # Fasting blood sugar < 120 mg/dl
# restecg = 0  # Normal ECG
# thalach = 180  # High maximum heart rate (good fitness)
# exang = 0  # No exercise-induced angina
# oldpeak = 0.0  # No ST depression (healthy heart)
# slope = 0  # Upsloping ST segment (healthy sign)
# ca = 0  # No major vessel blockage
# thal = 1  # Normal Thalassemia test

