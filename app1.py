import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('knn_model.sav')

# Load label encoders
label_encoders = joblib.load('label_encoders.pkl')

# Define the function to preprocess input
def preprocess_input(input_data):
    df = pd.DataFrame([input_data])
    for col in df.columns:
        if col in label_encoders:
            df[col] = label_encoders[col].transform(df[col])
    return df

# Define the prediction function
def predict(input_data):
    preprocessed_data = preprocess_input(input_data)
    prediction = model.predict(preprocessed_data)
    return int(prediction[0])

# Streamlit app
st.title("Prediksi KNN")

# Form input
st.header("Masukkan Data untuk Prediksi")

Gender = st.selectbox("Gender", ["Male", "Female"])
Marital_Status = st.selectbox("Marital Status", ["married", "divorced", "not married"])
Occupation = st.text_input("Occupation")
Monthly_Income = st.text_input("Monthly Income")
Educational_Qualifications = st.text_input("Educational Qualifications")
Feedback = st.text_input("Feedback")

# Prediction button
if st.button("Predict"):
    input_data = {
        "Gender": Gender,
        "Marital Status": Marital_Status,
        "Occupation": Occupation,
        "Monthly Income": Monthly_Income,
        "Educational Qualifications": Educational_Qualifications,
        "Feedback": Feedback
    }
    
    prediction = predict(input_data)
    st.write(f"Hasil Prediksi: {prediction}")
