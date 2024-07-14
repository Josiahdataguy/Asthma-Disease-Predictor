import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

model = pk.load(open("GBC.pkl", "rb"))

st.header("ASTHMA DISEASE PREDICTOR MODEL")

data = pd.read_csv("asthma.csv")

text_html = """
    <p style="color: teal;">
        The Asthma Disease Predictor Model is a machine learning project 
        aimed at predicting asthma, based on patient's data collected from 
        Kaggle.com. Various models, including Logistic Regression, Decision 
        Trees, Random Forest, Support Vector Machine (SVM), k-Nearest 
        Neighbors (k-NN), Gradient Boosting, XGBoost, and Neural Networks, 
        were employed and evaluated for this binary classification task. 
        This project underscores the importance of model selection in 
        achieving reliable and accurate predictions in healthcare applications.
    </p>
"""

st.markdown(text_html, unsafe_allow_html=True)

st.markdown('<br>', unsafe_allow_html=True)
st.markdown('<p style="font-size:20px; color:blue; font-weight:bold;">1. Patient Informations</p>', unsafe_allow_html=True)
PatientID = st.selectbox('**Enter Patient ID**', data['PatientID'].unique())

st.markdown('<br>', unsafe_allow_html=True)
st.markdown('<p style="font-size:20px; color:blue; font-weight:bold;">2. Demographic Details</p>', unsafe_allow_html=True)
Age = st.selectbox('**Age of Patient**', sorted(data['Age'].unique()))
Gender = st.selectbox('**Gender of Patient (0 is Female, 1 is Male)**', data['Gender'].unique())
Ethnicity = st.selectbox('**Ethnicity of the Patient (0: Caucasian, 1: African American, 2: Asian,3: Other)**', sorted(data['Ethnicity'].unique()))
EducationLevel = st.selectbox('**Education Level (0: None, 1: High School, 2: Bachelors, 3: Higher)**', sorted(data['EducationLevel'].unique()))

st.markdown('<br>', unsafe_allow_html=True)
st.markdown('<p style="font-size:20px; color:blue; font-weight:bold;">3. Lifestyle Factors</p>', unsafe_allow_html=True)
BMI = st.selectbox('**Enter BMI of the Patient**', sorted(data['BMI'].unique()))
Smoking = st.selectbox('**Does the Patient Smoke (0 is No / 1 is Yes)**', sorted(data['Smoking'].unique()))
PhysicalActivity = st.selectbox('**Physical Activity**', sorted(data['PhysicalActivity'].unique()))
DietQuality = st.selectbox('**Diet Quality**', sorted(data['DietQuality'].unique()))

st.markdown('<br>', unsafe_allow_html=True)
st.markdown('<p style="font-size:20px; color:blue; font-weight:bold;">4. Environmental and Allergy Factors</p>', unsafe_allow_html=True)
SleepQuality = st.selectbox('**Sleep Quality**', (data['SleepQuality'].unique()))
PollutionExposure = st.selectbox('**Pollution Exposure**', sorted(data['PollutionExposure'].unique()))
PollenExposure = st.selectbox('**Pollen Exposure**', sorted(data['PollenExposure'].unique()))
DustExposure = st.selectbox('**Dust Exposure**', sorted(data['DustExposure'].unique()))
PetAllergy = st.selectbox('**Pet Allergy (0 is No / 1 is Yes)**', sorted(data['PetAllergy'].unique()))

st.markdown('<br>', unsafe_allow_html=True)
st.markdown('<p style="font-size:20px; color:blue; font-weight:bold;">5. Medical History</p>', unsafe_allow_html=True)
FamilyHistoryAsthma = st.selectbox('**Family History Asthma (0 is No / 1 is Yes)**', sorted(data['FamilyHistoryAsthma'].unique()))
HistoryOfAllergies = st.selectbox('**History Of Allergies (0 is No / 1 is Yes)**', sorted(data['HistoryOfAllergies'].unique()))
Eczema  = st.selectbox('**Eczema (0 is No / 1 is Yes)**', sorted(data['Eczema'].unique()))
HayFever = st.selectbox('**Hay Fever (0 is No / 1 is Yes)**', sorted(data['HayFever'].unique()))
GastroesophagealReflux = st.selectbox('**Gastroesophageal Reflux (0 is No / 1 is Yes)**', sorted(data['GastroesophagealReflux'].unique()))

st.markdown('<br>', unsafe_allow_html=True)
st.markdown('<p style="font-size:20px; color:blue; font-weight:bold;">6. Clinical Measurements</p>', unsafe_allow_html=True)
LungFunctionFEV1 = st.selectbox('**Lung Function FEV1**', sorted(data['LungFunctionFEV1'].unique()))
LungFunctionFVC = st.selectbox('**Lung Function FVC**', sorted(data['LungFunctionFVC'].unique()))

st.markdown('<br>', unsafe_allow_html=True)
st.markdown('<p style="font-size:20px; color:blue; font-weight:bold;">7. Symptoms</p>', unsafe_allow_html=True)
Wheezing = st.selectbox('**Wheezing (0 is No / 1 is Yes)**', sorted(data['Wheezing'].unique()))
ShortnessOfBreath = st.selectbox('**Shortness Of Breath (0 is No / 1 is Yes)**', sorted(data['ShortnessOfBreath'].unique()))
ChestTightness = st.selectbox('**Chest Tightness (0 is No / 1 is Yes)**', sorted(data['ChestTightness'].unique()))
Coughing = st.selectbox('**Coughing (0 is No / 1 is Yes)**', sorted(data['Coughing'].unique()))
NighttimeSymptoms = st.selectbox('**Nighttime Symptoms (0 is No / 1 is Yes)**', sorted(data['NighttimeSymptoms'].unique()))
ExerciseInduced = st.selectbox('**Exercise Induced (0 is No / 1 is Yes)**', sorted(data['ExerciseInduced'].unique()))

if st.button("Predict"):
    inputdata = pd.DataFrame(
    [[PatientID, Age, Gender, Ethnicity, EducationLevel, BMI,Smoking, PhysicalActivity, DietQuality, SleepQuality, PollutionExposure, PollenExposure, DustExposure, PetAllergy,FamilyHistoryAsthma, HistoryOfAllergies, Eczema, HayFever,GastroesophagealReflux, LungFunctionFEV1, LungFunctionFVC,Wheezing, ShortnessOfBreath, ChestTightness, Coughing,NighttimeSymptoms, ExerciseInduced]],
    columns=['PatientID', 'Age', 'Gender', 'Ethnicity', 'EducationLevel', 'BMI',
       'Smoking', 'PhysicalActivity', 'DietQuality', 'SleepQuality',
       'PollutionExposure', 'PollenExposure', 'DustExposure', 'PetAllergy',
       'FamilyHistoryAsthma', 'HistoryOfAllergies', 'Eczema', 'HayFever',
       'GastroesophagealReflux', 'LungFunctionFEV1', 'LungFunctionFVC',
       'Wheezing', 'ShortnessOfBreath', 'ChestTightness', 'Coughing',
       'NighttimeSymptoms', 'ExerciseInduced'])

    Prediction = model.predict(inputdata)
    st.markdown('**THE PATIENT STATUS IS:** (0 indicates No and 1 indicates Yes.)<br>' +
            '**' + str(Prediction) + '**', 
            unsafe_allow_html=True)




