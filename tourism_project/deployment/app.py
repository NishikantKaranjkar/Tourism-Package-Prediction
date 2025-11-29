import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="karanjkarnishi/tourism_package_prediction_model", filename="tourism_prediction_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Prediction (Visit with Us)")

st.write(
    "This internal application for **'Visit with Us'** which helps our travel team to predict whether a customer is likely "
    "to purchase the **Wellness Tourism Package** before being contacted."
)

st.write(
    "Please enter the customer's details below to generate a purchase likelihood prediction."
)


# Collect user input - Adjusted for Tourism Package Prediction features
Age = st.number_input("Age (Customer's age in years)", max_value=100, value=30)
Gender = st.selectbox("Gender", ["Male", "Female"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, value=0)
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "FreeLance", "Government Sector"])
Designation = st.selectbox("Designation", ["Manager", "Senior Manager", "Executive", "AVP", "VP", "Director"])
MonthlyIncome = st.number_input("Monthly Income", min_value=0.0, value=10000.0)
OwnCar = st.selectbox("Own Car", ["Yes", "No"])
NumberOfTrips = st.number_input("Number of Trips (annually)", min_value=0, value=0)
Passport = st.selectbox("Passport", ["Yes", "No"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.number_input("Duration of Pitch (in minutes)", min_value=0.0, value=10.0)
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, value=2)
PreferredPropertyStar = st.selectbox("Preferred Property Star", [1.0, 2.0, 3.0, 4.0, 5.0])
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
NumberOfFollowups = st.number_input("Number of Followups", min_value=0, value=0)

# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age': Age,
    'Gender': Gender,
    'MaritalStatus': MaritalStatus,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Occupation': Occupation,    
    'Designation': Designation,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,    
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'PreferredPropertyStar': PreferredPropertyStar,    
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': 1 if OwnCar == "Yes" else 0,    
    'MonthlyIncome': MonthlyIncome,
    'ProductPitched': ProductPitched,
    'NumberOfFollowups': NumberOfFollowups # Added missing column
}])

# Set the classification threshold
classification_threshold = 0.47

# Predict button
# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[:, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)

    result_text = (
        "purchase the Wellness Tourism Package"
        if prediction == 1
        else "not purchase the Wellness Tourism Package"
    )

    st.write(
        f"Based on the details provided, the customer is likely to **{result_text}**."
    )
