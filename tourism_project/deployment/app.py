
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

if not MODEL_PATH.exists():
    st.error(
        "model.pkl was not found next to app.py. "
        "Please upload both files to the GitHub repository."
    )
    st.stop()

model = joblib.load(MODEL_PATH)

st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="✈️"
)

st.title("Wellness Tourism Package Prediction")
st.write(
    "Enter customer details to predict whether the customer may purchase "
    "the Wellness Tourism Package."
)

age = st.number_input("Age", min_value=18, max_value=100, value=35)
type_of_contact = st.selectbox(
    "Type of Contact", ["Company Invited", "Self Enquiry"]
)
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.number_input(
    "Duration of Pitch", min_value=0.0, max_value=150.0, value=10.0
)
occupation = st.selectbox(
    "Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"]
)
gender = st.selectbox("Gender", ["Female", "Male", "Fe Male"])
number_of_person_visiting = st.number_input(
    "Number Of Persons Visiting", min_value=1, max_value=20, value=3
)
number_of_followups = st.number_input(
    "Number Of Followups", min_value=0, max_value=20, value=3
)
product_pitched = st.selectbox(
    "Product Pitched", ["Basic", "Deluxe", "King", "Standard", "Super Deluxe"]
)
preferred_property_star = st.selectbox(
    "Preferred Property Star", [3.0, 4.0, 5.0]
)
marital_status = st.selectbox(
    "Marital Status", ["Single", "Married", "Divorced", "Unmarried"]
)
number_of_trips = st.number_input(
    "Number Of Trips", min_value=0.0, max_value=50.0, value=3.0
)
passport = st.selectbox("Passport", [0, 1])
pitch_satisfaction_score = st.selectbox(
    "Pitch Satisfaction Score", [1, 2, 3, 4, 5]
)
own_car = st.selectbox("Own Car", [0, 1])
number_of_children_visiting = st.number_input(
    "Number Of Children Visiting", min_value=0.0, max_value=20.0, value=1.0
)
designation = st.selectbox(
    "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)
monthly_income = st.number_input(
    "Monthly Income", min_value=0.0, max_value=500000.0, value=20000.0
)

if st.button("Predict"):
    input_data = pd.DataFrame([{
        "Age": age,
        "TypeofContact": type_of_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": number_of_person_visiting,
        "NumberOfFollowups": number_of_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_property_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": number_of_trips,
        "Passport": passport,
        "PitchSatisfactionScore": pitch_satisfaction_score,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": number_of_children_visiting,
        "Designation": designation,
        "MonthlyIncome": monthly_income
    }])

    prediction = model.predict(input_data)[0]

    probability = None
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success("Prediction: Customer is likely to purchase the package.")
    else:
        st.info("Prediction: Customer is unlikely to purchase the package.")

    if probability is not None:
        st.write("Purchase probability:", f"{probability:.2%}")
