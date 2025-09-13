import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

# ----------------------------
# Load and preprocess dataset
# ----------------------------
data = pd.read_csv("Plastic Waste Around the World.csv")

# Encode categorical columns
le_sources = LabelEncoder()
data["Main_Sources"] = le_sources.fit_transform(data["Main_Sources"])

le_risk = LabelEncoder()
data["Coastal_Waste_Risk"] = le_risk.fit_transform(data["Coastal_Waste_Risk"])

# Regression Model (Predict Recycling Rate)
X_reg = data[["Total_Plastic_Waste_MT", "Main_Sources", "Per_Capita_Waste_KG"]]
y_reg = data["Recycling_Rate"]

reg_model = LinearRegression()
reg_model.fit(X_reg, y_reg)

# Classification Model (Predict Coastal Waste Risk)
X_clf = data[["Total_Plastic_Waste_MT", "Main_Sources", "Per_Capita_Waste_KG", "Recycling_Rate"]]
y_clf = data["Coastal_Waste_Risk"]

clf_model = RandomForestClassifier(random_state=42)
clf_model.fit(X_clf, y_clf)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🌍 AI Plastic Waste Classifier")

st.write("This app predicts **Recycling Rate** and **Coastal Waste Risk** based on plastic waste data.")

# User Inputs
country = st.text_input("Country Name", "India")
total_waste = st.number_input("Total Plastic Waste (MT)", min_value=0.0, step=1000.0)
per_capita = st.number_input("Per Capita Waste (KG)", min_value=0.0, step=0.1)
main_source = st.selectbox("Main Source of Waste", le_sources.classes_)

# Encode main_source
main_source_encoded = le_sources.transform([main_source])[0]

# ----------------------------
# Predictions
# ----------------------------
if st.button("Predict"):
    # Regression Prediction
    reg_input = pd.DataFrame([[total_waste, main_source_encoded, per_capita]],
                             columns=["Total_Plastic_Waste_MT", "Main_Sources", "Per_Capita_Waste_KG"])
    recycling_pred = reg_model.predict(reg_input)[0]

    # Classification Prediction
    clf_input = pd.DataFrame([[total_waste, main_source_encoded, per_capita, recycling_pred]],
                             columns=["Total_Plastic_Waste_MT", "Main_Sources", "Per_Capita_Waste_KG", "Recycling_Rate"])
    risk_pred = clf_model.predict(clf_input)[0]
    risk_label = le_risk.inverse_transform([risk_pred])[0]

    # Results
    st.success(f"♻️ Predicted Recycling Rate: {recycling_pred:.2f}%")
    st.warning(f"🌊 Predicted Coastal Waste Risk: {risk_label}")
