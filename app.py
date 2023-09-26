# Import necessary libraries
import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load the saved XGBoost classifier model
loaded_model = joblib.load('customer_churn_classifier.pkl')

# Load the training data from a CSV file 
X_train = pd.read_csv('train_data.csv')

# Initialize LabelEncoder for Location
label_encoder = LabelEncoder()

# Fit the LabelEncoder with training data for 'Location'
label_encoder.fit(X_train['Location']) 

# Load the MinMaxScaler and fit it to your training data
scaler = MinMaxScaler()
scaler.fit(X_train)  

# Define the Streamlit app
st.title("Customer Churn Prediction")
st.write("This app predicts customer churn using a trained model.")

# Create input widgets for user interaction
churn_history_count = st.number_input("Churn History Count")
monthly_bill = st.number_input("Monthly Bill")
billing_to_usage_ratio = st.number_input("Billing to Usage Ratio")
usage_per_billing_cycle = st.number_input("Usage per Billing Cycle")
total_usage_gb = st.number_input("Total Usage GB")
age = st.slider("Age", 18, 100, 30)
tenure_years = st.number_input("Tenure Years")
location = st.selectbox("Location", ["Houston", "Los Angeles", "Miami", "New York"])

# Define a function to make predictions
def predict_churn(churn_history_count, monthly_bill, billing_to_usage_ratio, usage_per_billing_cycle, total_usage_gb, age, tenure_years, location):
    # Prepare input data as a DataFrame
    input_data = pd.DataFrame({
        'Churn_History_Count': [churn_history_count],
        'Monthly_Bill': [monthly_bill],
        'Billing_to_Usage_Ratio': [billing_to_usage_ratio],
        'Usage_Per_Billing_Cycle': [usage_per_billing_cycle],
        'Total_Usage_GB': [total_usage_gb],
        'Age': [age],
        'Tenure_Years': [tenure_years],
        'Location': [location]
    })

    # Label encode the 'Location' column
    input_data['Location'] = label_encoder.transform(input_data['Location'])

    # Scale the input data using Min-Max scaling
    input_data_scaled = scaler.transform(input_data)

    # Use the loaded model to make predictions
    prediction = loaded_model.predict(input_data_scaled)

    return prediction  # Return the prediction result

# Get predictions when the user clicks a button
if st.button("Predict"):
    result = predict_churn(churn_history_count, monthly_bill, billing_to_usage_ratio, usage_per_billing_cycle, total_usage_gb, age, tenure_years, location)
    if result == 1:
        st.write("Churn Prediction: Churn")
    else:
        st.write("Churn Prediction: No Churn")
