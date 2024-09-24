# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
# Replace 'data.csv' with the path to your dataset
data = pd.read_csv('data.csv')

# Explore the dataset
print(data.head())
print(data.info())

# Preprocessing
data.fillna(0, inplace=True)  # Handle missing values

# Feature selection
features = data.drop('is_fraud', axis=1)  # All columns except 'is_fraud'
target = data['is_fraud']  # Target variable

# Normalize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_classifier.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model and scaler for future use
joblib.dump(rf_classifier, 'fraud_detection_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Function to predict if a transaction is fraudulent
def predict_fraud(transaction_data):
    # Load the trained model and scaler
    model = joblib.load('fraud_detection_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # Prepare the transaction data
    transaction_df = pd.DataFrame(transaction_data)
    
    # Scale the transaction data
    transaction_scaled = scaler.transform(transaction_df)
    
    # Predict fraud
    predictions = model.predict(transaction_scaled)
    
    # Return predictions
    return predictions

# Example usage
new_transactions = [
    {'feature1': value1, 'feature2': value2, 'feature3': value3},  # Replace with actual feature names and values
    # Add more transactions as needed
]

predictions = predict_fraud(new_transactions)
print("Fraud Predictions:", predictions)
