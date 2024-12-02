import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import shap

# Simulating a synthetic dataset
np.random.seed(0)
n_samples = 1000

# Synthetic features (blood pressure, heart rate, glucose, etc.)
data = pd.DataFrame({
    'age': np.random.randint(20, 80, size=n_samples),
    'blood_pressure': np.random.randint(90, 180, size=n_samples),
    'heart_rate': np.random.randint(60, 120, size=n_samples),
    'glucose_level': np.random.randint(70, 200, size=n_samples),
    'cholesterol': np.random.randint(150, 250, size=n_samples),
    'weight': np.random.randint(45, 100, size=n_samples),
    'bmi': np.random.uniform(18.5, 40, size=n_samples),
    'family_history': np.random.randint(0, 2, size=n_samples)  # 0: No, 1: Yes
})

# Target variable (1: Disease risk, 0: No disease risk)
data['disease_risk'] = np.random.randint(0, 2, size=n_samples)

# Feature columns
features = ['age', 'blood_pressure', 'heart_rate', 'glucose_level', 'cholesterol', 'weight', 'bmi', 'family_history']
X = data[features]
y = data['disease_risk']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (Standardization)
scaler = StandardScaler()

# Fit scaler on training data and transform both training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled data back into DataFrame to retain feature names
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

# Build and train the RandomForest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled_df, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_scaled_df)

# Evaluate the model
print(classification_report(y_test, y_pred))

# SHAP values for interpretability
explainer = shap.TreeExplainer(rf_model)

# Get SHAP values for the predictions (for both classes)
shap_values = explainer.shap_values(X_test_scaled_df)

# Debugging: Check the shape of shap_values for class 1 (Disease Risk)
print(f"Shape of SHAP values for class 1: {shap_values[1].shape}")  # For disease risk (class 1)
print(f"Shape of X_test_scaled_df: {X_test_scaled_df.shape}")

# Select SHAP values for class 1 (disease risk) if shap_values is a list
# This is the case for binary classification with TreeExplainer
if isinstance(shap_values, list):
    shap_values = shap_values[1]
    print("Selected SHAP values for class 1 (disease risk).")

# Ensure that you are passing the correct data (scaled X_test with feature names)
# SHAP expects a DataFrame with the same number of features
shap.summary_plot(shap_values, X_test_scaled_df)  # Use X_test_scaled_df (scaled DataFrame)

