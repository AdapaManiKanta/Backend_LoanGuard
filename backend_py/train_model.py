import pandas as pd
import joblib
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Set experiment name
mlflow.set_experiment("Loan_Risk_Prediction")

if not os.path.exists("models"):
    os.makedirs("models")

data = pd.read_csv("data/loan_data.csv")

if "Loan_ID" in data.columns:
    data.drop("Loan_ID", axis=1, inplace=True)

for col in data.select_dtypes(include="object").columns:
    data[col] = data[col].fillna(data[col].mode()[0])

for col in data.select_dtypes(include="number").columns:
    data[col] = data[col].fillna(data[col].median())

data["Loan_Status"] = data["Loan_Status"].map({"Y":1,"N":0})

label_encoders = {}
for col in data.select_dtypes(include="object").columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    # Model Hyperparameters
    params = {"max_iter": 1000, "C": 1.0, "solver": "lbfgs"}
    mlflow.log_params(params)

    model = LogisticRegression(**params)
    model.fit(X_train, y_train)

    # Predictions and Metrics
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # Save artifacts locally
    joblib.dump(model, "models/loan_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(label_encoders, "models/label_encoders.pkl")

    # Log model to MLflow
    mlflow.sklearn.log_model(model, "loan_risk_model")
    
    # SHAP Explainability
    import shap
    explainer = shap.LinearExplainer(model, X_train)
    joblib.dump(explainer, "models/shap_explainer.pkl")
    
    print(f"Model Trained. Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print("Logged to MLflow successfully. SHAP Explainer saved.")
