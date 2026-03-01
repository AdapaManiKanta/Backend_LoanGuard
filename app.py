from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from datetime import datetime, timedelta
from flask_cors import CORS
from flasgger import Swagger
from pydantic import BaseModel, Field, ValidationError
from typing import Literal
import jwt
from functools import wraps
import glob
import subprocess
import sys
import json

# ─────────────────────────────────────────────
# Load Environment Variables
# ─────────────────────────────────────────────
load_dotenv()

JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    raise Exception("JWT_SECRET is not set in environment variables!")

# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
Swagger(app)

# ─────────────────────────────────────────────
# Database Connection
# ─────────────────────────────────────────────
def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT")
    )

# ─────────────────────────────────────────────
# JWT Decorator
# ─────────────────────────────────────────────
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if not token:
            return jsonify({"error": "Token missing"}), 401
        try:
            decoded = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            request.current_user = decoded["sub"]
            request.current_role = decoded.get("role")
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
        return f(*args, **kwargs)
    return decorated

# ─────────────────────────────────────────────
# Risk Classification Logic
# ─────────────────────────────────────────────
def classify_risk(prob):
    if prob >= 0.7:
        return "Low Risk"
    elif prob >= 0.4:
        return "Medium Risk"
    else:
        return "High Risk"

# ─────────────────────────────────────────────
# Pydantic Validation Schema
# ─────────────────────────────────────────────
class LoanApplication(BaseModel):
    Gender: Literal["Male", "Female"]
    Married: Literal["Yes", "No"]
    Dependents: Literal["0", "1", "2", "3+"]
    Education: Literal["Graduate", "Not Graduate"]
    Self_Employed: Literal["Yes", "No"]
    ApplicantIncome: int = Field(..., gt=0)
    CoapplicantIncome: int = Field(..., ge=0)
    LoanAmount: int = Field(..., gt=0)
    Loan_Amount_Term: int = Field(..., gt=0)
    Credit_History: int = Field(..., ge=0, le=1)
    Property_Area: Literal["Urban", "Semiurban", "Rural"]

# ─────────────────────────────────────────────
# Load ML Assets
# ─────────────────────────────────────────────
model = joblib.load("models/loan_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

try:
    explainer = joblib.load("models/shap_explainer.pkl")
except:
    explainer = None

# ─────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────
@app.route("/")
def health():
    return jsonify({"status": "LoanGuard API Running"})

# ─────────────────────────────────────────────
# Prediction Endpoint
# ─────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    try:
        raw = request.json
        if not raw:
            return jsonify({"error": "No input provided"}), 400

        try:
            validated = LoanApplication.model_validate(raw)
        except ValidationError as ve:
            return jsonify({"error": ve.errors()}), 422

        data = validated.model_dump()

        real_loan_amount = data["LoanAmount"]
        data["LoanAmount"] = real_loan_amount / 1000

        df = pd.DataFrame([data])

        for col in df.columns:
            if col in label_encoders:
                df[col] = label_encoders[col].transform(df[col])

        df_scaled = scaler.transform(df)

        prediction = int(model.predict(df_scaled)[0])
        probability = float(model.predict_proba(df_scaled)[0][1])
        risk_level = classify_risk(probability)

        return jsonify({
            "prediction": prediction,
            "probability": round(probability, 4),
            "risk_level": risk_level
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─────────────────────────────────────────────
# Login Endpoint
# ─────────────────────────────────────────────
@app.route("/login", methods=["POST"])
def login():
    creds = request.json or {}
    username = creds.get("username")
    password = creds.get("password")

    if username == os.getenv("ADMIN_USER") and password == os.getenv("ADMIN_PASS"):
        token = jwt.encode({
            "sub": username,
            "role": "ADMIN",
            "exp": datetime.utcnow() + timedelta(hours=8)
        }, JWT_SECRET, algorithm="HS256")

        return jsonify({"token": token})

    return jsonify({"error": "Invalid credentials"}), 401

# ─────────────────────────────────────────────
# Applications (Protected)
# ─────────────────────────────────────────────
@app.route("/applications", methods=["GET"])
@token_required
def get_applications():
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT * FROM applications ORDER BY id DESC")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return jsonify(rows)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─────────────────────────────────────────────
# Model Info Endpoint
# ─────────────────────────────────────────────
@app.route("/admin/model-info", methods=["GET"])
@token_required
def model_info():
    try:
        meta_files = glob.glob("models/model_meta.json")
        if meta_files:
            with open(meta_files[0]) as f:
                return jsonify(json.load(f))

        mtime = os.path.getmtime("models/loan_model.pkl")
        return jsonify({
            "version": "1.0",
            "trained_at": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M"),
            "accuracy": None
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─────────────────────────────────────────────
# Production Entry
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
