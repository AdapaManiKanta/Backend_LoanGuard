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
from pydantic import BaseModel, Field, validator, ValidationError
from typing import Literal
import jwt
from functools import wraps

# Load environment variables
load_dotenv()

JWT_SECRET = os.getenv("JWT_SECRET", "loanguard-dev-secret-change-in-prod")

app = Flask(__name__)
CORS(app)
Swagger(app)

# --- JWT Auth Helper ---
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if not token:
            return jsonify({"error": "Token is missing"}), 401
        try:
            jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
        return f(*args, **kwargs)
    return decorated

# --- Input Validation Schema ---
class LoanApplication(BaseModel):
    ApplicantName: str = Field("", description="Full name of the applicant")
    Gender: Literal["Male", "Female"]
    Married: Literal["Yes", "No"]
    Dependents: Literal["0", "1", "2", "3+"]
    Education: Literal["Graduate", "Not Graduate"]
    Self_Employed: Literal["Yes", "No"]
    ApplicantIncome: int = Field(..., gt=0, description="Monthly income in ₹")
    CoapplicantIncome: int = Field(..., ge=0)
    LoanAmount: int = Field(..., gt=0, description="Loan amount in real ₹ (divided by 1000 internally before ML model)")
    Loan_Amount_Term: int = Field(..., gt=0)
    Credit_History: int = Field(..., ge=0, le=1, description="0 or 1")
    Property_Area: Literal["Urban", "Semiurban", "Rural"]

    @validator("Credit_History", pre=True)
    def coerce_credit_history(cls, v):
        """Accept '0', '1', 0, or 1 — all valid."""
        return int(v)

# Load ML model
model = joblib.load("models/loan_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")
try:
    explainer = joblib.load("models/shap_explainer.pkl")
except:
    explainer = None

# Share models via app config (used by route modules)
app.config["model"] = model
app.config["scaler"] = scaler
app.config["label_encoders"] = label_encoders
app.config["explainer"] = explainer
app.config["classify_risk"] = None  # set after classify_risk is defined

def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT")
    )

def classify_risk(prob):
    """prob is P(Approved). Invert for risk rating."""
    if prob >= 0.7:
        return "Low Risk"     # Very likely approved = low risk
    elif prob >= 0.4:
        return "Medium Risk"  # Borderline
    else:
        return "High Risk"    # Very likely rejected = high risk

@app.route("/")
def home():
    """Health Check Endpoint
    ---
    responses:
      200:
        description: Returns a success message
    """
    return jsonify({"message": "Loan Risk API Running"})

@app.route("/check-eligibility", methods=["POST"])
def check_eligibility():
    """Public Eligibility Check — no auth required
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            Gender: {type: string, example: "Male"}
            Married: {type: string, example: "Yes"}
            Dependents: {type: string, example: "0"}
            Education: {type: string, example: "Graduate"}
            Self_Employed: {type: string, example: "No"}
            ApplicantIncome: {type: integer, example: 5000}
            CoapplicantIncome: {type: integer, example: 0}
            LoanAmount: {type: integer, example: 120}
            Loan_Amount_Term: {type: integer, example: 360}
            Credit_History: {type: integer, example: 1}
            Property_Area: {type: string, example: "Urban"}
    responses:
      200:
        description: Eligibility result with approval tips
    """
    try:
        raw = request.json or {}
        try:
            app_input = LoanApplication.model_validate(raw)
        except ValidationError as ve:
            errors = [{"field": e["loc"][0], "msg": e["msg"]} for e in ve.errors()]
            return jsonify({"error": "Validation failed", "details": errors}), 422

        data = app_input.model_dump()
        applicant_name = data.pop("ApplicantName", "")
        # Accept real ₹ LoanAmount; divide by 1000 for ML model (trained on ₹ thousands)
        real_loan_amount = data["LoanAmount"]
        data["LoanAmount"] = real_loan_amount / 1000
        df = pd.DataFrame([data])

        for col in df.columns:
            if col in label_encoders:
                df[col] = label_encoders[col].transform(df[col])

        df_scaled = scaler.transform(df)
        probability = float(model.predict_proba(df_scaled)[0][1])
        prediction = int(model.predict(df_scaled)[0])
        risk_level = classify_risk(probability)

        # Build human-readable tips from SHAP
        tips = []
        explanation = {}
        if explainer:
            shap_values = explainer.shap_values(df_scaled)
            vals = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
            feat_imp = dict(zip(df.columns, vals))
            sorted_feats = sorted(feat_imp.items(), key=lambda x: abs(x[1]), reverse=True)
            explanation = {k: round(float(v), 4) for k, v in sorted_feats[:5]}

            # Generate plain-language tips (INR context)
            tip_map = {
                "Credit_History": "🏦 Credit History is the #1 factor. Clear any pending EMIs or loan defaults to improve your CIBIL score.",
                "LoanAmount": "💰 Requesting a lower loan amount improves approval. Adding a co-applicant reduces per-person risk.",
                "ApplicantIncome": "📈 A higher monthly income strongly helps. Consider showing salary slips, rental income, or freelance earnings.",
                "CoapplicantIncome": "👥 Adding a working co-applicant (e.g., spouse) with ₹ income can significantly boost eligibility.",
                "Loan_Amount_Term": "📅 Choosing a longer repayment term (e.g., 360 months = 30 yrs) reduces monthly EMI and improves approval odds.",
            }
            for feat, impact in sorted_feats[:3]:
                if impact > 0 and feat in tip_map:  # positive shap = hurts approval
                    tips.append(tip_map[feat])

        # EMI estimate at 10% p.a. (using the real ₹ loan amount directly)
        p = real_loan_amount
        n = data["Loan_Amount_Term"]
        r = 10 / (12 * 100)
        emi = round(p * r * (1 + r)**n / ((1 + r)**n - 1), 2) if n > 0 else 0

        return jsonify({
            "prediction": prediction,
            "probability": round(probability, 4),
            "risk_level": risk_level,
            "explanation": explanation,
            "improvement_tips": tips,
            "estimated_emi": emi
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    """Predict Loan Risk
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            Gender: {type: string, example: "Male"}
            Married: {type: string, example: "Yes"}
            Dependents: {type: string, example: "0"}
            Education: {type: string, example: "Graduate"}
            Self_Employed: {type: string, example: "No"}
            ApplicantIncome: {type: integer, example: 5000}
            CoapplicantIncome: {type: integer, example: 0}
            LoanAmount: {type: integer, example: 120}
            Loan_Amount_Term: {type: integer, example: 360}
            Credit_History: {type: integer, example: 1}
            Property_Area: {type: string, example: "Urban"}
    responses:
      200:
        description: Prediction results with SHAP values
    """
    try:
        raw = request.json
        if not raw:
            return jsonify({"error": "No input provided"}), 400

        # Validate input with Pydantic
        try:
            app_input = LoanApplication.model_validate(raw)
        except ValidationError as ve:
            errors = [{"field": e["loc"][0], "msg": e["msg"]} for e in ve.errors()]
            return jsonify({"error": "Validation failed", "details": errors}), 422

        data = app_input.model_dump()
        applicant_name = data.pop("ApplicantName", "")
        input_data_for_db = data.copy()
        # Convert real ₹ LoanAmount → thousands for the ML model (trained in ₹ thousands)
        real_loan_amount = data["LoanAmount"]
        data["LoanAmount"] = real_loan_amount / 1000
        df = pd.DataFrame([data])

        # Encode categorical features
        for col in df.columns:
            if col in label_encoders:
                df[col] = label_encoders[col].transform(df[col])

        df_scaled = scaler.transform(df)

        prediction = int(model.predict(df_scaled)[0])
        probability = float(model.predict_proba(df_scaled)[0][1])
        risk_level = classify_risk(probability)

        # SHAP Explanation
        explanation = {}
        if explainer:
            shap_values = explainer.shap_values(df_scaled)
            # For linear explainer, it's a single array or list of arrays depending on model type
            if isinstance(shap_values, list):
                vals = shap_values[1][0] # Class 1 (Approved)
            else:
                vals = shap_values[0]
            
            # Map values back to features
            feat_imp = dict(zip(df.columns, vals))
            # Get top 3 positive and top 3 negative contributors
            sorted_feats = sorted(feat_imp.items(), key=lambda x: abs(x[1]), reverse=True)
            explanation = {k: round(float(v), 4) for k, v in sorted_feats[:5]}

        # Save to Supabase
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO applications (
                applicant_name, gender, married, dependents, education, self_employed,
                applicant_income, coapplicant_income, loan_amount,
                loan_term, credit_history, property_area,
                prediction, probability, risk_level, created_at
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            applicant_name,
            input_data_for_db.get("Gender"),
            input_data_for_db.get("Married"),
            input_data_for_db.get("Dependents"),
            input_data_for_db.get("Education"),
            input_data_for_db.get("Self_Employed"),
            int(input_data_for_db.get("ApplicantIncome", 0)),
            int(input_data_for_db.get("CoapplicantIncome", 0)),
            int(input_data_for_db.get("LoanAmount", 0)),
            int(input_data_for_db.get("Loan_Amount_Term", 0)),
            int(input_data_for_db.get("Credit_History", 0)),
            input_data_for_db.get("Property_Area"),
            prediction,
            probability,
            risk_level,
            datetime.now()
        ))
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({
            "prediction": prediction,
            "probability": round(probability, 4),
            "risk_level": risk_level,
            "explanation": explanation
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/login", methods=["POST"])
def login():
    """Admin Login - Get JWT Token
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required: [username, password]
          properties:
            username: {type: string, example: "admin"}
            password: {type: string, example: "password"}
    responses:
      200:
        description: JWT access token
      401:
        description: Invalid credentials
    """
    from auth import USERS
    creds = request.json or {}
    username = creds.get("username", "")
    password = creds.get("password", "")
    user = USERS.get(username)
    if user and user["password"] == password:
        token = jwt.encode(
            {
                "sub": username,
                "role": user["role"],
                "name": user["name"],
                "exp": datetime.utcnow() + timedelta(hours=8)
            },
            JWT_SECRET,
            algorithm="HS256"
        )
        # Audit log
        try:
            _audit_log("LOGIN", username, {"role": user["role"]})
        except Exception:
            pass
        return jsonify({"token": token, "role": user["role"], "name": user["name"]})
    return jsonify({"error": "Invalid credentials"}), 401

@app.route("/applications", methods=["GET"])
@token_required
def get_applications():
    """Get All Applications
    ---
    responses:
      200:
        description: List of applications
    """
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT * FROM applications ORDER BY id DESC")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(rows)

@app.route("/stats", methods=["GET"])
@token_required
def stats():
    """Get Performance Statistics
    ---
    responses:
      200:
        description: Statistics about applications
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM applications")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM applications WHERE prediction = 1")
    approved = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM applications WHERE prediction = 0")
    rejected = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    return jsonify({
        "total_applications": total,
        "approved": approved,
        "rejected": rejected
    })

# --- RBAC, Monitoring, Audit routes ---
from auth import token_required as _tr, role_required, get_admin_users, get_me, get_connection as _ac_conn
from monitoring import get_drift_status
import subprocess
import sys

def _audit_log(event_type, username, details):
    """Append-only audit log insert."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO audit_log (event_type, username, details, created_at)
            VALUES (%s, %s, %s::jsonb, %s)
        """, (event_type, username, __import__('json').dumps(details), datetime.now()))
        conn.commit()
        cur.close()
        conn.close()
    except Exception:
        pass  # audit never blocks the main flow

@app.route("/me", methods=["GET"])
@token_required
def me():
    return get_me()

@app.route("/admin/users", methods=["GET"])
@role_required("ADMIN")
def admin_users():
    return get_admin_users()

@app.route("/drift-status", methods=["GET"])
@token_required
def drift_status():
    return jsonify(get_drift_status())

@app.route("/audit", methods=["GET"])
@role_required("ADMIN")
def audit_log_view():
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT * FROM audit_log ORDER BY created_at DESC LIMIT 200
        """)
        rows = list(cursor.fetchall())
        cursor.close()
        conn.close()
        return jsonify(rows)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/admin/retrain", methods=["POST"])
@role_required("ADMIN")
def retrain_model():
    """Trigger model retraining — ADMIN only.
    ---
    responses:
      200:
        description: Retraining result with accuracy and F1
    """
    try:
        script_path = os.path.join(os.path.dirname(__file__), "train_model.py")
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=120,
            cwd=os.path.dirname(__file__)
        )
        output = result.stdout + result.stderr
        # Parse accuracy and F1 from the last print line
        accuracy, f1 = None, None
        for line in output.splitlines():
            if "Accuracy:" in line and "F1:" in line:
                try:
                    parts = line.split(",")
                    accuracy = float(parts[0].split(":")[1].strip())
                    f1 = float(parts[1].split(":")[1].strip())
                except Exception:
                    pass
        success = result.returncode == 0
        _audit_log("RETRAIN", request.current_user, {"success": success, "accuracy": accuracy, "f1": f1})
        return jsonify({"success": success, "accuracy": accuracy, "f1": f1, "output": output})
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Retraining timed out after 120s"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/applications/<int:app_id>/status", methods=["PATCH"])
@role_required("MANAGER", "ADMIN")
def update_application_status(app_id):
    """Update application status — MANAGER or ADMIN only.
    ---
    parameters:
      - name: app_id
        in: path
        type: integer
        required: true
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            status:
              type: string
              example: "Under Review"
    responses:
      200:
        description: Status updated
    """
    valid_statuses = {"Pending", "Under Review", "Approved", "Rejected"}
    body = request.json or {}
    new_status = body.get("status", "")
    if new_status not in valid_statuses:
        return jsonify({"error": f"Invalid status. Must be one of: {', '.join(valid_statuses)}"}), 400
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE applications SET status = %s WHERE id = %s",
            (new_status, app_id)
        )
        if cursor.rowcount == 0:
            cursor.close(); conn.close()
            return jsonify({"error": "Application not found"}), 404
        conn.commit()
        cursor.close()
        conn.close()
        _audit_log("STATUS_UPDATE", request.current_user, {"app_id": app_id, "new_status": new_status})
        return jsonify({"success": True, "id": app_id, "status": new_status})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── JWT Refresh ──────────────────────────────────────────────────────────────
@app.route("/refresh", methods=["POST"])
def refresh_token():
    """Refresh a valid JWT token
    ---
    responses:
      200:
        description: New token
      401:
        description: Token invalid or expired
    """
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not token:
        return jsonify({"error": "Token missing"}), 401
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        new_token = jwt.encode(
            {"sub": payload["sub"], "role": payload["role"],
             "exp": datetime.utcnow() + timedelta(hours=1)},
            JWT_SECRET, algorithm="HS256"
        )
        return jsonify({"token": new_token})
    except jwt.ExpiredSignatureError:
        return jsonify({"error": "Token expired — please log in again"}), 401
    except jwt.InvalidTokenError:
        return jsonify({"error": "Invalid token"}), 401

# ── Admin Model Info ──────────────────────────────────────────────────────────
@app.route("/admin/model-info", methods=["GET"])
@token_required
def get_model_info():
    """Get current ML model metadata
    ---
    responses:
      200:
        description: Model version, training date, accuracy
    """
    import glob
    try:
        meta_files = glob.glob("models/model_meta.json")
        if meta_files:
            import json
            with open(meta_files[0]) as f:
                meta = json.load(f)
            return jsonify(meta)
        # Fallback: read model file modification time
        mtime = os.path.getmtime("models/loan_model.pkl")
        from datetime import datetime as dt
        return jsonify({
            "version": "1.0",
            "trained_at": dt.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M"),
            "accuracy": None,
            "note": "Run retrain to get full metadata"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Register modular routes
from routes.batch import batch_predict
from routes.analytics import (
    get_trends, get_income_bracket, get_risk_distribution,
    get_loan_amount_distribution, get_property_area_stats
)
from routes.reports import generate_report

app.config["classify_risk"] = classify_risk

app.add_url_rule("/batch-predict", "batch_predict", token_required(batch_predict), methods=["POST"])
app.add_url_rule("/analytics/trends", "analytics_trends", token_required(get_trends), methods=["GET"])
app.add_url_rule("/analytics/income-bracket", "analytics_income", token_required(get_income_bracket), methods=["GET"])
app.add_url_rule("/analytics/risk", "analytics_risk", token_required(get_risk_distribution), methods=["GET"])
app.add_url_rule("/analytics/loan-distribution", "analytics_loan", token_required(get_loan_amount_distribution), methods=["GET"])
app.add_url_rule("/analytics/property-area", "analytics_area", token_required(get_property_area_stats), methods=["GET"])
app.add_url_rule("/report/<int:app_id>", "report", token_required(generate_report), methods=["GET"])

if __name__ == "__main__":
    app.run(debug=True)
