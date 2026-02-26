import io
import pandas as pd
from flask import request, jsonify, send_file, current_app
import joblib
import os
import csv


def get_models():
    """Lazy-load shared models from the parent app context."""
    model = current_app.config["model"]
    scaler = current_app.config["scaler"]
    label_encoders = current_app.config["label_encoders"]
    explainer = current_app.config.get("explainer")
    classify_risk = current_app.config["classify_risk"]
    return model, scaler, label_encoders, explainer, classify_risk


def batch_predict():
    """Batch Predict from CSV Upload
    ---
    consumes:
      - multipart/form-data
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: CSV file with loan application rows
    responses:
      200:
        description: CSV file with predictions
      400:
        description: No file or bad format
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files are accepted"}), 400

    model, scaler, label_encoders, explainer, classify_risk = get_models()

    try:
        df_in = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Could not parse CSV: {e}"}), 400

    required = ["Gender","Married","Dependents","Education","Self_Employed",
                "ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term",
                "Credit_History","Property_Area"]

    missing = [c for c in required if c not in df_in.columns]
    if missing:
        return jsonify({"error": f"Missing columns: {missing}"}), 400

    results = []
    for idx, row in df_in.iterrows():
        try:
            row_data = row[required].to_dict()
            df_row = pd.DataFrame([row_data])
            for col in df_row.columns:
                if col in label_encoders:
                    df_row[col] = label_encoders[col].transform(df_row[col].astype(str))
            df_scaled = scaler.transform(df_row)
            prob = float(model.predict_proba(df_scaled)[0][1])
            pred = int(model.predict(df_scaled)[0])
            risk = classify_risk(prob)

            # Top SHAP factor
            top_factor = ""
            if explainer:
                shap_vals = explainer.shap_values(df_scaled)
                vals = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
                feat_imp = dict(zip(df_row.columns, vals))
                top_factor = max(feat_imp, key=lambda k: abs(feat_imp[k]))

            results.append({
                **row_data,
                "Prediction": "Approved" if pred == 1 else "Rejected",
                "Probability": round(prob, 4),
                "Risk_Level": risk,
                "Top_Factor": top_factor
            })
        except Exception as e:
            results.append({**row.to_dict(), "Prediction": "ERROR", "Probability": 0, "Risk_Level": "N/A", "Top_Factor": str(e)})

    out = io.StringIO()
    if results:
        writer = csv.DictWriter(out, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    out.seek(0)
    return send_file(
        io.BytesIO(out.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="loan_predictions.csv"
    )
