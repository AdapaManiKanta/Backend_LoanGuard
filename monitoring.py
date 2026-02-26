"""
monitoring.py — Model Drift Detection + Email Alerts via Resend
Compares rolling 7-day accuracy against a baseline.
Sends a no-reply email via Resend API when drift is detected.
"""
import os
import requests
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

BASELINE_ACCURACY = float(os.getenv("BASELINE_ACCURACY", "0.82"))

_drift_alert_sent = False  # In-process guard: send at most once per server restart


def send_drift_alert(accuracy_7d: float, baseline: float) -> None:
    """
    Send a drift-detected alert via Resend API (no-reply email).
    Skipped silently if RESEND_API_KEY is not configured.
    """
    global _drift_alert_sent
    api_key = os.getenv("RESEND_API_KEY", "")
    alert_email = os.getenv("ALERT_EMAIL", "")
    if not api_key or not alert_email or _drift_alert_sent:
        return
    try:
        requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "from": "LoanGuard Alerts <onboarding@resend.dev>",
                "to": [alert_email],
                "subject": "⚠️ LoanGuard: Model Drift Detected",
                "html": (
                    f"""
                    <h2 style='color:#dc2626'>⚠️ Model Drift Detected</h2>
                    <p>The rolling 7-day accuracy has dropped significantly below baseline.</p>
                    <table style='border-collapse:collapse;font-family:monospace'>
                      <tr><td style='padding:4px 12px;font-weight:bold'>7-Day Accuracy</td>
                          <td style='padding:4px 12px;color:#dc2626'>{accuracy_7d:.2%}</td></tr>
                      <tr><td style='padding:4px 12px;font-weight:bold'>Baseline</td>
                          <td style='padding:4px 12px'>{baseline:.2%}</td></tr>
                      <tr><td style='padding:4px 12px;font-weight:bold'>Drop</td>
                          <td style='padding:4px 12px;color:#dc2626'>{(baseline - accuracy_7d):.2%}</td></tr>
                    </table>
                    <p style='margin-top:16px'>Please consider retraining the model via the Admin Panel.</p>
                    <p style='color:#6b7280;font-size:12px'>This alert is sent at most once per server session.</p>
                    """
                )
            },
            timeout=8
        )
        _drift_alert_sent = True
    except Exception:
        pass  # Never let email failure break monitoring


def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT")
    )


def get_drift_status():
    """
    GET /drift-status
    Computes rolling 7-day accuracy by comparing model predictions
    against a simple ground-truth proxy: applications with high income +
    credit history = expected Approved; otherwise Rejected.
    Returns drift_detected=True if accuracy drops >10% below baseline.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Last 7 days of predictions
        cursor.execute("""
            SELECT applicant_income, credit_history, prediction, probability
            FROM applications
            WHERE created_at >= NOW() - INTERVAL '7 days'
            ORDER BY created_at DESC
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        if not rows:
            return {
                "drift_detected": False,
                "accuracy_7d": None,
                "accuracy_baseline": BASELINE_ACCURACY,
                "sample_size": 0,
                "message": "No predictions in the last 7 days."
            }

        # Simple proxy ground truth
        correct = 0
        for row in rows:
            expected = 1 if (row["applicant_income"] > 3000 and row["credit_history"] == 1) else 0
            if row["prediction"] == expected:
                correct += 1

        accuracy_7d = round(correct / len(rows), 4)
        drift_detected = accuracy_7d < (BASELINE_ACCURACY - 0.10)

        if drift_detected:
            send_drift_alert(accuracy_7d, BASELINE_ACCURACY)

        return {
            "drift_detected": drift_detected,
            "accuracy_7d": accuracy_7d,
            "accuracy_baseline": BASELINE_ACCURACY,
            "sample_size": len(rows),
            "message": (
                "⚠️ Model drift detected — accuracy dropped significantly. Consider retraining."
                if drift_detected else
                "✅ Model performance is stable."
            )
        }

    except Exception as e:
        return {"drift_detected": False, "error": str(e)}
