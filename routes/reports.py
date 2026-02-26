import io
from flask import send_file, jsonify
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT


ROYAL_BLUE = colors.HexColor("#1E40AF")
LIGHT_BLUE = colors.HexColor("#DBEAFE")
SLATE_50   = colors.HexColor("#F8FAFC")
SLATE_200  = colors.HexColor("#E2E8F0")
SLATE_500  = colors.HexColor("#64748B")
EMERALD    = colors.HexColor("#059669")
RED        = colors.HexColor("#DC2626")
AMBER      = colors.HexColor("#D97706")


def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"), database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"), password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT")
    )


def generate_report(app_id):
    """Generate PDF Report for a Single Application
    ---
    parameters:
      - name: app_id
        in: path
        type: integer
        required: true
    responses:
      200:
        description: PDF file
      404:
        description: Application not found
    """
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT * FROM applications WHERE id = %s", (app_id,))
        app = cursor.fetchone()
        cursor.close()
        conn.close()

        if not app:
            return jsonify({"error": "Application not found"}), 404

        app = dict(app)
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                rightMargin=2*cm, leftMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)

        styles = getSampleStyleSheet()

        # ── Styles ──────────────────────────────────────────────────────────
        brand_style   = ParagraphStyle("brand",   fontSize=24, fontName="Helvetica-Bold",
                                       leading=28, spaceAfter=4,  textColor=ROYAL_BLUE)
        tagline_style = ParagraphStyle("tagline", fontSize=10, textColor=SLATE_500, spaceAfter=18)
        section_style = ParagraphStyle("section", fontSize=12, fontName="Helvetica-Bold",
                                       spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#1E293B"))
        meta_style    = ParagraphStyle("meta",    fontSize=8,  textColor=SLATE_500,
                                       alignment=TA_RIGHT, spaceAfter=12)
        footer_style  = ParagraphStyle("footer",  fontSize=8,  textColor=SLATE_500, alignment=TA_CENTER)
        body_style    = styles["BodyText"]

        # ── Decision colours ─────────────────────────────────────────────────
        prediction = app.get("prediction")
        status     = app.get("status", "Pending")
        dec_color  = EMERALD if prediction == 1 else RED
        dec_text   = "\u2713 APPROVED" if prediction == 1 else "\u2717 REJECTED"

        status_color = {
            "Approved":     EMERALD,
            "Rejected":     RED,
            "Under Review": ROYAL_BLUE,
        }.get(status, AMBER)

        # ── Build story ──────────────────────────────────────────────────────
        story = [
            # Header brand block
            Paragraph("LoanGuard", brand_style),
            Paragraph("AI-Powered Loan Risk Assessment Platform", tagline_style),
            Paragraph(f"Generated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}  |  Report #LG-{app_id:05d}", meta_style),
            HRFlowable(width="100%", thickness=2, color=ROYAL_BLUE, spaceAfter=18),

            # Decision summary
            Paragraph("Decision Summary", section_style),
            Table(
                [[
                    Paragraph(
                        f'<font color="{dec_color.hexval() if hasattr(dec_color, "hexval") else "#059669"}" size="20"><b>{dec_text}</b></font>',
                        body_style
                    ),
                    Paragraph(
                        f'Risk Level: <b>{app.get("risk_level", "N/A")}</b><br/>'
                        f'Probability: <b>{round(float(app.get("probability", 0)) * 100, 1)}%</b><br/>'
                        f'Status: <b>{status}</b>',
                        body_style
                    )
                ]],
                colWidths=["42%", "58%"],
                style=TableStyle([
                    ("BACKGROUND", (0, 0), (0, 0), LIGHT_BLUE),
                    ("BACKGROUND", (1, 0), (1, 0), SLATE_50),
                    ("GRID",       (0, 0), (-1, -1), 0.5, SLATE_200),
                    ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING",  (0, 0), (-1, -1), 14),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 14),
                    ("TOPPADDING",   (0, 0), (-1, -1), 14),
                    ("BOTTOMPADDING",(0, 0), (-1, -1), 14),
                    ("ROUNDEDCORNERS", [6]),
                ])
            ),

            Spacer(1, 14),
            Paragraph("Applicant Details", section_style),
            Table(
                [
                    ["Field", "Value"],
                    ["Application ID",      f"#LG-{app.get('id'):05d}"],
                    ["Applicant Name",      app.get("applicant_name") or "—"],
                    ["Gender",              app.get("gender", "—")],
                    ["Married",             app.get("married", "—")],
                    ["Dependents",          str(app.get("dependents", "—"))],
                    ["Education",           app.get("education", "—")],
                    ["Self Employed",       app.get("self_employed", "—")],
                    ["Applicant Income",    f"Rs. {int(app.get('applicant_income', 0)):,}/mo"],
                    ["Co-applicant Income", f"Rs. {int(app.get('coapplicant_income', 0)):,}/mo"],
                    ["Loan Amount",         f"Rs. {int(app.get('loan_amount', 0)) * 1000:,}"],
                    ["Loan Term",           f"{app.get('loan_amount_term', '—')} months"],
                    ["Credit History",      "Yes" if app.get("credit_history") == 1 else "No"],
                    ["Property Area",       app.get("property_area", "—")],
                    ["Application Status",  status],
                ],
                colWidths=["45%", "55%"],
                style=TableStyle([
                    ("BACKGROUND",    (0, 0), (-1, 0), ROYAL_BLUE),
                    ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
                    ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE",      (0, 0), (-1, -1), 10),
                    ("ROWBACKGROUNDS",(0, 1), (-1, -1), [SLATE_50, colors.white]),
                    ("GRID",          (0, 0), (-1, -1), 0.5, SLATE_200),
                    ("LEFTPADDING",   (0, 0), (-1, -1), 12),
                    ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
                    ("TOPPADDING",    (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ])
            ),

            Spacer(1, 20),
            HRFlowable(width="100%", thickness=1, color=SLATE_200, spaceAfter=8),
            Paragraph(
                "This report is AI-generated for informational purposes only and does not constitute a formal loan offer. "
                "LoanGuard decisions are based on a machine learning model trained on historical data. "
                "Final approval is subject to the bank\u2019s discretion.",
                footer_style
            ),
            Spacer(1, 4),
            Paragraph(
                f"\u00a9 {datetime.now().year} LoanGuard MLOps \u00b7 AI-Powered Loan Risk Assessment",
                ParagraphStyle("brand_footer", fontSize=8, textColor=ROYAL_BLUE,
                               alignment=TA_CENTER, fontName="Helvetica-Bold")
            ),
        ]

        doc.build(story)
        buf.seek(0)
        return send_file(buf, mimetype="application/pdf", as_attachment=True,
                         download_name=f"LoanGuard_Report_{app_id:05d}.pdf")

    except Exception as e:
        return jsonify({"error": str(e)}), 500
