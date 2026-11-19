from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image as PdfImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def create_pdf_report(record, probability_rows, heatmap_image=None):
    buffer = BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    content = [
        Paragraph("Brain MRI model analysis report", styles["Title"]),
        Spacer(1, 12),
        Paragraph(
            "RESEARCH USE ONLY — This AI output is not a diagnosis and must not "
            "be used to make clinical decisions. Review the original scan with a "
            "qualified clinician.",
            styles["BodyText"],
        ),
        Spacer(1, 12),
        Paragraph(f"Patient ID: {escape(str(record['patient_id']))}", styles["BodyText"]),
        Paragraph(f"Timestamp (UTC): {escape(str(record['timestamp']))}", styles["BodyText"]),
        Paragraph(f"Model: {escape(str(record['model']))}", styles["BodyText"]),
        Paragraph(f"Predicted class: {escape(str(record['prediction']))}", styles["BodyText"]),
        Paragraph(f"Top-class probability: {record['confidence']:.1%}", styles["BodyText"]),
        Spacer(1, 12),
        Paragraph("Class probabilities", styles["Heading2"]),
    ]
    table = Table([["Class", "Probability"]] + probability_rows, colWidths=[280, 150])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#15354a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("PADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    content.extend([table, Spacer(1, 14)])
    if heatmap_image is not None:
        image_buffer = BytesIO()
        heatmap_image.save(image_buffer, format="PNG")
        image_buffer.seek(0)
        content.extend(
            [
                Paragraph("Grad-CAM visualization", styles["Heading2"]),
                PdfImage(image_buffer, width=300, height=300),
            ]
        )
    document.build(content)
    return buffer.getvalue()
