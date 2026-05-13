"""PDF report generator using ReportLab."""
from __future__ import annotations
import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image,
)
from reportlab.lib.utils import ImageReader


NAVY = colors.HexColor("#0B1F3A")
GREY = colors.HexColor("#4B5563")
LIGHT = colors.HexColor("#F3F4F6")


def build_report(meta: dict, sections: list[dict]) -> bytes:
    """sections: [{'title': str, 'paragraphs': [str], 'table': [[...]] | None}]"""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("H1", parent=styles["Heading1"],
                              fontName="Helvetica-Bold", textColor=NAVY, fontSize=20, spaceAfter=12))
    styles.add(ParagraphStyle("H2", parent=styles["Heading2"],
                              fontName="Helvetica-Bold", textColor=NAVY, fontSize=13, spaceAfter=8, spaceBefore=14))
    styles.add(ParagraphStyle("Body", parent=styles["Normal"],
                              fontName="Helvetica", fontSize=10, leading=14, textColor=GREY))
    styles.add(ParagraphStyle("Small", parent=styles["Normal"],
                              fontName="Helvetica", fontSize=8, textColor=GREY))

    flow = []
    flow.append(Paragraph("SovereignIQ — Risk Intelligence Report", styles["H1"]))
    flow.append(Paragraph(
        f"Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} · "
        f"Country focus: {meta.get('country','—')} · Indicator: {meta.get('indicator','—')}",
        styles["Small"],
    ))
    flow.append(Spacer(1, 12))

    for sec in sections:
        flow.append(Paragraph(sec["title"], styles["H2"]))
        for p in sec.get("paragraphs", []):
            flow.append(Paragraph(p, styles["Body"]))
            flow.append(Spacer(1, 4))
        # Embed chart images (PNG bytes) — sized to page width
        for img_bytes in sec.get("images", []) or []:
            try:
                img = Image(io.BytesIO(img_bytes), width=16 * cm, height=8.5 * cm)
                flow.append(Spacer(1, 4))
                flow.append(img)
                flow.append(Spacer(1, 6))
            except Exception:
                pass
        if sec.get("table"):
            tbl = Table(sec["table"], hAlign="LEFT")
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D1D5DB")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            flow.append(tbl)
        flow.append(Spacer(1, 6))

    flow.append(Spacer(1, 20))
    flow.append(Paragraph(
        "Data sources: World Bank Open Data API (annual macro indicators) and "
        "European Central Bank via Frankfurter (daily FX). All values are fetched "
        "in real time at report generation. This report is for analytical purposes "
        "only and does not constitute investment advice.",
        styles["Small"],
    ))

    doc.build(flow)
    return buf.getvalue()
