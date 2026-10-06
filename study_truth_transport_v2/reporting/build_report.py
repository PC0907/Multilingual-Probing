#!/usr/bin/env python3
"""Render the exact fifteen-page Truth Transport research report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


INK = colors.HexColor("#17243A")
BLUE = colors.HexColor("#2457C5")
TEAL = colors.HexColor("#0B8585")
MINT = colors.HexColor("#EAF7F5")
PALE = colors.HexColor("#F2F5FA")
MID = colors.HexColor("#667085")
RULE = colors.HexColor("#CCD5E3")


EXPECTED_PAGES = 18  # 15 core pages + X14, translation robustness, v5 replication (v3-primary build)

def register_fonts() -> tuple[str, str]:
    regular = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    bold = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont("TruthReport", str(regular)))
        pdfmetrics.registerFont(TTFont("TruthReportBold", str(bold)))
        pdfmetrics.registerFontFamily(
            "TruthReport", normal="TruthReport", bold="TruthReportBold"
        )
        return "TruthReport", "TruthReportBold"
    return "Helvetica", "Helvetica-Bold"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--content", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data = json.loads(args.content.read_text(encoding="utf-8"))
    n_pages = len(data["pages"])
    if n_pages != EXPECTED_PAGES:
        raise ValueError(f"Expected exactly {EXPECTED_PAGES} authored pages, got {n_pages}")

    font, bold = register_fonts()
    styles = {
        "eyebrow": ParagraphStyle(
            "eyebrow", fontName=bold, fontSize=7.7, leading=9.5,
            textColor=TEAL, spaceAfter=7,
        ),
        "title": ParagraphStyle(
            "title", fontName=bold, fontSize=21.5, leading=25,
            textColor=INK, spaceAfter=11,
        ),
        "dek": ParagraphStyle(
            "dek", fontName=font, fontSize=11, leading=14.7,
            textColor=MID, spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", fontName=bold, fontSize=11.5, leading=14.5,
            textColor=BLUE, spaceBefore=6, spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body", fontName=font, fontSize=9.7, leading=13.1,
            textColor=INK, spaceAfter=7,
        ),
        "small": ParagraphStyle(
            "small", fontName=font, fontSize=8.1, leading=10.2,
            textColor=MID, spaceAfter=5,
        ),
        "reference": ParagraphStyle(
            "reference", fontName=font, fontSize=7.6, leading=9.5,
            textColor=INK, spaceAfter=4,
        ),
        "callout": ParagraphStyle(
            "callout", fontName=bold, fontSize=9.7, leading=13,
            textColor=INK,
        ),
        "cell": ParagraphStyle(
            "cell", fontName=font, fontSize=7.9, leading=9.8,
            textColor=INK,
        ),
        "headcell": ParagraphStyle(
            "headcell", fontName=bold, fontSize=7.8, leading=9.7,
            textColor=colors.white,
        ),
    }

    page_width = A4[0] - 80
    story = []
    heights = []

    for page_number, page in enumerate(data["pages"], 1):
        flow = [
            Paragraph(escape(page.get("eyebrow", "TRUTH TRANSPORT / RESEARCH REPORT")), styles["eyebrow"]),
            Paragraph(escape(page["title"]), styles["title"]),
        ]
        if page.get("dek"):
            flow.append(Paragraph(page["dek"], styles["dek"]))

        for element in page["elements"]:
            kind = element["type"]
            if kind in {"body", "small", "subtitle", "reference"}:
                flow.append(Paragraph(element["text"], styles[kind]))
            elif kind == "callout":
                box = Table(
                    [[Paragraph(element["text"], styles["callout"]) ]],
                    colWidths=[page_width],
                )
                box.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), MINT),
                    ("BOX", (0, 0), (-1, -1), 0.7, TEAL),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ]))
                flow.extend([box, Spacer(1, 9)])
            elif kind == "table":
                rows = []
                for row_index, row in enumerate(element["rows"]):
                    style = styles["headcell" if row_index == 0 else "cell"]
                    rows.append([Paragraph(escape(str(cell)), style) for cell in row])
                widths = element.get("widths", [1 / len(rows[0])] * len(rows[0]))
                table = Table(
                    rows,
                    colWidths=[page_width * value for value in widths],
                    hAlign="LEFT",
                    repeatRows=1,
                )
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), INK),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
                    ("LINEBELOW", (0, -1), (-1, -1), 0.4, RULE),
                ]))
                flow.extend([table, Spacer(1, 8)])
            elif kind == "figure":
                path = Path(element["path"])
                with PILImage.open(path) as source:
                    pixel_width, pixel_height = source.size
                draw_width = page_width * element.get("width_fraction", 1.0)
                draw_height = draw_width * pixel_height / pixel_width
                max_height = element.get("max_height", 360)
                if draw_height > max_height:
                    ratio = max_height / draw_height
                    draw_width *= ratio
                    draw_height *= ratio
                figure = Image(str(path), width=draw_width, height=draw_height)
                figure.hAlign = "CENTER"
                flow.append(figure)
                if element.get("caption"):
                    flow.append(Paragraph(element["caption"], styles["small"]))
            elif kind == "space":
                flow.append(Spacer(1, element.get("height", 6)))
            else:
                raise ValueError(f"Unsupported element type: {kind}")

        total_height = 0.0
        for item in flow:
            _, height = item.wrap(page_width, 760)
            total_height += height + item.getSpaceBefore() + item.getSpaceAfter()
        heights.append({"page": page_number, "estimated_height": total_height})
        if total_height > 745:
            raise ValueError(f"Page {page_number} is too tall: {total_height:.1f} points")
        story.extend(flow)
        if page_number < n_pages:
            story.append(PageBreak())

    if args.dry_run:
        print(json.dumps({"layout_only": True, "pages": heights}, indent=2))
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(args.output), pagesize=A4,
        leftMargin=40, rightMargin=40, topMargin=46, bottomMargin=40,
        title=data["title"], author="Multilingual probing research project",
        subject=data["subject"],
    )

    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(TEAL)
        canvas.setLineWidth(1.1)
        canvas.line(40, A4[1] - 29, A4[0] - 40, A4[1] - 29)
        canvas.setFont(font, 6.8)
        canvas.setFillColor(MID)
        canvas.drawString(40, 22, data["footer"])
        canvas.drawRightString(A4[0] - 40, 22, f"{doc.page} / {n_pages}")
        canvas.restoreState()

    document.build(story, onFirstPage=decorate, onLaterPages=decorate)
    actual_pages = len(PdfReader(str(args.output)).pages)
    if actual_pages != n_pages:
        raise AssertionError(f"Unexpected pagination: {actual_pages} pages")
    Path(str(args.output) + ".layout.json").write_text(
        json.dumps(heights, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output.resolve()), "pages": actual_pages}))


if __name__ == "__main__":
    main()
