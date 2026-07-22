#!/usr/bin/env python3
"""Repair high-value heatmap annotation contrast in Supplementary Figure S3.

The archived vector figure was generated before the plotting threshold in
``audit_candidate_stability.py`` was corrected. This document-only repair
replaces white numeric annotations on the bright end of the viridis heatmap
with black annotations while preserving every displayed value and panel.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import fitz
from matplotlib import colormaps
from matplotlib import font_manager


NUMERIC_ANNOTATION = re.compile(r"^\d{1,3}\.\d$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input one-page Figure S3 PDF")
    parser.add_argument("output", type=Path, help="Output repaired PDF")
    parser.add_argument(
        "--threshold",
        type=float,
        default=80.0,
        help="Values above this threshold are redrawn in black (default: 80)",
    )
    return parser.parse_args()


def heatmap_spans(page: fitz.Page, threshold: float) -> list[dict[str, object]]:
    spans: list[dict[str, object]] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span["text"].strip()
                if not NUMERIC_ANNOTATION.fullmatch(text):
                    continue
                value = float(text)
                x0, y0, x1, y1 = span["bbox"]
                is_heatmap_cell = (
                    140.0 < x0 < 1510.0
                    and 45.0 < y0 < 545.0
                    and 6.0 <= float(span["size"]) <= 8.0
                )
                if is_heatmap_cell and value > threshold:
                    spans.append(
                        {
                            "text": text,
                            "value": value,
                            "bbox": fitz.Rect(x0, y0, x1, y1),
                            "origin": fitz.Point(*span["origin"]),
                            "fontsize": float(span["size"]),
                        }
                    )
    return spans


def repair(input_path: Path, output_path: Path, threshold: float) -> int:
    document = fitz.open(input_path)
    if document.page_count != 1:
        document.close()
        raise RuntimeError(
            f"Expected a one-page Figure S3 PDF, observed {document.page_count}"
        )

    page = document[0]
    spans = heatmap_spans(page, threshold)
    if not spans:
        document.close()
        raise RuntimeError("No high-value heatmap annotations were found")

    viridis = colormaps["viridis"]
    for span in spans:
        bbox = fitz.Rect(span["bbox"])
        patch = fitz.Rect(
            bbox.x0 - 1.2,
            bbox.y0 - 0.6,
            bbox.x1 + 1.2,
            bbox.y1 + 0.6,
        )
        background = tuple(
            float(channel) for channel in viridis(float(span["value"]) / 100.0)[:3]
        )
        page.add_redact_annot(patch, fill=background)

    page.apply_redactions(
        images=fitz.PDF_REDACT_IMAGE_NONE,
        graphics=fitz.PDF_REDACT_LINE_ART_NONE,
        text=fitz.PDF_REDACT_TEXT_REMOVE,
    )

    font_path = font_manager.findfont("DejaVu Sans", fallback_to_default=False)
    embedded_font = fitz.Font(fontfile=font_path)
    for span in spans:
        text = str(span["text"])
        fontsize = float(span["fontsize"])
        bbox = fitz.Rect(span["bbox"])
        width = embedded_font.text_length(text, fontsize=fontsize)
        x = bbox.x0 + (bbox.width - width) / 2.0
        page.insert_text(
            fitz.Point(x, float(span["origin"].y)),
            text,
            fontsize=fontsize,
            fontname="S3ContrastDejaVuSans",
            fontfile=font_path,
            color=(0.0, 0.0, 0.0),
            overlay=True,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document.subset_fonts()
    document.save(output_path, garbage=4, deflate=True)
    document.close()
    return len(spans)


def main() -> None:
    args = parse_args()
    repaired = repair(args.input, args.output, args.threshold)
    print(f"FIGURE S3 CONTRAST REPAIR: PASSED ({repaired} annotations)")


if __name__ == "__main__":
    main()
