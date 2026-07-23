#!/usr/bin/env python3
"""Enlarge text in the frozen vector figures without changing plotted data.

The major-revision repository intentionally keeps the final, audited PDF
figures even when the large upstream datasets are not redistributed.  This
post-processing step replaces only PDF text objects, preserves every raster
and vector data layer, removes redundant footers from main Figures 2 and 5,
and keeps enlarged Figure 6B row labels outside the plotted heatmap.

The operation is idempotent: processed PDFs carry a metadata marker and are
skipped on later runs unless ``--force`` is supplied.
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = (
    ROOT
    if (ROOT / "manuscript" / "figures").is_dir()
    else ROOT / "revision" / "rework_20260720"
)
MARKER = "figure-font-qa-v1"

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
FONT_FILES = {
    "regular": FONT_DIR / "DejaVuSans.ttf",
    "bold": FONT_DIR / "DejaVuSans-Bold.ttf",
    "italic": Path("/usr/share/fonts/opentype/urw-base35/NimbusSans-Italic.otf"),
    "bolditalic": Path("/usr/share/fonts/opentype/urw-base35/NimbusSans-BoldItalic.otf"),
}


@dataclass(frozen=True)
class FigureRule:
    targets: dict[float, float]
    remove_prefixes: tuple[str, ...] = ()
    crop_bottom: float = 0.0
    preserve_texts: tuple[str, ...] = ()
    right_anchor_prefixes: tuple[str, ...] = ()


RULES: dict[str, FigureRule] = {
    # Main Figure 1: the 176 heatmap values are the most important repair.
    "manuscript/figures/screening_landscape_hdac_enrichment.pdf": FigureRule(
        {5.5: 12.0, 7.5: 9.5, 8.0: 10.5, 9.0: 11.0, 10.0: 11.5, 12.0: 12.5, 16.0: 18.0}
    ),
    # Main Figure 2: compensate for the unusually wide 22-inch source canvas.
    "manuscript/figures/measured_reversal_official_conditions.pdf": FigureRule(
        {6.5: 15.0, 9.0: 11.0, 10.0: 11.5, 12.0: 14.0, 16.0: 18.5},
        remove_prefixes=("Official siginfo is authoritative.",),
        crop_bottom=18.0,
    ),
    "manuscript/figures/predicted_measured_hiq_two_way.pdf": FigureRule(
        {8.0: 9.5, 8.5: 10.0, 10.0: 11.0, 12.0: 13.0, 15.0: 16.5}
    ),
    "manuscript/figures/candidate_stability_primary.pdf": FigureRule(
        {8.0: 10.0, 10.0: 11.5, 12.0: 13.5, 16.0: 18.0}
    ),
    # Main Figure 5: the footer repeats methods/caption information and becomes
    # unreadable after journal-width scaling, so it is intentionally removed.
    "manuscript/figures/structural_neighbor_audit.pdf": FigureRule(
        {8.0: 11.0, 9.0: 12.0, 10.0: 12.0, 12.0: 14.0, 16.0: 19.0},
        remove_prefixes=("Morgan radius 2, 2048-bit fingerprints with chirality.",),
        crop_bottom=24.0,
    ),
    "manuscript/figures/candidate_gene_overlap_heatmap.pdf": FigureRule(
        {8.0: 12.0, 10.0: 11.5, 12.0: 14.0}
    ),
    "manuscript/figures/reversal_pathway_enrichment_heatmap.pdf": FigureRule(
        {7.0: 7.5, 8.0: 10.0, 10.0: 11.5, 12.0: 14.0},
        right_anchor_prefixes=("GO:BP |", "KEGG |", "REAC |"),
    ),
    "manuscript/figures/primary_candidate_physical_networks.pdf": FigureRule(
        {9.0: 10.5, 10.0: 11.5, 13.0: 14.0, 15.0: 17.0}
    ),
    "manuscript/figures/depmap_hdac_overall_distributions.pdf": FigureRule(
        {10.0: 11.5, 12.0: 13.5}
    ),
    "manuscript/figures/depmap_hdac_gene_correlation.pdf": FigureRule(
        {10.0: 11.5, 12.0: 13.5}
    ),
    "manuscript/figures/redocking_protocol_control.pdf": FigureRule(
        {10.0: 11.5, 12.0: 13.5}
    ),
    "manuscript/figures/controlled_docking_panel_4lxz.pdf": FigureRule(
        {9.0: 9.5, 10.0: 11.5, 12.0: 13.5}
    ),
    "manuscript/figures/docking_receptor_sensitivity_4bkx.pdf": FigureRule(
        {9.0: 9.5, 10.0: 11.5, 12.0: 13.5}
    ),
    # Supplementary figures S1--S6.
    "supplement/figures/predicted_measured_seed_variability.pdf": FigureRule(
        {10.0: 10.5, 12.0: 13.0, 14.0: 15.5}
    ),
    "supplement/figures/measured_condition_sensitivity.pdf": FigureRule(
        {7.0: 8.0, 9.0: 9.5, 10.0: 11.0, 12.0: 13.0, 14.0: 15.5}
    ),
    "supplement/figures/candidate_stability_expanded_72.pdf": FigureRule(
        {7.0: 11.0, 9.0: 10.0, 10.0: 11.5, 12.0: 13.0, 16.0: 18.0}
    ),
    "supplement/figures/supplementary_expanded_physical_networks.pdf": FigureRule(
        {6.3: 7.2, 9.0: 10.0, 10.0: 11.0, 13.0: 14.0, 14.0: 16.0}
    ),
    "supplement/figures/depmap_hdac_lineage_heatmap.pdf": FigureRule(
        {7.5: 9.5, 8.0: 9.5, 10.0: 11.5, 12.0: 14.0},
        preserve_texts=(
            "Median HDAC gene effect by OncoTree lineage (n ",
            " 10)",
        ),
    ),
    "supplement/figures/redocking_pose_diagnostics.pdf": FigureRule(
        {10.0: 11.0, 12.0: 13.0, 14.0: 15.5},
        preserve_texts=(
            "Score does not uniquely determine pose recovery",
            "Zn proximity does not uniquely determine pose recovery",
            "Symmetry-corrected crystal RMSD (Å)",
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Optional separate output tree. Omit to replace source PDFs in place.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def target_size(size: float, rule: FigureRule, text: str) -> float:
    for observed, target in rule.targets.items():
        if abs(size - observed) <= 0.08:
            result = max(size, target)
            if text == "Tianeptinaline_or_BG-1010":
                result = min(result, max(size, 10.5))
            return result
    return size


def font_key(font_name: str, text: str, original_size: float) -> str:
    lowered = font_name.lower()
    bold = "bold" in lowered
    italic = "italic" in lowered or "oblique" in lowered
    if original_size >= 11.5 and (
        re.match(r"^[A-C](?:\s|$)", text)
        or original_size >= 14
    ):
        bold = True
    if bold and italic:
        return "bolditalic"
    if bold:
        return "bold"
    if italic:
        return "italic"
    return "regular"


def repaired_text(text: str) -> str:
    # Type-3 extraction omits rho in the four correlation-summary lines.
    if text.startswith("Spearman ="):
        return text.replace("Spearman =", "Spearman ρ=", 1)
    # The original source intentionally spells rho out in this colorbar label;
    # its Unicode minus is not returned by PDF text extraction.
    if text == "rho; higher = stronger)":
        return "−rho; higher = stronger)"
    if text == "A  HiQ measured Spearman reversal, aggregated condition ":
        return "A  HiQ measured Spearman reversal, aggregated condition → official cell line"
    return text


def union_rect(spans: list[dict[str, object]]) -> fitz.Rect:
    rectangle = fitz.Rect(spans[0]["bbox"])
    for span in spans[1:]:
        rectangle |= fitz.Rect(span["bbox"])
    return rectangle


def line_items(page: fitz.Page, rule: FigureRule) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = [span for span in line.get("spans", []) if span["text"].strip()]
            if not spans:
                continue
            # Ghostscript split the bold titles in Figure 1 into one span per
            # word. Recombine only title-sized, same-style runs; heatmap cells
            # and mathematical fragments remain independent.
            same_size = max(float(span["size"]) for span in spans) - min(
                float(span["size"]) for span in spans
            ) <= 0.08
            if len(spans) > 1 and same_size:
                text = " ".join(str(span["text"]).strip() for span in spans)
                source_spans = [
                    {
                        **spans[0],
                        "text": text,
                        "bbox": tuple(union_rect(spans)),
                    }
                ]
            else:
                source_spans = spans
            for span in source_spans:
                text = str(span["text"])
                if text in rule.preserve_texts:
                    # Preserve titles whose existing 12-pt size is already
                    # readable but whose extracted pieces or adjacent panels
                    # do not leave enough room for safe enlargement.
                    continue
                remove = any(text.startswith(prefix) for prefix in rule.remove_prefixes)
                if text == " official cell line":
                    remove = True
                old_size = float(span["size"])
                new_size = target_size(old_size, rule, text)
                if not remove and new_size <= old_size + 0.05:
                    continue
                rectangle = fitz.Rect(span["bbox"])
                if text == "A  HiQ measured Spearman reversal, aggregated condition ":
                    rectangle.x1 += 118.0
                items.append(
                    {
                        "text": repaired_text(text),
                        "old_text": text,
                        "rect": rectangle,
                        "origin": fitz.Point(span["origin"]),
                        "direction": fitz.Point(line["dir"]),
                        "color": int(span["color"]),
                        "font": str(span["font"]),
                        "old_size": old_size,
                        "new_size": new_size,
                        "remove": remove,
                        "right_anchor": any(
                            text.startswith(prefix)
                            for prefix in rule.right_anchor_prefixes
                        ),
                    }
                )
    return items


def is_left_anchored(text: str, rect: fitz.Rect, page: fitz.Page) -> bool:
    if re.match(r"^[A-C](?:\s|$)", text):
        return True
    if rect.x0 > page.rect.width * 0.70 and not re.fullmatch(r"[-−]?\d+(?:\.\d+)?", text):
        return True
    return False


def insertion_origin(
    page: fitz.Page,
    item: dict[str, object],
    font: fitz.Font,
    size: float,
) -> tuple[fitz.Point, float]:
    text = str(item["text"])
    rect = fitz.Rect(item["rect"])
    direction = fitz.Point(item["direction"])
    normal = fitz.Point(-direction.y, direction.x)
    center = (rect.tl + rect.br) / 2
    # PyMuPDF's morph angle uses the opposite sign from the top-down text
    # direction returned by get_text().
    angle = -math.degrees(math.atan2(direction.y, direction.x))

    def positioned_origin(length: float, vertical_offset: float) -> fitz.Point:
        if abs(direction.y) >= 0.01:
            return fitz.Point(item["origin"])
        if bool(item.get("right_anchor")):
            # Matplotlib y tick labels are right-aligned to the axis. Preserve
            # that right edge so enlargement proceeds into the label margin
            # instead of covering the first heatmap column.
            return fitz.Point(rect.x1 - length, center.y - vertical_offset)
        if is_left_anchored(text, rect, page):
            return fitz.Point(rect.x0, center.y - vertical_offset)
        return center - direction * (length / 2) - normal * vertical_offset

    # Keep the enlarged text on the PDF page. For left-aligned panel titles,
    # retain their original left edge; for y tick labels, retain the right
    # edge at the axis; otherwise preserve the visual center.
    requested = size
    while size > float(item["old_size"]):
        length = font.text_length(text, fontsize=size)
        vertical_offset = -(font.ascender + font.descender) * size / 2
        origin = positioned_origin(length, vertical_offset)
        corners = [
            origin - normal * (font.ascender * size),
            origin + direction * length - normal * (font.ascender * size),
            origin - normal * (font.descender * size),
            origin + direction * length - normal * (font.descender * size),
        ]
        left = min(point.x for point in corners)
        right = max(point.x for point in corners)
        top = min(point.y for point in corners)
        bottom = max(point.y for point in corners)
        if (
            left >= 2
            and right <= page.rect.width - 2
            and top >= 2
            and bottom <= page.rect.height - 2
        ):
            item["new_size"] = size
            return origin, angle
        size -= 0.25
    if requested != size:
        size = max(size, float(item["old_size"]))
    length = font.text_length(text, fontsize=size)
    vertical_offset = -(font.ascender + font.descender) * size / 2
    origin = positioned_origin(length, vertical_offset)
    # A few source labels already extended slightly beyond the MediaBox. If
    # enlargement cannot fit even at the original size, retain that size and
    # translate the label just inside a two-point safety margin.
    corners = [
        origin - normal * (font.ascender * size),
        origin + direction * length - normal * (font.ascender * size),
        origin - normal * (font.descender * size),
        origin + direction * length - normal * (font.descender * size),
    ]
    left = min(point.x for point in corners)
    right = max(point.x for point in corners)
    top = min(point.y for point in corners)
    bottom = max(point.y for point in corners)
    dx = max(0.0, 2.0 - left) - max(0.0, right - (page.rect.width - 2.0))
    dy = max(0.0, 2.0 - top) - max(0.0, bottom - (page.rect.height - 2.0))
    origin += fitz.Point(dx, dy)
    item["new_size"] = size
    return origin, angle


def process_pdf(source: Path, destination: Path, rule: FigureRule, force: bool) -> tuple[int, int]:
    document = fitz.open(source)
    metadata = document.metadata or {}
    keywords = metadata.get("keywords", "") or ""
    if MARKER in keywords and not force:
        document.close()
        if source != destination:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        return 0, 0

    replaced = 0
    removed = 0
    for page in document:
        items = line_items(page, rule)
        for item in items:
            rectangle = fitz.Rect(item["rect"])
            padding = max(0.35, float(item["old_size"]) * 0.04)
            rectangle += (-padding, -padding, padding, padding)
            # Include the omitted Unicode-minus position in the vertical
            # measured-reversal colorbar label.
            if item["old_text"] == "rho; higher = stronger)":
                rectangle.y1 += float(item["old_size"]) * 1.15
            page.add_redact_annot(rectangle, fill=None, cross_out=False)
        if items:
            page.apply_redactions(
                images=fitz.PDF_REDACT_IMAGE_NONE,
                graphics=fitz.PDF_REDACT_LINE_ART_NONE,
                text=fitz.PDF_REDACT_TEXT_REMOVE,
            )
        fonts = {key: fitz.Font(fontfile=str(path)) for key, path in FONT_FILES.items()}
        for item in items:
            if item["remove"]:
                removed += 1
                continue
            key = font_key(str(item["font"]), str(item["text"]), float(item["old_size"]))
            font = fonts[key]
            origin, angle = insertion_origin(page, item, font, float(item["new_size"]))
            page.insert_text(
                origin,
                str(item["text"]),
                fontsize=float(item["new_size"]),
                fontname=f"qa_{key}",
                fontfile=str(FONT_FILES[key]),
                color=fitz.sRGB_to_pdf(int(item["color"])),
                morph=(origin, fitz.Matrix(angle)),
                overlay=True,
            )
            replaced += 1
        if rule.crop_bottom:
            crop = fitz.Rect(page.cropbox)
            crop.y1 -= rule.crop_bottom
            page.set_cropbox(crop)

    metadata["keywords"] = "; ".join(
        part for part in [keywords.strip("; "), MARKER] if part
    )
    document.set_metadata(metadata)
    # Keep the portable source and compiled submission PDFs compact: PyMuPDF
    # otherwise embeds each full replacement font in every standalone figure.
    document.subset_fonts()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".fontqa.tmp.pdf")
    document.save(temporary, garbage=4, deflate=True)
    document.close()
    temporary.replace(destination)
    return replaced, removed


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve() if args.output_root else source_root
    for path in FONT_FILES.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required font not found: {path}")

    total_replaced = 0
    total_removed = 0
    for relative, rule in RULES.items():
        source = source_root / relative
        destination = output_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"Figure not found: {source}")
        replaced, removed = process_pdf(source, destination, rule, args.force)
        total_replaced += replaced
        total_removed += removed
        print(f"{relative}: enlarged={replaced}, removed={removed}")

    print("FIGURE FONT QA POST-PROCESSING: PASSED")
    print(f"figures={len(RULES)}")
    print(f"text_objects_enlarged={total_replaced}")
    print(f"redundant_text_objects_removed={total_removed}")
    print(f"output_root={output_root}")


if __name__ == "__main__":
    main()
