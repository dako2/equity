"""
Dynamic Report Builder - JPM-style template with auto-generated placeholder content.

All prefilled content uses repeated [PLACEHOLDER] markers with yellow highlighting.
Content is auto-generated to fill target word counts, not hardcoded.

Includes hard boundary enforcement and violation detection.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepInFrame
)
from reportlab.graphics.shapes import Drawing, Rect, String

from .layout_params import LayoutParams
from .content_placeholders import PlaceholderRegistry, create_jpm_style_placeholders


# ============================================
# SECTION BOUNDARY DEFINITIONS
# ============================================

class SectionType(Enum):
    """Types of sections in the report layout."""
    PAGE_MARGIN = "page_margin"
    HEADER = "header"
    FOOTER = "footer"
    MAIN_CONTENT = "main_content"
    SIDEBAR = "sidebar"
    BOTTOM_BAND = "bottom_band"
    TWO_COLUMN = "two_column"
    TABLE = "table"
    CHART = "chart"


@dataclass
class SectionBoundary:
    """Defines a hard boundary for a layout section."""
    name: str
    section_type: SectionType
    page: int
    x_min: float
    x_max: float
    y_min: float  # Bottom of section (lower Y value)
    y_max: float  # Top of section (higher Y value)
    
    @property
    def width(self) -> float:
        return self.x_max - self.x_min
    
    @property
    def height(self) -> float:
        return self.y_max - self.y_min
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is within this boundary."""
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max
    
    def intersects(self, other: 'SectionBoundary') -> bool:
        """Check if this boundary intersects with another."""
        return (self.x_min < other.x_max and self.x_max > other.x_min and
                self.y_min < other.y_max and self.y_max > other.y_min)
    
    def get_intersection(self, other: 'SectionBoundary') -> Optional[Tuple[float, float, float, float]]:
        """Get intersection rectangle if boundaries overlap."""
        if not self.intersects(other):
            return None
        return (
            max(self.x_min, other.x_min),
            max(self.y_min, other.y_min),
            min(self.x_max, other.x_max),
            min(self.y_max, other.y_max)
        )


@dataclass
class BoundaryViolation:
    """Represents a boundary violation."""
    section_name: str
    section_type: SectionType
    page: int
    violation_type: str  # "overflow", "overlap", "margin_breach"
    severity: str  # "warning", "error"
    message: str
    bbox: Tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)
    overflow_amount: Optional[Tuple[float, float]] = None  # (x_overflow, y_overflow)


@dataclass
class LayoutBoundaries:
    """Collection of all section boundaries for a report."""
    sections: List[SectionBoundary] = field(default_factory=list)
    
    def add(self, section: SectionBoundary):
        """Add a section boundary."""
        self.sections.append(section)
    
    def get_by_page(self, page: int) -> List[SectionBoundary]:
        """Get all sections on a specific page."""
        return [s for s in self.sections if s.page == page]
    
    def get_by_type(self, section_type: SectionType) -> List[SectionBoundary]:
        """Get all sections of a specific type."""
        return [s for s in self.sections if s.section_type == section_type]
    
    def get_by_name(self, name: str) -> Optional[SectionBoundary]:
        """Get a section by name."""
        for s in self.sections:
            if s.name == name:
                return s
        return None


def create_page_boundaries(params: LayoutParams, page_num: int = 1) -> LayoutBoundaries:
    """Create standard section boundaries for a page based on layout parameters."""
    W, H = LETTER
    boundaries = LayoutBoundaries()
    
    lm = params.margin_left
    rm = params.margin_right
    tm = params.margin_top
    bm = params.margin_bottom
    
    # Page content area (within margins)
    boundaries.add(SectionBoundary(
        name=f"page_{page_num}_content",
        section_type=SectionType.PAGE_MARGIN,
        page=page_num,
        x_min=lm,
        x_max=W - rm,
        y_min=bm,
        y_max=H - tm
    ))
    
    # Header area
    boundaries.add(SectionBoundary(
        name=f"page_{page_num}_header",
        section_type=SectionType.HEADER,
        page=page_num,
        x_min=lm,
        x_max=W - rm,
        y_min=H - params.header_line_y_offset - 30,
        y_max=H
    ))
    
    # Footer area
    boundaries.add(SectionBoundary(
        name=f"page_{page_num}_footer",
        section_type=SectionType.FOOTER,
        page=page_num,
        x_min=lm,
        x_max=W - rm,
        y_min=0,
        y_max=params.footer_line_y_offset + 10
    ))
    
    return boundaries


def create_summary_page_boundaries(params: LayoutParams) -> LayoutBoundaries:
    """Create boundaries specific to the summary page (page 1)."""
    W, H = LETTER
    boundaries = create_page_boundaries(params, page_num=1)
    
    lm = params.margin_left
    rm = params.margin_right
    tm = params.margin_top
    bm = params.margin_bottom
    
    content_w = W - lm - rm
    gutter = params.gutter_width
    sidebar_w = params.sidebar_width
    main_w = content_w - gutter - sidebar_w
    
    # Header clearance
    content_top = H - tm - params.header_clearance - 20
    
    # Two-column layout area
    two_col_height = H - tm - bm - params.header_clearance - 150
    two_col_bottom = content_top - two_col_height
    
    # Main content column
    boundaries.add(SectionBoundary(
        name="main_content",
        section_type=SectionType.MAIN_CONTENT,
        page=1,
        x_min=lm,
        x_max=lm + main_w,
        y_min=two_col_bottom,
        y_max=content_top
    ))
    
    # Sidebar
    sidebar_start = lm + main_w + gutter
    boundaries.add(SectionBoundary(
        name="sidebar",
        section_type=SectionType.SIDEBAR,
        page=1,
        x_min=sidebar_start,
        x_max=sidebar_start + sidebar_w,
        y_min=two_col_bottom,
        y_max=content_top
    ))
    
    # Bottom band
    bottom_band_height = params.bottom_band_height
    boundaries.add(SectionBoundary(
        name="bottom_band",
        section_type=SectionType.BOTTOM_BAND,
        page=1,
        x_min=lm,
        x_max=W - rm,
        y_min=bm,
        y_max=bm + bottom_band_height
    ))
    
    return boundaries


class BoundaryValidator:
    """Validates PDF content against defined boundaries."""
    
    def __init__(self, boundaries: LayoutBoundaries):
        self.boundaries = boundaries
        self.violations: List[BoundaryViolation] = []
    
    def validate_pdf(self, pdf_path: str) -> List[BoundaryViolation]:
        """Validate a PDF against all boundaries."""
        self.violations = []
        
        try:
            import fitz
        except ImportError:
            self.violations.append(BoundaryViolation(
                section_name="system",
                section_type=SectionType.PAGE_MARGIN,
                page=0,
                violation_type="error",
                severity="error",
                message="PyMuPDF not available for boundary validation",
                bbox=(0, 0, 0, 0)
            ))
            return self.violations
        
        doc = fitz.open(pdf_path)
        
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1
            
            # Get page boundaries
            page_sections = self.boundaries.get_by_page(page_num)
            content_boundary = None
            for sec in page_sections:
                if sec.section_type == SectionType.PAGE_MARGIN:
                    content_boundary = sec
                    break
            
            if not content_boundary:
                continue
            
            # Check all text blocks
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                bbox = block.get("bbox", [0, 0, 0, 0])
                block_type = "text" if block.get("type") == 0 else "image"
                
                # Check margin violations
                self._check_margin_violation(
                    page_num, content_boundary, bbox, block_type
                )
                
                # Check section-specific violations
                self._check_section_violations(
                    page_num, page_sections, bbox, block_type
                )
        
        doc.close()
        return self.violations
    
    def _check_margin_violation(
        self, 
        page: int,
        content_boundary: SectionBoundary,
        bbox: Tuple[float, float, float, float],
        element_type: str
    ):
        """Check if an element violates page margins."""
        x_min, y_min, x_max, y_max = bbox
        
        # Calculate overflow amounts
        left_overflow = content_boundary.x_min - x_min if x_min < content_boundary.x_min else 0
        right_overflow = x_max - content_boundary.x_max if x_max > content_boundary.x_max else 0
        top_overflow = y_max - content_boundary.y_max if y_max > content_boundary.y_max else 0
        bottom_overflow = content_boundary.y_min - y_min if y_min < content_boundary.y_min else 0
        
        # Check for significant overflow (> 2pt tolerance)
        TOLERANCE = 2.0
        
        if right_overflow > TOLERANCE:
            self.violations.append(BoundaryViolation(
                section_name=f"{element_type}_block",
                section_type=SectionType.PAGE_MARGIN,
                page=page,
                violation_type="margin_breach",
                severity="error" if element_type == "image" else "warning",
                message=f"{element_type.title()} overflows right margin by {right_overflow:.1f}pt",
                bbox=bbox,
                overflow_amount=(right_overflow, 0)
            ))
        
        if left_overflow > TOLERANCE:
            self.violations.append(BoundaryViolation(
                section_name=f"{element_type}_block",
                section_type=SectionType.PAGE_MARGIN,
                page=page,
                violation_type="margin_breach",
                severity="error",
                message=f"{element_type.title()} overflows left margin by {left_overflow:.1f}pt",
                bbox=bbox,
                overflow_amount=(left_overflow, 0)
            ))
        
        if bottom_overflow > TOLERANCE:
            self.violations.append(BoundaryViolation(
                section_name=f"{element_type}_block",
                section_type=SectionType.PAGE_MARGIN,
                page=page,
                violation_type="margin_breach",
                severity="warning",
                message=f"{element_type.title()} overflows bottom margin by {bottom_overflow:.1f}pt",
                bbox=bbox,
                overflow_amount=(0, bottom_overflow)
            ))
    
    def _check_section_violations(
        self,
        page: int,
        page_sections: List[SectionBoundary],
        bbox: Tuple[float, float, float, float],
        element_type: str
    ):
        """Check if content violates section boundaries."""
        x_min, y_min, x_max, y_max = bbox
        
        # Create a temporary boundary for this element
        element_boundary = SectionBoundary(
            name="element",
            section_type=SectionType.TABLE if element_type == "image" else SectionType.MAIN_CONTENT,
            page=page,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max
        )
        
        # Check for overlaps between non-overlapping sections
        for section in page_sections:
            if section.section_type in [SectionType.HEADER, SectionType.FOOTER]:
                if section.intersects(element_boundary):
                    intersection = section.get_intersection(element_boundary)
                    if intersection:
                        ix_min, iy_min, ix_max, iy_max = intersection
                        overlap_area = (ix_max - ix_min) * (iy_max - iy_min)
                        if overlap_area > 10:  # Significant overlap
                            self.violations.append(BoundaryViolation(
                                section_name=section.name,
                                section_type=section.section_type,
                                page=page,
                                violation_type="overlap",
                                severity="warning",
                                message=f"Content overlaps with {section.section_type.value} area",
                                bbox=bbox
                            ))
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get violations as alert dictionaries for display."""
        alerts = []
        for v in self.violations:
            alerts.append({
                "level": v.severity,
                "page": v.page,
                "section": v.section_name,
                "type": v.violation_type,
                "message": v.message,
                "bbox": v.bbox,
                "overflow": v.overflow_amount
            })
        return alerts
    
    def print_alerts(self):
        """Print all violations as alerts to console."""
        if not self.violations:
            print("✅ No boundary violations detected")
            return
        
        errors = [v for v in self.violations if v.severity == "error"]
        warnings = [v for v in self.violations if v.severity == "warning"]
        
        print(f"\n{'='*60}")
        print(f"🚨 BOUNDARY VIOLATIONS DETECTED")
        print(f"{'='*60}")
        print(f"  Errors: {len(errors)}")
        print(f"  Warnings: {len(warnings)}")
        print(f"{'='*60}\n")
        
        for v in self.violations:
            icon = "❌" if v.severity == "error" else "⚠️"
            print(f"{icon} [{v.severity.upper()}] Page {v.page}: {v.message}")
            if v.overflow_amount:
                print(f"   Overflow: {v.overflow_amount[0]:.1f}pt x {v.overflow_amount[1]:.1f}pt")
            print(f"   Section: {v.section_name} ({v.section_type.value})")
            print(f"   BBox: ({v.bbox[0]:.1f}, {v.bbox[1]:.1f}) - ({v.bbox[2]:.1f}, {v.bbox[3]:.1f})")
            print()


def validate_report_boundaries(
    pdf_path: str, 
    params: LayoutParams,
    print_alerts: bool = True
) -> Tuple[bool, List[BoundaryViolation]]:
    """
    Validate a generated report against its layout boundaries.
    
    Returns:
        Tuple of (is_valid, violations_list)
    """
    # Create boundaries for all pages
    boundaries = create_summary_page_boundaries(params)
    
    # Add boundaries for additional pages (2-6)
    for page_num in range(2, 7):
        page_boundaries = create_page_boundaries(params, page_num)
        for section in page_boundaries.sections:
            boundaries.add(section)
    
    # Validate
    validator = BoundaryValidator(boundaries)
    violations = validator.validate_pdf(pdf_path)
    
    if print_alerts:
        validator.print_alerts()
    
    # Consider valid if no errors (warnings are OK)
    errors = [v for v in violations if v.severity == "error"]
    is_valid = len(errors) == 0
    
    return is_valid, violations


# ============================================
# YELLOW HIGHLIGHT COLOR FOR PLACEHOLDERS
# ============================================
HIGHLIGHT_YELLOW = colors.HexColor("#FFFF00")
HIGHLIGHT_LIGHT = colors.HexColor("#FFFACD")


# ============================================
# AUTO-GENERATED PLACEHOLDER CONTENT
# ============================================

def generate_placeholder_text(word_count: int, marker: str = "[PLACEHOLDER]") -> str:
    """Generate repeated placeholder markers to fill target word count."""
    words = [marker] * word_count
    return " ".join(words)


def generate_placeholder_paragraph(word_count: int = 40) -> str:
    """Generate a paragraph of placeholder text."""
    return generate_placeholder_text(word_count)


def generate_placeholder_bullet(word_count: int = 12) -> str:
    """Generate a bullet point of placeholder text."""
    return generate_placeholder_text(word_count)


def generate_placeholder_paragraphs(count: int, words_per_para: int = 40) -> List[str]:
    """Generate multiple placeholder paragraphs."""
    return [generate_placeholder_paragraph(words_per_para) for _ in range(count)]


def generate_placeholder_bullets(count: int, words_per_bullet: int = 12) -> List[str]:
    """Generate multiple placeholder bullets."""
    return [generate_placeholder_bullet(words_per_bullet) for _ in range(count)]


def generate_placeholder_table(rows: int, cols: int) -> Dict[str, Any]:
    """Generate a placeholder table with specified dimensions."""
    return {
        "headers": ["[PLACEHOLDER]"] * cols,
        "rows": [["[PLACEHOLDER]"] * cols for _ in range(rows)]
    }


def generate_placeholder_chart(width: float = 200, height: float = 100) -> Drawing:
    """Generate a placeholder chart area with yellow background."""
    d = Drawing(width, height)
    
    # Yellow background to indicate placeholder
    d.add(Rect(0, 0, width, height, fillColor=HIGHLIGHT_LIGHT, strokeColor=colors.gray, strokeWidth=1))
    
    # Center label
    d.add(String(width/2, height/2, "[PLACEHOLDER CHART]", 
                 fontSize=10, textAnchor='middle', fillColor=colors.black))
    
    return d


# ============================================
# STYLES WITH YELLOW HIGHLIGHTING
# ============================================

def make_styles(params: LayoutParams):
    """Create paragraph styles - highlighted versions for placeholders."""
    s = getSampleStyleSheet()
    
    # Regular styles (for headers/structure)
    s.add(ParagraphStyle(
        "H1",
        fontName="Helvetica-Bold",
        fontSize=params.h1_font_size,
        leading=params.h1_leading,
        textColor=colors.black,
        spaceAfter=8
    ))
    s.add(ParagraphStyle(
        "H2",
        fontName="Helvetica-Bold",
        fontSize=params.h2_font_size,
        leading=params.h2_leading,
        textColor=colors.black,
        spaceBefore=params.h2_space_before,
        spaceAfter=params.h2_space_after
    ))
    s.add(ParagraphStyle(
        "Small",
        fontName="Helvetica",
        fontSize=params.small_font_size,
        leading=params.small_leading,
        textColor=colors.HexColor("#444444")
    ))
    s.add(ParagraphStyle(
        "Tiny",
        fontName="Helvetica",
        fontSize=params.tiny_font_size,
        leading=params.tiny_leading,
        textColor=colors.HexColor("#555555")
    ))
    
    # HIGHLIGHTED STYLES for placeholder content
    s.add(ParagraphStyle(
        "Body",
        fontName="Helvetica",
        fontSize=params.body_font_size,
        leading=params.body_leading,
        textColor=colors.black,
        backColor=HIGHLIGHT_YELLOW
    ))
    s.add(ParagraphStyle(
        "BodySmall",
        fontName="Helvetica",
        fontSize=7.5,
        leading=9,
        textColor=colors.black,
        backColor=HIGHLIGHT_YELLOW
    ))
    s.add(ParagraphStyle(
        "ReportBullet",
        fontName="Helvetica",
        fontSize=params.body_font_size,
        leading=params.body_leading,
        textColor=colors.black,
        leftIndent=12,
        bulletIndent=0,
        backColor=HIGHLIGHT_YELLOW
    ))
    s.add(ParagraphStyle(
        "PlaceholderTiny",
        fontName="Helvetica",
        fontSize=params.tiny_font_size,
        leading=params.tiny_leading,
        textColor=colors.black,
        backColor=HIGHLIGHT_YELLOW
    ))
    
    return s


def draw_header(canvas, doc, data: Dict, params: LayoutParams, page_type: str):
    """Draw page header and footer with placeholder markers."""
    W, H = LETTER
    lm = params.margin_left
    rm = params.margin_right

    canvas.saveState()
    
    # Top divider line
    canvas.setStrokeColor(colors.HexColor("#D0D6DC"))
    canvas.setLineWidth(0.6)
    y = H - params.header_line_y_offset
    canvas.line(lm, y, W - rm, y)

    # Yellow highlight for header placeholders
    canvas.setFillColor(HIGHLIGHT_YELLOW)
    canvas.rect(lm, y + 2, 150, 12, fill=1, stroke=0)
    canvas.rect(W - rm - 100, y + 2, 100, 12, fill=1, stroke=0)
    
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.black)
    canvas.drawString(lm + 2, y + 7, "[PLACEHOLDER]")
    canvas.drawRightString(W - rm - 2, y + 7, "[PLACEHOLDER]")

    # Analyst info
    canvas.setFillColor(HIGHLIGHT_YELLOW)
    canvas.rect(lm, y - 25, 120, 20, fill=1, stroke=0)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.black)
    canvas.drawString(lm + 2, y - 10, "[PLACEHOLDER]")
    canvas.drawString(lm + 2, y - 20, "[PLACEHOLDER]")

    # Footer
    canvas.setStrokeColor(colors.HexColor("#D0D6DC"))
    canvas.line(lm, params.footer_line_y_offset, W - rm, params.footer_line_y_offset)
    
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawRightString(W - rm, params.footer_text_y_offset, f"{doc.page}")

    canvas.restoreState()


def create_placeholder_table(rows: int, cols: int, params: LayoutParams, 
                             font_size: float = 7, col_widths: Optional[List[float]] = None) -> Table:
    """Create a styled table with auto-generated placeholder cells."""
    table_data = generate_placeholder_table(rows, cols)
    
    # Create a style for table cells that ensures word wrapping
    cell_style = ParagraphStyle(
        "TableCell",
        fontName="Helvetica",
        fontSize=font_size,
        leading=font_size + 2,
        textColor=colors.black,
        wordWrap='CJK'  # Force word wrap at any character
    )
    header_style = ParagraphStyle(
        "TableHeader",
        fontName="Helvetica-Bold",
        fontSize=font_size,
        leading=font_size + 2,
        textColor=colors.black,
        wordWrap='CJK'
    )
    
    # Wrap all cell content in Paragraphs to ensure proper wrapping
    header_row = [Paragraph(cell, header_style) for cell in table_data["headers"]]
    data_rows = [
        [Paragraph(cell, cell_style) for cell in row]
        for row in table_data["rows"]
    ]
    all_rows = [header_row] + data_rows
    
    t = Table(all_rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        # Font styles applied via Paragraph styles above, not TableStyle
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EDF2")),
        ("BACKGROUND", (0, 1), (-1, -1), HIGHLIGHT_LIGHT),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D0D6DC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]))
    return t


# ============================================
# MAIN BUILDER
# ============================================

def build_dynamic_report(
    data: Dict[str, Any],
    params: LayoutParams,
    registry: PlaceholderRegistry,
    output_path: str
) -> Dict[str, Any]:
    """
    Build a full equity research PDF with auto-generated placeholder content.
    
    All content areas use repeated [PLACEHOLDER] markers with yellow highlighting.
    """
    styles = make_styles(params)
    W, H = LETTER

    lm = params.margin_left
    rm = params.margin_right
    tm = params.margin_top
    bm = params.margin_bottom
    
    gutter = params.gutter_width
    sidebar_w = params.sidebar_width
    content_w = W - lm - rm
    main_w = content_w - gutter - sidebar_w

    doc = SimpleDocTemplate(
        output_path,
        pagesize=LETTER,
        leftMargin=lm,
        rightMargin=rm,
        topMargin=tm + params.header_clearance + 20,
        bottomMargin=bm,
        title="Equity Research Report Template"
    )

    def on_page(canvas, doc_obj):
        draw_header(canvas, doc_obj, data, params, "body")

    story = []
    rendered = []

    # ============================================
    # PAGE 1: SUMMARY PAGE
    # ============================================
    
    main_content = []
    
    # Headline
    main_content.append(Paragraph("<b>[PLACEHOLDER] [PLACEHOLDER]</b>", styles["H1"]))
    
    # Subheadline
    main_content.append(Paragraph(generate_placeholder_text(8), styles["Body"]))
    main_content.append(Spacer(1, 8))
    
    # Key Points
    main_content.append(Paragraph("<b>Key Investment Points</b>", styles["H2"]))
    
    ph = registry.get("key_points")
    num_bullets = ph.current_items if ph else 4
    bullets = generate_placeholder_bullets(num_bullets, words_per_bullet=10)
    for bullet in bullets:
        main_content.append(Paragraph(f"• {bullet}", styles["ReportBullet"]))
        main_content.append(Spacer(1, 2))
    rendered.append(("key_points", num_bullets))
    
    main_content.append(Spacer(1, 8))
    
    # Summary paragraphs
    ph = registry.get("summary_body")
    num_paras = ph.current_items if ph else 3
    paragraphs = generate_placeholder_paragraphs(num_paras, words_per_para=35)
    for para in paragraphs:
        main_content.append(Paragraph(para, styles["BodySmall"]))
        main_content.append(Spacer(1, 4))
    rendered.append(("summary_body", num_paras))
    
    # Sidebar
    sidebar_content = []
    
    sidebar_content.append(Paragraph("<b>[PLACEHOLDER]</b>", styles["H2"]))
    sidebar_content.append(Spacer(1, 4))
    
    sidebar_content.append(Paragraph("[PLACEHOLDER] [PLACEHOLDER]", styles["Small"]))
    sidebar_content.append(Paragraph("<b>[PLACEHOLDER] [PLACEHOLDER]</b>", styles["Body"]))
    sidebar_content.append(Paragraph("<b>[PLACEHOLDER] [PLACEHOLDER]</b>", styles["Body"]))
    sidebar_content.append(Paragraph("[PLACEHOLDER]", styles["Small"]))
    sidebar_content.append(Spacer(1, 8))
    
    sidebar_content.append(Paragraph("<b>Analyst</b>", styles["Small"]))
    sidebar_content.append(Paragraph("[PLACEHOLDER]", styles["Small"]))
    sidebar_content.append(Paragraph("[PLACEHOLDER]", styles["Tiny"]))
    sidebar_content.append(Paragraph("[PLACEHOLDER]", styles["Tiny"]))
    sidebar_content.append(Spacer(1, 8))
    
    # Price chart placeholder
    sidebar_content.append(Paragraph("<b>Price Performance</b>", styles["Small"]))
    chart = generate_placeholder_chart(width=sidebar_w - 20, height=70)
    sidebar_content.append(chart)
    sidebar_content.append(Spacer(1, 6))
    
    # Price perf table
    perf_table = create_placeholder_table(rows=2, cols=5, params=params, font_size=6.5)
    sidebar_content.append(perf_table)
    rendered.append(("price_perf_table", 2))
    
    # Two-column layout
    main_frame_h = H - tm - bm - params.header_clearance - 150
    
    main_flowable = KeepInFrame(main_w - 10, main_frame_h, main_content, mode='shrink')
    sidebar_flowable = KeepInFrame(sidebar_w - 12, main_frame_h, sidebar_content, mode='shrink')
    
    two_col = Table(
        [[main_flowable, sidebar_flowable]],
        colWidths=[main_w, sidebar_w],
        hAlign='LEFT'
    )
    two_col.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (0, 0), gutter),
        ("RIGHTPADDING", (1, 0), (1, 0), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("BOX", (1, 0), (1, 0), 0.6, colors.HexColor("#D0D6DC")),
        ("LEFTPADDING", (1, 0), (1, 0), 6),
        ("RIGHTPADDING", (1, 0), (1, 0), 6),
        ("TOPPADDING", (1, 0), (1, 0), 6),
        ("BOTTOMPADDING", (1, 0), (1, 0), 6),
    ]))
    
    story.append(two_col)
    story.append(Spacer(1, 12))
    
    # Bottom band tables
    eps_table = create_placeholder_table(
        rows=6, cols=6, params=params, font_size=6.5,
        col_widths=[60, 45, 45, 45, 45, 45]
    )
    
    cd_table = create_placeholder_table(
        rows=10, cols=2, params=params, font_size=6.5,
        col_widths=[90, 70]
    )
    
    band_left_w = content_w * 0.6
    band_right_w = content_w * 0.4
    
    bottom_band = Table(
        [[eps_table, cd_table]],
        colWidths=[band_left_w, band_right_w],
        hAlign='LEFT'
    )
    bottom_band.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (0, 0), 12),
    ]))
    
    story.append(bottom_band)
    rendered.append(("eps_quarterly", 6))
    rendered.append(("company_data", 10))
    
    # ============================================
    # PAGE 2: EXECUTIVE SUMMARY
    # ============================================
    story.append(PageBreak())
    
    story.append(Paragraph("Executive Summary", styles["H1"]))
    story.append(Spacer(1, 8))
    
    ph = registry.get("exec_summary")
    num_paras = ph.current_items if ph else 5
    paragraphs = generate_placeholder_paragraphs(num_paras, words_per_para=45)
    for para in paragraphs:
        story.append(Paragraph(para, styles["Body"]))
        story.append(Spacer(1, 6))
    rendered.append(("exec_summary", num_paras))
    
    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Valuation Summary</b>", styles["H2"]))
    
    ph = registry.get("valuation_summary")
    num_bullets = ph.current_items if ph else 3
    bullets = generate_placeholder_bullets(num_bullets, words_per_bullet=12)
    for bullet in bullets:
        story.append(Paragraph(f"• {bullet}", styles["ReportBullet"]))
        story.append(Spacer(1, 2))
    rendered.append(("valuation_summary", num_bullets))
    
    # Chart
    story.append(Spacer(1, 12))
    chart = generate_placeholder_chart(width=400, height=160)
    story.append(chart)
    story.append(Paragraph("<i>[PLACEHOLDER]</i>", styles["Tiny"]))
    
    # ============================================
    # PAGE 3: INDUSTRY OVERVIEW
    # ============================================
    story.append(PageBreak())
    
    story.append(Paragraph("Industry Overview", styles["H1"]))
    story.append(Spacer(1, 8))
    
    ph = registry.get("industry_text")
    num_paras = ph.current_items if ph else 4
    paragraphs = generate_placeholder_paragraphs(num_paras, words_per_para=45)
    for para in paragraphs:
        story.append(Paragraph(para, styles["Body"]))
        story.append(Spacer(1, 6))
    rendered.append(("industry_text", num_paras))
    
    # Competitor table
    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Peer Valuation Comparison</b>", styles["H2"]))
    story.append(Spacer(1, 6))
    
    comp_table = create_placeholder_table(
        rows=8, cols=7, params=params, font_size=7,
        col_widths=[100, 40, 50, 55, 55, 55, 55]
    )
    story.append(comp_table)
    rendered.append(("competitor_comparison", 8))
    
    story.append(Spacer(1, 8))
    story.append(Paragraph("<i>[PLACEHOLDER]</i>", styles["Tiny"]))
    
    # ============================================
    # PAGE 4: FINANCIAL PROJECTIONS
    # ============================================
    story.append(PageBreak())
    
    story.append(Paragraph("Financial Projections", styles["H1"]))
    story.append(Spacer(1, 8))
    
    story.append(Paragraph(generate_placeholder_text(40), styles["Body"]))
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("<b>Income Statement Summary</b>", styles["H2"]))
    story.append(Spacer(1, 6))
    
    income_table = create_placeholder_table(
        rows=11, cols=6, params=params, font_size=7,
        col_widths=[90, 60, 60, 60, 60, 60]
    )
    story.append(income_table)
    rendered.append(("income_statement", 11))
    
    story.append(Spacer(1, 8))
    story.append(Paragraph("<i>[PLACEHOLDER]</i>", styles["Tiny"]))
    
    # ============================================
    # PAGE 5: INVESTMENT RISKS
    # ============================================
    story.append(PageBreak())
    
    story.append(Paragraph("Investment Risks", styles["H1"]))
    story.append(Spacer(1, 8))
    
    story.append(Paragraph(generate_placeholder_text(20), styles["Body"]))
    story.append(Spacer(1, 8))
    
    ph = registry.get("risk_factors")
    num_risks = ph.current_items if ph else 6
    risks = generate_placeholder_bullets(num_risks, words_per_bullet=10)
    for risk in risks:
        story.append(Paragraph(f"• {risk}", styles["ReportBullet"]))
        story.append(Spacer(1, 4))
    rendered.append(("risk_factors", num_risks))
    
    story.append(Spacer(1, 12))
    
    # Risk details
    risk_details = generate_placeholder_paragraphs(2, words_per_para=50)
    for detail in risk_details:
        story.append(Paragraph(detail, styles["Body"]))
        story.append(Spacer(1, 6))
    rendered.append(("risk_details", 2))
    
    # ============================================
    # PAGE 6: DISCLOSURES
    # ============================================
    story.append(PageBreak())
    
    story.append(Paragraph("Analyst Certification", styles["H2"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(generate_placeholder_text(30), styles["PlaceholderTiny"]))
    story.append(Spacer(1, 8))
    
    story.append(Paragraph("Important Disclosures", styles["H2"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(generate_placeholder_text(35), styles["PlaceholderTiny"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(generate_placeholder_text(35), styles["PlaceholderTiny"]))
    rendered.append(("disclosures", 3))
    
    # Build
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    
    return {
        "output_path": output_path,
        "rendered_placeholders": rendered,
        "page_count": 6,
        "registry_state": registry.to_manifest()
    }


def create_sample_dynamic_data() -> Dict[str, Any]:
    """Create sample data for the report."""
    return {
        "region": "[PLACEHOLDER]",
        "date": "[PLACEHOLDER]",
        "company": "[PLACEHOLDER]",
        "ticker": "[PLACEHOLDER]",
        "ticker_bbg": "[PLACEHOLDER]",
        "rating": "[PLACEHOLDER]",
        "price": 0.00,
        "target": 0.00,
        "sector": "[PLACEHOLDER]",
        "headline": "[PLACEHOLDER]",
        "analyst_name": "[PLACEHOLDER]",
        "analyst_phone": "[PLACEHOLDER]",
        "analyst_email": "[PLACEHOLDER]",
    }


if __name__ == "__main__":
    # Test the builder
    data = create_sample_dynamic_data()
    params = LayoutParams()
    registry = create_jpm_style_placeholders()
    
    result = build_dynamic_report(data, params, registry, "test_placeholder_report.pdf")
    
    print(f"Generated {result['page_count']}-page template: {result['output_path']}")
    print("\nRendered placeholders:")
    for name, count in result['rendered_placeholders']:
        print(f"  • {name}: {count} items")
    print("\n⚠️  All content is auto-generated [PLACEHOLDER] text with yellow highlighting")
