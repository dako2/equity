"""
PDF Layout Analyzer

Analyzes PDF structure to extract:
- Page layout and dimensions
- Section headers and hierarchy
- Word/character counts per section
- Tables (rows, columns, content)
- Charts/figures (dimensions, captions)
- Text blocks and their positions
- Font usage statistics
- Content summaries using LLM

Usage:
    python -m backend.pdf_analyzer path/to/document.pdf
"""

from __future__ import annotations
import os
import re
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False
    print("⚠️  PyMuPDF not installed. Run: pip install PyMuPDF")


@dataclass
class TextBlock:
    """A block of text with position and style info."""
    text: str
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    font_name: str = ""
    font_size: float = 0.0
    is_bold: bool = False
    is_italic: bool = False
    color: str = "#000000"
    block_type: str = "text"  # text, header, bullet, caption
    page: int = 0
    
    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]
    
    @property
    def word_count(self) -> int:
        return len(self.text.split())
    
    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass
class TableInfo:
    """Information about a detected table."""
    page: int
    bbox: Tuple[float, float, float, float]
    rows: int
    cols: int
    headers: List[str]
    sample_data: List[List[str]]
    caption: str = ""
    table_type: str = "data"  # data, financial, comparison


@dataclass
class FigureInfo:
    """Information about a detected figure/chart."""
    page: int
    bbox: Tuple[float, float, float, float]
    width: float
    height: float
    image_type: str = "unknown"  # chart, photo, logo, diagram
    caption: str = ""
    alt_text: str = ""


@dataclass
class SectionInfo:
    """Information about a document section."""
    title: str
    page_start: int
    page_end: int
    level: int  # 1 = main section, 2 = subsection, etc.
    word_count: int
    char_count: int
    paragraph_count: int
    bullet_count: int
    table_count: int
    figure_count: int
    content_summary: str = ""
    key_topics: List[str] = field(default_factory=list)
    subsections: List["SectionInfo"] = field(default_factory=list)


@dataclass
class PageLayout:
    """Layout information for a single page."""
    page_num: int
    width: float
    height: float
    
    # Margins (detected from content bounds)
    margin_left: float = 0
    margin_right: float = 0
    margin_top: float = 0
    margin_bottom: float = 0
    
    # Content areas
    has_header: bool = False
    has_footer: bool = False
    has_sidebar: bool = False
    column_count: int = 1
    
    # Content counts
    text_block_count: int = 0
    word_count: int = 0
    image_count: int = 0
    table_count: int = 0


@dataclass
class DocumentAnalysis:
    """Complete analysis of a PDF document."""
    filename: str
    page_count: int
    total_word_count: int
    total_char_count: int
    
    # Structure
    sections: List[SectionInfo]
    pages: List[PageLayout]
    
    # Content inventory
    tables: List[TableInfo]
    figures: List[FigureInfo]
    
    # Typography
    fonts_used: Dict[str, int]  # font_name -> usage count
    font_sizes: Dict[float, int]  # size -> usage count
    
    # Detected patterns
    header_style: Dict[str, Any] = field(default_factory=dict)
    body_style: Dict[str, Any] = field(default_factory=dict)
    
    # Document metadata
    title: str = ""
    author: str = ""
    subject: str = ""
    creation_date: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


class PDFAnalyzer:
    """
    Analyzes PDF documents to extract structure and content.
    """
    
    def __init__(self, use_llm: bool = False, llm_provider: str = "xai"):
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        self.llm_client = None
        
        if use_llm:
            self._init_llm()
    
    def _init_llm(self):
        """Initialize LLM client for content summarization."""
        try:
            from openai import OpenAI
            
            if self.llm_provider == "xai":
                api_key = os.getenv("XAI_API_KEY")
                if api_key:
                    self.llm_client = OpenAI(
                        api_key=api_key,
                        base_url="https://api.x.ai/v1"
                    )
            else:
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.llm_client = OpenAI(api_key=api_key)
        except Exception as e:
            print(f"⚠️  Failed to initialize LLM: {e}")
    
    def analyze(self, pdf_path: str) -> DocumentAnalysis:
        """
        Analyze a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            DocumentAnalysis with complete structure and content info
        """
        if not FITZ_AVAILABLE:
            raise RuntimeError("PyMuPDF not installed")
        
        doc = fitz.open(pdf_path)
        
        # Extract metadata
        metadata = doc.metadata or {}
        
        # Analyze each page
        all_text_blocks: List[TextBlock] = []
        pages: List[PageLayout] = []
        tables: List[TableInfo] = []
        figures: List[FigureInfo] = []
        fonts_used: Counter = Counter()
        font_sizes: Counter = Counter()
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Analyze page layout
            page_layout = self._analyze_page_layout(page, page_num)
            pages.append(page_layout)
            
            # Extract text blocks
            text_blocks = self._extract_text_blocks(page, page_num)
            all_text_blocks.extend(text_blocks)
            
            # Update font statistics
            for block in text_blocks:
                if block.font_name:
                    fonts_used[block.font_name] += 1
                if block.font_size > 0:
                    font_sizes[block.font_size] += 1
            
            # Detect tables
            page_tables = self._detect_tables(page, page_num)
            tables.extend(page_tables)
            
            # Detect figures
            page_figures = self._detect_figures(page, page_num)
            figures.extend(page_figures)
        
        # Identify sections from headers
        sections = self._identify_sections(all_text_blocks, tables, figures)
        
        # Calculate totals
        total_words = sum(b.word_count for b in all_text_blocks)
        total_chars = sum(b.char_count for b in all_text_blocks)
        
        # Detect common styles
        header_style = self._detect_header_style(all_text_blocks)
        body_style = self._detect_body_style(all_text_blocks)
        
        # Use LLM to summarize sections if enabled
        if self.use_llm and self.llm_client:
            sections = self._summarize_sections_with_llm(sections, all_text_blocks)
        
        doc.close()
        
        return DocumentAnalysis(
            filename=os.path.basename(pdf_path),
            page_count=len(pages),
            total_word_count=total_words,
            total_char_count=total_chars,
            sections=sections,
            pages=pages,
            tables=tables,
            figures=figures,
            fonts_used=dict(fonts_used),
            font_sizes=dict(font_sizes),
            header_style=header_style,
            body_style=body_style,
            title=metadata.get("title", ""),
            author=metadata.get("author", ""),
            subject=metadata.get("subject", ""),
            creation_date=metadata.get("creationDate", "")
        )
    
    def _analyze_page_layout(self, page: Any, page_num: int) -> PageLayout:
        """Analyze the layout of a single page."""
        rect = page.rect
        
        # Get all content bounds
        blocks = page.get_text("dict")["blocks"]
        
        if not blocks:
            return PageLayout(
                page_num=page_num,
                width=rect.width,
                height=rect.height
            )
        
        # Find content bounds
        min_x = min(b["bbox"][0] for b in blocks if "bbox" in b)
        max_x = max(b["bbox"][2] for b in blocks if "bbox" in b)
        min_y = min(b["bbox"][1] for b in blocks if "bbox" in b)
        max_y = max(b["bbox"][3] for b in blocks if "bbox" in b)
        
        # Detect columns (if content has gap in middle)
        text_blocks = [b for b in blocks if b.get("type") == 0]
        x_positions = [b["bbox"][0] for b in text_blocks]
        
        column_count = 1
        if len(x_positions) > 5:
            # Check for bimodal distribution of x positions
            x_sorted = sorted(set(x_positions))
            if len(x_sorted) >= 2:
                gaps = [x_sorted[i+1] - x_sorted[i] for i in range(len(x_sorted)-1)]
                if gaps and max(gaps) > rect.width * 0.2:
                    column_count = 2
        
        # Check for header (content in top 10%)
        header_threshold = rect.height * 0.1
        has_header = any(b["bbox"][1] < header_threshold for b in blocks if "bbox" in b)
        
        # Check for footer (content in bottom 10%)
        footer_threshold = rect.height * 0.9
        has_footer = any(b["bbox"][3] > footer_threshold for b in blocks if "bbox" in b)
        
        # Check for sidebar (content in right 25%)
        sidebar_threshold = rect.width * 0.75
        has_sidebar = any(b["bbox"][0] > sidebar_threshold for b in blocks if "bbox" in b)
        
        # Count content
        image_count = len([b for b in blocks if b.get("type") == 1])
        word_count = sum(
            len(b.get("lines", [{}])[0].get("spans", [{}])[0].get("text", "").split())
            for b in text_blocks if b.get("lines")
        )
        
        return PageLayout(
            page_num=page_num,
            width=rect.width,
            height=rect.height,
            margin_left=min_x,
            margin_right=rect.width - max_x,
            margin_top=min_y,
            margin_bottom=rect.height - max_y,
            has_header=has_header,
            has_footer=has_footer,
            has_sidebar=has_sidebar,
            column_count=column_count,
            text_block_count=len(text_blocks),
            word_count=word_count,
            image_count=image_count
        )
    
    def _extract_text_blocks(self, page: Any, page_num: int) -> List[TextBlock]:
        """Extract text blocks with styling information."""
        blocks = []
        
        page_dict = page.get_text("dict")
        
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:  # Not a text block
                continue
            
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    
                    font = span.get("font", "")
                    size = span.get("size", 0)
                    flags = span.get("flags", 0)
                    color = span.get("color", 0)
                    
                    # Detect bold/italic from flags
                    is_bold = bool(flags & 2 ** 4) or "Bold" in font or "bold" in font.lower()
                    is_italic = bool(flags & 2 ** 1) or "Italic" in font or "italic" in font.lower()
                    
                    # Convert color to hex
                    if isinstance(color, int):
                        color_hex = f"#{color:06x}"
                    else:
                        color_hex = "#000000"
                    
                    # Classify block type
                    block_type = self._classify_text_block(text, size, is_bold, page_dict)
                    
                    blocks.append(TextBlock(
                        text=text,
                        bbox=tuple(span.get("bbox", (0, 0, 0, 0))),
                        font_name=font,
                        font_size=size,
                        is_bold=is_bold,
                        is_italic=is_italic,
                        color=color_hex,
                        block_type=block_type,
                        page=page_num
                    ))
        
        return blocks
    
    def _classify_text_block(
        self, 
        text: str, 
        font_size: float, 
        is_bold: bool,
        page_dict: Dict
    ) -> str:
        """Classify the type of text block."""
        # Get average font size for comparison
        all_sizes = []
        for block in page_dict.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span.get("size"):
                        all_sizes.append(span["size"])
        
        avg_size = sum(all_sizes) / len(all_sizes) if all_sizes else 10
        
        # Detect headers (larger font, bold, or all caps)
        if font_size > avg_size * 1.3 or (is_bold and font_size > avg_size):
            return "header"
        
        # Detect bullets
        if text.strip().startswith(("•", "-", "▪", "○", "*", "–")):
            return "bullet"
        
        # Detect numbered lists
        if re.match(r"^\d+[\.\)]\s", text.strip()):
            return "numbered"
        
        # Detect captions (smaller font, italic)
        if font_size < avg_size * 0.85:
            return "caption"
        
        return "text"
    
    def _detect_tables(self, page: Any, page_num: int) -> List[TableInfo]:
        """Detect tables on a page."""
        tables = []
        
        try:
            # Use PyMuPDF's table detection
            page_tables = page.find_tables()
            
            for table in page_tables:
                bbox = table.bbox
                cells = table.extract()
                
                if not cells or len(cells) < 2:
                    continue
                
                # Get headers (first row)
                headers = [str(c) if c else "" for c in cells[0]]
                
                # Get sample data (up to 3 rows)
                sample_data = []
                for row in cells[1:4]:
                    sample_data.append([str(c) if c else "" for c in row])
                
                # Classify table type
                table_type = self._classify_table(headers, sample_data)
                
                tables.append(TableInfo(
                    page=page_num,
                    bbox=bbox,
                    rows=len(cells),
                    cols=len(headers),
                    headers=headers,
                    sample_data=sample_data,
                    table_type=table_type
                ))
        except Exception as e:
            # Table detection may fail on some pages
            pass
        
        return tables
    
    def _classify_table(self, headers: List[str], data: List[List[str]]) -> str:
        """Classify the type of table based on content."""
        header_text = " ".join(headers).lower()
        
        # Financial indicators
        financial_keywords = ["eps", "revenue", "ebitda", "margin", "growth", 
                           "fy", "q1", "q2", "q3", "q4", "2024", "2025", "2026",
                           "price", "target", "p/e", "ev/"]
        
        if any(kw in header_text for kw in financial_keywords):
            return "financial"
        
        # Comparison tables
        if any(kw in header_text for kw in ["vs", "comparison", "peer"]):
            return "comparison"
        
        # Performance tables
        if any(kw in header_text for kw in ["ytd", "1m", "3m", "12m", "performance"]):
            return "performance"
        
        return "data"
    
    def _detect_figures(self, page: Any, page_num: int) -> List[FigureInfo]:
        """Detect figures/images on a page."""
        figures = []
        
        images = page.get_images(full=True)
        
        for img_index, img in enumerate(images):
            xref = img[0]
            
            try:
                # Get image bbox
                img_rects = page.get_image_rects(xref)
                
                for rect in img_rects:
                    width = rect.width
                    height = rect.height
                    
                    # Classify image type based on dimensions
                    if width > 200 and height > 100:
                        img_type = "chart"
                    elif width < 100 and height < 100:
                        img_type = "logo"
                    elif width > height * 2:
                        img_type = "banner"
                    else:
                        img_type = "figure"
                    
                    figures.append(FigureInfo(
                        page=page_num,
                        bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                        width=width,
                        height=height,
                        image_type=img_type
                    ))
            except Exception:
                pass
        
        return figures
    
    def _identify_sections(
        self,
        text_blocks: List[TextBlock],
        tables: List[TableInfo],
        figures: List[FigureInfo]
    ) -> List[SectionInfo]:
        """Identify document sections from text blocks."""
        sections = []
        
        # Find header blocks
        headers = [b for b in text_blocks if b.block_type == "header"]
        
        # Sort by page then y position
        headers.sort(key=lambda h: (h.page, h.bbox[1]))
        
        # Determine header levels by font size
        if headers:
            sizes = sorted(set(h.font_size for h in headers), reverse=True)
            size_to_level = {s: i + 1 for i, s in enumerate(sizes[:3])}
        else:
            size_to_level = {}
        
        # Create sections
        for i, header in enumerate(headers):
            # Find end of section (next header of same or higher level, or end of doc)
            level = size_to_level.get(header.font_size, 2)
            
            page_end = header.page
            for next_header in headers[i + 1:]:
                next_level = size_to_level.get(next_header.font_size, 2)
                if next_level <= level:
                    page_end = next_header.page
                    break
                page_end = next_header.page
            
            # Count content in section
            section_blocks = [
                b for b in text_blocks
                if header.page <= b.page <= page_end
                and b.bbox[1] >= header.bbox[1]
                and (b != header)
            ]
            
            # Limit to blocks before next header
            if i + 1 < len(headers):
                next_header = headers[i + 1]
                section_blocks = [
                    b for b in section_blocks
                    if not (b.page == next_header.page and b.bbox[1] >= next_header.bbox[1])
                ]
            
            word_count = sum(b.word_count for b in section_blocks)
            char_count = sum(b.char_count for b in section_blocks)
            paragraph_count = len([b for b in section_blocks if b.block_type == "text"])
            bullet_count = len([b for b in section_blocks if b.block_type in ("bullet", "numbered")])
            
            # Count tables and figures in section
            section_tables = [t for t in tables if header.page <= t.page <= page_end]
            section_figures = [f for f in figures if header.page <= f.page <= page_end]
            
            sections.append(SectionInfo(
                title=header.text,
                page_start=header.page,
                page_end=page_end,
                level=level,
                word_count=word_count,
                char_count=char_count,
                paragraph_count=paragraph_count,
                bullet_count=bullet_count,
                table_count=len(section_tables),
                figure_count=len(section_figures)
            ))
        
        return sections
    
    def _detect_header_style(self, blocks: List[TextBlock]) -> Dict[str, Any]:
        """Detect the most common header style."""
        headers = [b for b in blocks if b.block_type == "header"]
        
        if not headers:
            return {}
        
        # Most common font
        fonts = Counter(h.font_name for h in headers)
        sizes = Counter(h.font_size for h in headers)
        
        return {
            "font": fonts.most_common(1)[0][0] if fonts else "",
            "size": sizes.most_common(1)[0][0] if sizes else 0,
            "bold": sum(1 for h in headers if h.is_bold) > len(headers) / 2,
            "count": len(headers)
        }
    
    def _detect_body_style(self, blocks: List[TextBlock]) -> Dict[str, Any]:
        """Detect the most common body text style."""
        body = [b for b in blocks if b.block_type == "text"]
        
        if not body:
            return {}
        
        fonts = Counter(b.font_name for b in body)
        sizes = Counter(b.font_size for b in body)
        
        return {
            "font": fonts.most_common(1)[0][0] if fonts else "",
            "size": sizes.most_common(1)[0][0] if sizes else 0,
            "count": len(body)
        }
    
    def _summarize_sections_with_llm(
        self,
        sections: List[SectionInfo],
        text_blocks: List[TextBlock]
    ) -> List[SectionInfo]:
        """Use LLM to summarize section content and extract topics."""
        if not self.llm_client:
            return sections
        
        for section in sections:
            # Get section text
            section_text = " ".join(
                b.text for b in text_blocks
                if section.page_start <= b.page <= section.page_end
            )[:2000]  # Limit to 2000 chars
            
            if len(section_text) < 50:
                continue
            
            try:
                response = self.llm_client.chat.completions.create(
                    model="grok-3" if self.llm_provider == "xai" else "gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You analyze equity research report sections. Return JSON only."
                        },
                        {
                            "role": "user",
                            "content": f"""Analyze this section titled "{section.title}":

{section_text}

Return JSON with:
{{"summary": "1-2 sentence summary", "topics": ["topic1", "topic2", "topic3"]}}"""
                        }
                    ],
                    max_tokens=200,
                    temperature=0.3
                )
                
                result = response.choices[0].message.content
                if "```json" in result:
                    result = result.split("```json")[1].split("```")[0]
                
                data = json.loads(result)
                section.content_summary = data.get("summary", "")
                section.key_topics = data.get("topics", [])
                
            except Exception as e:
                print(f"⚠️  Failed to summarize section '{section.title}': {e}")
        
        return sections
    
    def print_analysis(self, analysis: DocumentAnalysis):
        """Print a formatted analysis report."""
        print("\n" + "="*70)
        print(f"📄 PDF ANALYSIS: {analysis.filename}")
        print("="*70)
        
        print(f"\n📊 DOCUMENT OVERVIEW")
        print("-"*40)
        print(f"  Pages: {analysis.page_count}")
        print(f"  Total Words: {analysis.total_word_count:,}")
        print(f"  Total Characters: {analysis.total_char_count:,}")
        print(f"  Tables: {len(analysis.tables)}")
        print(f"  Figures: {len(analysis.figures)}")
        print(f"  Sections: {len(analysis.sections)}")
        
        if analysis.title:
            print(f"  Title: {analysis.title}")
        if analysis.author:
            print(f"  Author: {analysis.author}")
        
        print(f"\n🔤 TYPOGRAPHY")
        print("-"*40)
        if analysis.header_style:
            print(f"  Header: {analysis.header_style.get('font', 'N/A')} @ {analysis.header_style.get('size', 0):.1f}pt")
        if analysis.body_style:
            print(f"  Body: {analysis.body_style.get('font', 'N/A')} @ {analysis.body_style.get('size', 0):.1f}pt")
        
        print(f"\n  Font Usage:")
        for font, count in sorted(analysis.fonts_used.items(), key=lambda x: -x[1])[:5]:
            print(f"    • {font}: {count} blocks")
        
        print(f"\n📐 PAGE LAYOUTS")
        print("-"*40)
        for page in analysis.pages[:3]:  # First 3 pages
            cols = f"{page.column_count}-col" if page.column_count > 1 else "single-col"
            features = []
            if page.has_header: features.append("header")
            if page.has_footer: features.append("footer")
            if page.has_sidebar: features.append("sidebar")
            
            print(f"  Page {page.page_num + 1}: {page.width:.0f}x{page.height:.0f}pt, {cols}")
            print(f"    Margins: L={page.margin_left:.0f} R={page.margin_right:.0f} T={page.margin_top:.0f} B={page.margin_bottom:.0f}")
            if features:
                print(f"    Features: {', '.join(features)}")
            print(f"    Content: {page.text_block_count} blocks, {page.word_count} words, {page.image_count} images")
        
        if len(analysis.pages) > 3:
            print(f"  ... and {len(analysis.pages) - 3} more pages")
        
        print(f"\n📑 SECTIONS ({len(analysis.sections)})")
        print("-"*40)
        for section in analysis.sections:
            level_indent = "  " * section.level
            print(f"{level_indent}• {section.title}")
            print(f"{level_indent}  Pages {section.page_start + 1}-{section.page_end + 1} | "
                  f"{section.word_count:,} words | "
                  f"{section.paragraph_count} paras | "
                  f"{section.bullet_count} bullets | "
                  f"{section.table_count} tables | "
                  f"{section.figure_count} figures")
            
            if section.content_summary:
                print(f"{level_indent}  Summary: {section.content_summary}")
            if section.key_topics:
                print(f"{level_indent}  Topics: {', '.join(section.key_topics)}")
        
        print(f"\n📊 TABLES ({len(analysis.tables)})")
        print("-"*40)
        for i, table in enumerate(analysis.tables):
            print(f"  Table {i + 1}: Page {table.page + 1} | {table.rows}x{table.cols} | Type: {table.table_type}")
            if table.headers:
                headers_str = " | ".join(table.headers[:5])
                if len(table.headers) > 5:
                    headers_str += f" | (+{len(table.headers) - 5} more)"
                print(f"    Headers: {headers_str}")
        
        print(f"\n🖼️  FIGURES ({len(analysis.figures)})")
        print("-"*40)
        for i, fig in enumerate(analysis.figures):
            print(f"  Figure {i + 1}: Page {fig.page + 1} | {fig.width:.0f}x{fig.height:.0f}pt | Type: {fig.image_type}")
        
        print("\n" + "="*70)


def analyze_pdf(pdf_path: str, use_llm: bool = False) -> DocumentAnalysis:
    """
    Convenience function to analyze a PDF.
    
    Args:
        pdf_path: Path to PDF file
        use_llm: Whether to use LLM for content summarization
        
    Returns:
        DocumentAnalysis with complete analysis
    """
    analyzer = PDFAnalyzer(use_llm=use_llm)
    return analyzer.analyze(pdf_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m backend.pdf_analyzer <pdf_path> [--llm]")
        print("\nOptions:")
        print("  --llm    Use LLM to summarize sections (requires XAI_API_KEY)")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    use_llm = "--llm" in sys.argv
    
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)
    
    print(f"📖 Analyzing: {pdf_path}")
    if use_llm:
        print("🤖 LLM summarization enabled")
    
    analyzer = PDFAnalyzer(use_llm=use_llm)
    analysis = analyzer.analyze(pdf_path)
    analyzer.print_analysis(analysis)
    
    # Save JSON output
    output_path = pdf_path.replace(".pdf", "_analysis.json")
    with open(output_path, "w") as f:
        f.write(analysis.to_json())
    print(f"\n💾 Analysis saved to: {output_path}")

