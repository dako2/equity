"""
Unified Report Builder

This module provides a single entry point for building equity research reports
in either PDF or HTML format. It abstracts away the underlying rendering engine
differences and provides a consistent interface.

Pathways:
  1. PDF: ReportLab-based, suitable for print/professional output
  2. HTML: Markdown/CSS-based, suitable for web/interactive viewing

Usage:
    builder = ReportBuilder(output_format="pdf")
    builder.build(content, output_path)
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

from .layout_params import LayoutParams
from .content_placeholders import PlaceholderRegistry, create_jpm_style_placeholders


class OutputFormat(Enum):
    """Supported output formats."""
    PDF = "pdf"
    HTML = "html"
    MARKDOWN = "markdown"


@dataclass
class ReportContent:
    """
    Structured content for an equity research report.
    This is the unified data model used by both PDF and HTML builders.
    """
    # Metadata
    company_name: str = "[Company]"
    ticker: str = "[TICKER]"
    rating: str = "Overweight"
    current_price: str = "$0.00"
    price_target: str = "$0.00"
    analyst_name: str = "Analyst"
    sector: str = "Technology"
    report_date: str = ""
    
    # Headline and key points
    headline: str = "Investment Report"
    key_points: List[str] = None
    
    # Sections (list of dicts with title, content, bullets, tables)
    sections: List[Dict[str, Any]] = None
    
    # Sidebar data
    sidebar_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if not self.report_date:
            self.report_date = datetime.now().strftime("%Y-%m-%d")
        if self.key_points is None:
            self.key_points = []
        if self.sections is None:
            self.sections = []
        if self.sidebar_data is None:
            self.sidebar_data = {}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReportContent":
        """Create ReportContent from dictionary."""
        metadata = data.get("metadata", {})
        return cls(
            company_name=metadata.get("company_name", data.get("company_name", "[Company]")),
            ticker=metadata.get("ticker", data.get("ticker", "[TICKER]")),
            rating=metadata.get("rating", data.get("rating", "Overweight")),
            current_price=metadata.get("current_price", data.get("current_price", "$0.00")),
            price_target=metadata.get("price_target", data.get("price_target", "$0.00")),
            analyst_name=metadata.get("analyst_name", data.get("analyst_name", "Analyst")),
            sector=metadata.get("sector", data.get("sector", "Technology")),
            report_date=metadata.get("report_date", data.get("report_date", "")),
            headline=data.get("headline", "Investment Report"),
            key_points=data.get("key_points", []),
            sections=data.get("sections", []),
            sidebar_data=data.get("sidebar_data", {})
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metadata": {
                "company_name": self.company_name,
                "ticker": self.ticker,
                "rating": self.rating,
                "current_price": self.current_price,
                "price_target": self.price_target,
                "analyst_name": self.analyst_name,
                "sector": self.sector,
                "report_date": self.report_date,
            },
            "headline": self.headline,
            "key_points": self.key_points,
            "sections": self.sections,
            "sidebar_data": self.sidebar_data
        }


class ReportBuilder:
    """
    Unified report builder supporting PDF and HTML output.
    
    Example:
        builder = ReportBuilder(output_format="pdf")
        content = ReportContent(company_name="Tesla", ticker="TSLA")
        builder.build(content, "output/report.pdf")
    """
    
    def __init__(
        self,
        output_format: Union[OutputFormat, str] = OutputFormat.PDF,
        layout_params: Optional[LayoutParams] = None
    ):
        """
        Initialize the report builder.
        
        Args:
            output_format: Output format (pdf, html, markdown)
            layout_params: Layout parameters for the report
        """
        if isinstance(output_format, str):
            output_format = OutputFormat(output_format.lower())
        
        self.output_format = output_format
        self.params = layout_params or LayoutParams()
        self.registry = create_jpm_style_placeholders()
    
    def build(
        self,
        content: Union[ReportContent, Dict[str, Any]],
        output_path: str
    ) -> str:
        """
        Build a report from content.
        
        Args:
            content: Report content (ReportContent or dict)
            output_path: Output file path
        
        Returns:
            Path to generated report
        """
        # Convert dict to ReportContent if needed
        if isinstance(content, dict):
            content = ReportContent.from_dict(content)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        if self.output_format == OutputFormat.PDF:
            return self._build_pdf(content, output_path)
        elif self.output_format == OutputFormat.HTML:
            return self._build_html(content, output_path)
        elif self.output_format == OutputFormat.MARKDOWN:
            return self._build_markdown(content, output_path)
        else:
            raise ValueError(f"Unknown format: {self.output_format}")
    
    def _build_pdf(self, content: ReportContent, output_path: str) -> str:
        """Build PDF report using ReportLab."""
        from .dynamic_report_builder import build_dynamic_report, create_sample_dynamic_data
        
        # Convert ReportContent to the format expected by dynamic_report_builder
        data = create_sample_dynamic_data()
        data.update({
            "company": content.company_name,
            "ticker": content.ticker,
            "rating": content.rating,
            "target": content.price_target,
            "headline": content.headline,
        })
        
        # Add key points
        if content.key_points:
            for i, point in enumerate(content.key_points[:6]):
                data[f"bullet_{i+1}"] = point
        
        # Add section content
        for section in content.sections:
            section_key = section.get("title", "").lower().replace(" ", "_")
            if section.get("content"):
                data[section_key] = section["content"]
        
        build_dynamic_report(data, self.params, self.registry, output_path)
        return output_path
    
    def _build_html(self, content: ReportContent, output_path: str) -> str:
        """Build HTML report."""
        html_content = self._render_html(content)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        return output_path
    
    def _build_markdown(self, content: ReportContent, output_path: str) -> str:
        """Build Markdown report (for md-viewer.html)."""
        md_content = self._render_markdown(content)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        
        return output_path
    
    def _render_markdown(self, content: ReportContent) -> str:
        """Render content as Markdown with sidebar markers."""
        md = f"""---MAIN---

<h1>{content.headline}</h1>
<p><strong>Rating:</strong> {content.rating} | <strong>Price Target:</strong> {content.price_target}</p>

<h2>Key Investment Points</h2>
<ul>
"""
        for point in content.key_points:
            md += f"<li><strong>{point}</strong></li>\n"
        
        md += "</ul>\n"
        
        # Add sections
        for section in content.sections:
            md += f"\n<h2>{section.get('title', 'Section')}</h2>\n"
            
            section_content = section.get("content", "").strip()
            if section_content:
                paragraphs = section_content.split("\n\n")
                for para in paragraphs:
                    if para.strip():
                        md += f"<p>{para.strip()}</p>\n"
            
            bullets = section.get("bullets", [])
            if bullets:
                md += "<ul>\n"
                for bullet in bullets:
                    md += f"<li>{bullet}</li>\n"
                md += "</ul>\n"
            
            tables = section.get("tables", [])
            for table in tables:
                md += f"\n<h3>{table.get('title', 'Table')}</h3>\n"
                md += "<table>\n<thead><tr>\n"
                for header in table.get("headers", []):
                    md += f"<th>{header}</th>\n"
                md += "</tr></thead>\n<tbody>\n"
                for row in table.get("rows", []):
                    md += "<tr>\n"
                    for cell in row:
                        md += f"<td>{cell}</td>\n"
                    md += "</tr>\n"
                md += "</tbody></table>\n"
        
        # Sidebar
        md += f"""
---SIDEBAR---

<h2>{content.company_name}</h2>
<p><strong>{content.ticker}</strong></p>

<p><strong>Price:</strong> {content.current_price}</p>
<p><strong>Target:</strong> {content.price_target}</p>
<p><strong>Rating:</strong> {content.rating}</p>

<h3>Analyst</h3>
<p>{content.analyst_name}</p>
<p>{content.report_date}</p>
"""
        
        sidebar_data = content.sidebar_data
        if sidebar_data.get("price_performance"):
            md += "\n<h3>Price Performance</h3>\n<table>\n"
            for period, value in sidebar_data["price_performance"].items():
                md += f"<tr><td>{period}</td><td>{value}</td></tr>\n"
            md += "</table>\n"
        
        if sidebar_data.get("key_metrics"):
            md += "\n<h3>Key Metrics</h3>\n<table>\n"
            for metric, value in sidebar_data["key_metrics"].items():
                md += f"<tr><td>{metric}</td><td>{value}</td></tr>\n"
            md += "</table>\n"
        
        md += "\n---ENDSIDEBAR---\n"
        
        return md
    
    def _render_html(self, content: ReportContent) -> str:
        """Render content as standalone HTML."""
        # Create a full HTML page with embedded CSS
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{content.company_name} - Equity Research Report</title>
    <style>
        :root {{
            --page-width: {self.params.page_width}pt;
            --page-height: {self.params.page_height}pt;
            --margin-left: {self.params.margin_left}pt;
            --margin-right: {self.params.margin_right}pt;
            --margin-top: {self.params.margin_top}pt;
            --margin-bottom: {self.params.margin_bottom}pt;
            --sidebar-width: {self.params.sidebar_width}pt;
            --gutter-width: {self.params.gutter_width}pt;
            --h1-size: {self.params.h1_font_size}pt;
            --h2-size: {self.params.h2_font_size}pt;
            --body-size: {self.params.body_font_size}pt;
        }}
        
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            font-family: 'Times New Roman', Times, serif;
            font-size: var(--body-size);
            line-height: 1.4;
            background: #f0f0f0;
            padding: 20px;
        }}
        
        .report-page {{
            width: var(--page-width);
            min-height: var(--page-height);
            background: white;
            margin: 0 auto 20px;
            padding: var(--margin-top) var(--margin-right) var(--margin-bottom) var(--margin-left);
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .header-band {{
            background: linear-gradient(to right, #0a3d62, #0d47a1);
            color: white;
            padding: 8pt 15pt;
            margin: calc(-1 * var(--margin-top)) calc(-1 * var(--margin-right)) 15pt calc(-1 * var(--margin-left));
        }}
        
        .two-column {{
            display: grid;
            grid-template-columns: 1fr var(--sidebar-width);
            gap: var(--gutter-width);
        }}
        
        h1 {{
            font-size: var(--h1-size);
            font-weight: bold;
            color: #0a3d62;
            margin-bottom: 10pt;
        }}
        
        h2 {{
            font-size: var(--h2-size);
            font-weight: bold;
            color: #0a3d62;
            margin: 12pt 0 6pt;
            border-bottom: 1pt solid #0a3d62;
            padding-bottom: 3pt;
        }}
        
        h3 {{
            font-size: 9pt;
            font-weight: bold;
            margin: 8pt 0 4pt;
        }}
        
        p {{ margin-bottom: 6pt; }}
        
        ul {{ margin-left: 15pt; margin-bottom: 8pt; }}
        li {{ margin-bottom: 4pt; }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 8pt;
            margin: 6pt 0;
        }}
        
        th, td {{
            padding: 4pt 6pt;
            border: 0.5pt solid #ccc;
            text-align: left;
        }}
        
        th {{
            background: #f5f5f5;
            font-weight: bold;
        }}
        
        .sidebar {{
            background: #f8f9fa;
            padding: 10pt;
            border-left: 2pt solid #0a3d62;
        }}
        
        .rating {{
            display: inline-block;
            background: #0a3d62;
            color: white;
            padding: 2pt 8pt;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="report-page">
        <div class="header-band">
            <strong>{content.sector}</strong> | Equity Research
        </div>
        
        <div class="two-column">
            <div class="main-content">
                <h1>{content.headline}</h1>
                <p><span class="rating">{content.rating}</span> | Price Target: {content.price_target}</p>
                
                <h2>Key Investment Points</h2>
                <ul>
                    {"".join(f"<li><strong>{point}</strong></li>" for point in content.key_points)}
                </ul>
                
                {"".join(self._render_section_html(section) for section in content.sections)}
            </div>
            
            <div class="sidebar">
                <h2>{content.company_name}</h2>
                <p><strong>{content.ticker}</strong></p>
                <p>Price: {content.current_price}</p>
                <p>Target: {content.price_target}</p>
                <p>Rating: {content.rating}</p>
                
                <h3>Analyst</h3>
                <p>{content.analyst_name}<br>{content.report_date}</p>
                
                {self._render_sidebar_data_html(content.sidebar_data)}
            </div>
        </div>
    </div>
</body>
</html>"""
    
    def _render_section_html(self, section: Dict[str, Any]) -> str:
        """Render a section as HTML."""
        html = f"<h2>{section.get('title', 'Section')}</h2>\n"
        
        if section.get("content"):
            paragraphs = section["content"].split("\n\n")
            for para in paragraphs:
                if para.strip():
                    html += f"<p>{para.strip()}</p>\n"
        
        if section.get("bullets"):
            html += "<ul>\n"
            for bullet in section["bullets"]:
                html += f"<li>{bullet}</li>\n"
            html += "</ul>\n"
        
        for table in section.get("tables", []):
            html += f"<h3>{table.get('title', '')}</h3>\n"
            html += "<table>\n<thead><tr>\n"
            for header in table.get("headers", []):
                html += f"<th>{header}</th>\n"
            html += "</tr></thead>\n<tbody>\n"
            for row in table.get("rows", []):
                html += "<tr>\n"
                for cell in row:
                    html += f"<td>{cell}</td>\n"
                html += "</tr>\n"
            html += "</tbody></table>\n"
        
        return html
    
    def _render_sidebar_data_html(self, sidebar_data: Dict[str, Any]) -> str:
        """Render sidebar data as HTML."""
        html = ""
        
        if sidebar_data.get("price_performance"):
            html += "<h3>Price Performance</h3>\n<table>\n"
            for period, value in sidebar_data["price_performance"].items():
                html += f"<tr><td>{period}</td><td>{value}</td></tr>\n"
            html += "</table>\n"
        
        if sidebar_data.get("key_metrics"):
            html += "<h3>Key Metrics</h3>\n<table>\n"
            for metric, value in sidebar_data["key_metrics"].items():
                html += f"<tr><td>{metric}</td><td>{value}</td></tr>\n"
            html += "</table>\n"
        
        return html


# Convenience functions
def build_pdf_report(
    content: Union[ReportContent, Dict[str, Any]],
    output_path: str,
    params: Optional[LayoutParams] = None
) -> str:
    """Build a PDF report."""
    builder = ReportBuilder(output_format=OutputFormat.PDF, layout_params=params)
    return builder.build(content, output_path)


def build_html_report(
    content: Union[ReportContent, Dict[str, Any]],
    output_path: str,
    params: Optional[LayoutParams] = None
) -> str:
    """Build an HTML report."""
    builder = ReportBuilder(output_format=OutputFormat.HTML, layout_params=params)
    return builder.build(content, output_path)


def build_markdown_report(
    content: Union[ReportContent, Dict[str, Any]],
    output_path: str,
    params: Optional[LayoutParams] = None
) -> str:
    """Build a Markdown report (for use with md-viewer.html)."""
    builder = ReportBuilder(output_format=OutputFormat.MARKDOWN, layout_params=params)
    return builder.build(content, output_path)
