"""
Markdown to Equity Research Report Converter

Takes an arbitrary Markdown file and converts it into a professionally
formatted equity research report following the JPM-style template.

Features:
- LLM-powered content restructuring
- Automatic section detection and organization
- Two output formats: PDF and HTML
- Template-aware formatting
"""

import os
import json
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from .layout_params import LayoutParams
from .content_placeholders import PlaceholderRegistry, create_jpm_style_placeholders


# Template structure for equity research reports
REPORT_STRUCTURE = {
    "summary_page": {
        "headline": "Main investment thesis headline",
        "rating": "Overweight/Neutral/Underweight",
        "price_target": "Target price",
        "key_points": ["Key investment point 1", "Key investment point 2"],
        "summary": "Executive summary paragraph",
        "sidebar": {
            "company_name": "Company Name",
            "ticker": "TICKER",
            "current_price": "Current stock price",
            "analyst": "Analyst name",
            "date": "Report date"
        }
    },
    "sections": [
        "Executive Summary",
        "Investment Thesis",
        "Company Overview", 
        "Industry Analysis",
        "Financial Analysis",
        "Valuation",
        "Risks",
        "Disclosures"
    ]
}


RESTRUCTURE_SYSTEM_PROMPT = """You are an expert equity research analyst who restructures content into professional investment reports.

Your task is to take raw markdown content and restructure it into a JPM-style equity research report format.

## OUTPUT STRUCTURE:

You must output valid JSON with this exact structure:

```json
{
  "metadata": {
    "company_name": "Company Name",
    "ticker": "TICKER",
    "rating": "Overweight|Neutral|Underweight",
    "current_price": "$XX.XX",
    "price_target": "$XX.XX",
    "analyst_name": "Analyst Name",
    "sector": "Industry Sector",
    "report_date": "YYYY-MM-DD"
  },
  "headline": "Compelling 6-10 word headline summarizing the investment thesis",
  "key_points": [
    "Key investment point 1 (1 sentence)",
    "Key investment point 2 (1 sentence)",
    "Key investment point 3 (1 sentence)",
    "Key investment point 4 (1 sentence)"
  ],
  "sections": [
    {
      "title": "Executive Summary",
      "content": "2-3 paragraph executive summary...",
      "bullets": ["Optional bullet 1", "Optional bullet 2"]
    },
    {
      "title": "Investment Thesis",
      "content": "Investment thesis content...",
      "bullets": []
    },
    {
      "title": "Company Overview",
      "content": "Company background...",
      "bullets": []
    },
    {
      "title": "Industry Analysis",
      "content": "Industry context...",
      "bullets": []
    },
    {
      "title": "Financial Analysis",
      "content": "Financial discussion...",
      "tables": [
        {
          "title": "Key Financials",
          "headers": ["Metric", "2024E", "2025E", "2026E"],
          "rows": [
            ["Revenue ($B)", "10.2", "12.5", "15.1"],
            ["EPS ($)", "2.45", "3.12", "4.05"]
          ]
        }
      ]
    },
    {
      "title": "Valuation",
      "content": "Valuation methodology and conclusion...",
      "bullets": []
    },
    {
      "title": "Investment Risks",
      "content": "Risk introduction...",
      "bullets": ["Risk factor 1", "Risk factor 2", "Risk factor 3"]
    }
  ],
  "sidebar_data": {
    "price_performance": {
      "1M": "+X.X%",
      "3M": "+X.X%",
      "YTD": "+X.X%",
      "1Y": "+X.X%"
    },
    "key_metrics": {
      "Market Cap": "$XXB",
      "P/E (FY25E)": "XX.Xx",
      "EV/EBITDA": "XX.Xx"
    }
  }
}
```

## RULES:
1. Extract ALL relevant information from the source content
2. If data is missing, make reasonable estimates or mark as "[TBD]"
3. Maintain factual accuracy - don't invent financial numbers
4. Use professional, concise language
5. Each section should be 1-3 paragraphs
6. Key points should be punchy, actionable insights
7. Tables should have proper headers and numeric data
"""


class MarkdownToReportConverter:
    """
    Converts arbitrary Markdown to equity research report format.
    """
    
    def __init__(
        self,
        provider: str = "anthropic",
        api_key: Optional[str] = None
    ):
        """
        Initialize the converter.
        
        Args:
            provider: LLM provider ("anthropic", "openai", "xai")
            api_key: API key (or from environment)
        """
        self.provider = provider.lower()
        self.api_key = api_key or self._get_api_key()
        self.client = self._create_client()
        
    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment."""
        env_vars = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "xai": "XAI_API_KEY"
        }
        return os.getenv(env_vars.get(self.provider, ""))
    
    def _create_client(self):
        """Create the API client."""
        if not self.api_key:
            print(f"⚠️  No API key for {self.provider}")
            return None
            
        if self.provider == "anthropic":
            try:
                import anthropic
                return anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                print("⚠️  anthropic package not installed")
                return None
        elif self.provider in ["openai", "xai"]:
            from openai import OpenAI
            base_url = "https://api.x.ai/v1" if self.provider == "xai" else None
            return OpenAI(api_key=self.api_key, base_url=base_url)
        
        return None
    
    def read_markdown(self, md_path: str) -> str:
        """Read markdown file content."""
        with open(md_path, "r", encoding="utf-8") as f:
            return f.read()
    
    def restructure_content(
        self,
        markdown_content: str,
        company_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Use LLM to restructure markdown into report format.
        
        Args:
            markdown_content: Raw markdown content
            company_hint: Optional company name hint
        
        Returns:
            Structured report data dictionary
        """
        if not self.client:
            return self._fallback_structure(markdown_content)
        
        user_prompt = f"""Convert this markdown content into a structured equity research report.

{f"Company hint: {company_hint}" if company_hint else ""}

---
SOURCE MARKDOWN:
---

{markdown_content}

---

Restructure this into the JSON format specified. Extract all relevant data and organize it professionally."""

        try:
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=4000,
                    system=RESTRUCTURE_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}]
                )
                response_text = response.content[0].text
            else:
                response = self.client.chat.completions.create(
                    model="gpt-4o" if self.provider == "openai" else "grok-3",
                    max_tokens=4000,
                    messages=[
                        {"role": "system", "content": RESTRUCTURE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                response_text = response.choices[0].message.content
            
            return self._parse_response(response_text)
            
        except Exception as e:
            print(f"⚠️  LLM restructuring failed: {e}")
            return self._fallback_structure(markdown_content)
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response into structured data."""
        try:
            # Extract JSON from response
            text = response_text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            return json.loads(text)
        except (json.JSONDecodeError, IndexError) as e:
            print(f"⚠️  Failed to parse response: {e}")
            return {}
    
    def _fallback_structure(self, markdown_content: str) -> Dict[str, Any]:
        """Create a basic structure from markdown without LLM."""
        lines = markdown_content.strip().split("\n")
        
        # Extract title from first H1
        title = "Investment Report"
        for line in lines:
            if line.startswith("# "):
                title = line[2:].strip()
                break
        
        # Extract sections
        sections = []
        current_section = {"title": "Overview", "content": "", "bullets": []}
        
        for line in lines:
            if line.startswith("## "):
                if current_section["content"] or current_section["bullets"]:
                    sections.append(current_section)
                current_section = {"title": line[3:].strip(), "content": "", "bullets": []}
            elif line.startswith("- ") or line.startswith("* "):
                current_section["bullets"].append(line[2:].strip())
            elif line.strip():
                current_section["content"] += line + "\n"
        
        if current_section["content"] or current_section["bullets"]:
            sections.append(current_section)
        
        return {
            "metadata": {
                "company_name": title,
                "ticker": "[TICKER]",
                "rating": "Neutral",
                "current_price": "[TBD]",
                "price_target": "[TBD]",
                "analyst_name": "[Analyst]",
                "sector": "[Sector]",
                "report_date": datetime.now().strftime("%Y-%m-%d")
            },
            "headline": title,
            "key_points": ["See full report for details"],
            "sections": sections,
            "sidebar_data": {}
        }
    
    def convert_to_html(
        self,
        report_data: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Convert structured report data to HTML.
        
        Args:
            report_data: Structured report dictionary
            output_path: Optional path to save HTML
        
        Returns:
            HTML content string
        """
        metadata = report_data.get("metadata", {})
        
        # Build main content
        main_content = f"""
<h1>{report_data.get('headline', 'Investment Report')}</h1>
<p><strong>Rating:</strong> {metadata.get('rating', 'Neutral')} | <strong>Price Target:</strong> {metadata.get('price_target', 'TBD')}</p>

<h2>Key Investment Points</h2>
<ul>
"""
        for point in report_data.get("key_points", []):
            main_content += f"<li><strong>{point}</strong></li>\n"
        
        main_content += "</ul>\n"
        
        # Add sections
        for section in report_data.get("sections", []):
            main_content += f"\n<h2>{section.get('title', 'Section')}</h2>\n"
            
            content = section.get("content", "").strip()
            if content:
                paragraphs = content.split("\n\n")
                for para in paragraphs:
                    if para.strip():
                        main_content += f"<p>{para.strip()}</p>\n"
            
            bullets = section.get("bullets", [])
            if bullets:
                main_content += "<ul>\n"
                for bullet in bullets:
                    main_content += f"<li>{bullet}</li>\n"
                main_content += "</ul>\n"
            
            tables = section.get("tables", [])
            for table in tables:
                main_content += f"\n<h3>{table.get('title', 'Table')}</h3>\n"
                main_content += "<table>\n<thead><tr>\n"
                for header in table.get("headers", []):
                    main_content += f"<th>{header}</th>\n"
                main_content += "</tr></thead>\n<tbody>\n"
                for row in table.get("rows", []):
                    main_content += "<tr>\n"
                    for cell in row:
                        main_content += f"<td>{cell}</td>\n"
                    main_content += "</tr>\n"
                main_content += "</tbody></table>\n"
        
        # Build sidebar
        sidebar_content = f"""
<h2>{metadata.get('company_name', 'Company')}</h2>
<p><strong>{metadata.get('ticker', 'TICKER')}</strong></p>

<p><strong>Price:</strong> {metadata.get('current_price', 'TBD')}</p>
<p><strong>Target:</strong> {metadata.get('price_target', 'TBD')}</p>
<p><strong>Rating:</strong> {metadata.get('rating', 'Neutral')}</p>

<h3>Analyst</h3>
<p>{metadata.get('analyst_name', 'Analyst')}</p>
<p>{metadata.get('report_date', datetime.now().strftime('%Y-%m-%d'))}</p>
"""
        
        sidebar_data = report_data.get("sidebar_data", {})
        
        if sidebar_data.get("price_performance"):
            sidebar_content += "\n<h3>Price Performance</h3>\n<table>\n"
            for period, value in sidebar_data["price_performance"].items():
                sidebar_content += f"<tr><td>{period}</td><td>{value}</td></tr>\n"
            sidebar_content += "</table>\n"
        
        if sidebar_data.get("key_metrics"):
            sidebar_content += "\n<h3>Key Metrics</h3>\n<table>\n"
            for metric, value in sidebar_data["key_metrics"].items():
                sidebar_content += f"<tr><td>{metric}</td><td>{value}</td></tr>\n"
            sidebar_content += "</table>\n"
        
        # Combine with template markers
        html_output = f"""---MAIN---
{main_content}
---SIDEBAR---
{sidebar_content}
---ENDSIDEBAR---
"""
        
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_output)
        
        return html_output
    
    def convert_to_pdf(
        self,
        report_data: Dict[str, Any],
        output_path: str,
        params: Optional[LayoutParams] = None
    ) -> str:
        """
        Convert structured report data to PDF.
        
        Args:
            report_data: Structured report dictionary
            output_path: Path to save PDF
            params: Optional layout parameters
        
        Returns:
            Path to generated PDF
        """
        from .dynamic_report_builder import build_report_from_data
        
        params = params or LayoutParams()
        registry = create_jpm_style_placeholders()
        
        # Build PDF
        result = build_report_from_data(report_data, params, registry, output_path)
        
        return output_path
    
    def convert(
        self,
        md_path: str,
        output_format: str = "both",
        output_dir: Optional[str] = None,
        company_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Convert a Markdown file to equity research report.
        
        Args:
            md_path: Path to input Markdown file
            output_format: "html", "pdf", or "both"
            output_dir: Output directory (default: same as input)
            company_hint: Optional company name hint
        
        Returns:
            Dictionary with output paths and report data
        """
        print(f"\n{'='*60}")
        print("📄 MARKDOWN TO EQUITY REPORT CONVERTER")
        print(f"{'='*60}")
        print(f"  Input: {md_path}")
        print(f"  Format: {output_format}")
        print(f"{'='*60}\n")
        
        # Setup output directory
        if output_dir is None:
            output_dir = os.path.dirname(md_path) or "."
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = Path(md_path).stem
        
        # Read markdown
        print("📖 Reading markdown file...")
        markdown_content = self.read_markdown(md_path)
        print(f"   {len(markdown_content)} characters read")
        
        # Restructure with LLM
        print("\n🤖 Restructuring content with AI...")
        report_data = self.restructure_content(markdown_content, company_hint)
        
        if report_data.get("metadata"):
            print(f"   Company: {report_data['metadata'].get('company_name', 'Unknown')}")
            print(f"   Rating: {report_data['metadata'].get('rating', 'Unknown')}")
            print(f"   Sections: {len(report_data.get('sections', []))}")
        
        result = {
            "input": md_path,
            "report_data": report_data,
            "outputs": {}
        }
        
        # Generate HTML
        if output_format in ["html", "both"]:
            print("\n📝 Generating HTML...")
            html_path = os.path.join(output_dir, f"{base_name}_report.md")
            self.convert_to_html(report_data, html_path)
            result["outputs"]["html"] = html_path
            print(f"   ✅ Saved: {html_path}")
        
        # Generate PDF
        if output_format in ["pdf", "both"]:
            print("\n📄 Generating PDF...")
            pdf_path = os.path.join(output_dir, f"{base_name}_report.pdf")
            try:
                self.convert_to_pdf(report_data, pdf_path)
                result["outputs"]["pdf"] = pdf_path
                print(f"   ✅ Saved: {pdf_path}")
            except Exception as e:
                print(f"   ⚠️  PDF generation failed: {e}")
        
        # Save structured data
        json_path = os.path.join(output_dir, f"{base_name}_data.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        result["outputs"]["json"] = json_path
        print(f"\n💾 Data saved: {json_path}")
        
        print(f"\n{'='*60}")
        print("✅ CONVERSION COMPLETE")
        print(f"{'='*60}\n")
        
        return result


def convert_markdown_to_report(
    md_path: str,
    output_format: str = "both",
    output_dir: Optional[str] = None,
    provider: str = "xai",
    company_hint: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to convert Markdown to equity report.
    
    Args:
        md_path: Path to Markdown file
        output_format: "html", "pdf", or "both"
        output_dir: Output directory
        provider: LLM provider
        company_hint: Company name hint
    
    Returns:
        Conversion result dictionary
    """
    converter = MarkdownToReportConverter(provider=provider)
    return converter.convert(
        md_path=md_path,
        output_format=output_format,
        output_dir=output_dir,
        company_hint=company_hint
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python md_to_report.py <markdown_file> [--output-dir DIR] [--format html|pdf|both]")
        print("\nExample:")
        print("  python md_to_report.py my_analysis.md --format both")
        sys.exit(1)
    
    md_path = sys.argv[1]
    output_format = "both"
    output_dir = None
    
    if "--format" in sys.argv:
        idx = sys.argv.index("--format")
        if idx + 1 < len(sys.argv):
            output_format = sys.argv[idx + 1]
    
    if "--output-dir" in sys.argv:
        idx = sys.argv.index("--output-dir")
        if idx + 1 < len(sys.argv):
            output_dir = sys.argv[idx + 1]
    
    result = convert_markdown_to_report(
        md_path=md_path,
        output_format=output_format,
        output_dir=output_dir
    )
    
    print("\nOutputs:")
    for fmt, path in result["outputs"].items():
        print(f"  {fmt}: {path}")
