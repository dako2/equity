"""
Content Placeholder System for Dynamic Report Generation

Redesigned based on JPM Equity Research Report analysis:
- 31 pages, 16,128 words (~520 words/page)
- ArialNarrow fonts: Headers 10pt, Body 6-8pt
- 20 tables, 33 figures
- 234 bullet points
- 2-column layouts with sidebar on summary page

Token Format: {{TYPE:id:min:max:current}}
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import re


class PlaceholderType(Enum):
    TEXT = "TEXT"           # Paragraphs of body text
    BULLETS = "BULLETS"     # Bullet point lists
    TABLE = "TABLE"         # Data/financial tables
    CHART = "CHART"         # Charts and figures
    SPACER = "SPACER"       # Flexible whitespace
    ANALYTICS = "ANALYTICS" # Key metrics/data points
    HEADER = "HEADER"       # Section headers
    TOC = "TOC"            # Table of contents entries


@dataclass
class Placeholder:
    """A content placeholder that the LLM can manipulate."""
    type: PlaceholderType
    id: str
    
    # Content bounds
    min_items: int = 1
    max_items: int = 10
    current_items: int = 3
    
    # Dimensions (for charts/tables)
    width: Optional[float] = None
    height: Optional[float] = None
    
    # Content
    content: List[str] = field(default_factory=list)
    visible: bool = True
    
    # Metadata
    description: str = ""
    page_hint: str = ""  # e.g., "summary", "body", "appendix"
    
    def to_token(self) -> str:
        """Generate the special token string."""
        return f"{{{{{self.type.value}:{self.id}:{self.min_items}:{self.max_items}:{self.current_items}}}}}"


@dataclass
class PlaceholderRegistry:
    """Registry of all placeholders in a report template."""
    placeholders: Dict[str, Placeholder] = field(default_factory=dict)
    
    def register(self, placeholder: Placeholder) -> str:
        """Register a placeholder and return its token."""
        self.placeholders[placeholder.id] = placeholder
        return placeholder.to_token()
    
    def get(self, id_: str) -> Optional[Placeholder]:
        return self.placeholders.get(id_)
    
    def apply_patch(self, patch: Dict[str, Any]) -> List[str]:
        """
        Apply an LLM-generated patch to placeholders.
        
        Patch format:
        {
            "placeholder_id": {
                "action": "expand" | "shrink" | "hide" | "show" | "set",
                "value": <int or content list>
            }
        }
        """
        changes = []
        
        for ph_id, ops in patch.items():
            if ph_id not in self.placeholders:
                continue
            
            ph = self.placeholders[ph_id]
            action = ops.get("action", "set")
            
            if action == "expand":
                delta = ops.get("value", 1)
                new_items = min(ph.current_items + delta, ph.max_items)
                if new_items != ph.current_items:
                    ph.current_items = new_items
                    changes.append(f"{ph_id}: expanded to {new_items} items")
            
            elif action == "shrink":
                delta = ops.get("value", 1)
                new_items = max(ph.current_items - delta, ph.min_items)
                if new_items != ph.current_items:
                    ph.current_items = new_items
                    changes.append(f"{ph_id}: shrunk to {new_items} items")
            
            elif action == "hide":
                ph.visible = False
                changes.append(f"{ph_id}: hidden")
            
            elif action == "show":
                ph.visible = True
                changes.append(f"{ph_id}: shown")
            
            elif action == "set":
                if "current_items" in ops:
                    new_items = max(ph.min_items, min(ops["current_items"], ph.max_items))
                    ph.current_items = new_items
                    changes.append(f"{ph_id}: set to {new_items} items")
                
                if "content" in ops:
                    ph.content = ops["content"][:ph.max_items]
                    changes.append(f"{ph_id}: content updated")
                
                if "width" in ops:
                    ph.width = ops["width"]
                if "height" in ops:
                    ph.height = ops["height"]
        
        return changes
    
    def to_manifest(self) -> Dict[str, Any]:
        """Export placeholder manifest for LLM context."""
        return {
            ph_id: {
                "type": ph.type.value,
                "min": ph.min_items,
                "max": ph.max_items,
                "current": ph.current_items,
                "visible": ph.visible,
                "description": ph.description,
                "page": ph.page_hint
            }
            for ph_id, ph in self.placeholders.items()
        }
    
    def get_fillable_summary(self) -> str:
        """Generate a summary for LLM of what can be adjusted."""
        lines = ["Content placeholders (based on JPM report structure):"]
        
        # Group by page hint
        by_page = {}
        for ph_id, ph in self.placeholders.items():
            page = ph.page_hint or "other"
            if page not in by_page:
                by_page[page] = []
            by_page[page].append((ph_id, ph))
        
        for page, items in sorted(by_page.items()):
            lines.append(f"\n  [{page.upper()}]")
            for ph_id, ph in items:
                status = "hidden" if not ph.visible else f"{ph.current_items}/{ph.max_items}"
                lines.append(f"    • {ph_id} ({ph.type.value}): {status} - {ph.description}")
        
        return "\n".join(lines)


def create_jpm_style_placeholders() -> PlaceholderRegistry:
    """
    Create placeholders matching JPM equity research report structure.
    
    Based on analysis of JPM-Equity-Research-Report-Hulu.pdf:
    - 31 pages, ~520 words/page
    - Page 1: Summary with sidebar (rating, price, analysts, key points)
    - Page 2: Executive summary continuation
    - Page 3: Table of Contents
    - Pages 4-21: Body sections (Hulu Overview, Strategy, Revenue Model, SVOD Industry, etc.)
    - Pages 22-27: Financial tables and valuation
    - Pages 28-31: Disclosures
    """
    registry = PlaceholderRegistry()
    
    # ============================================
    # PAGE 1: SUMMARY PAGE (2-column + sidebar)
    # ============================================
    
    # Main column - headline and key points
    registry.register(Placeholder(
        type=PlaceholderType.TEXT,
        id="headline",
        min_items=1, max_items=2, current_items=1,
        description="Main headline (catchy, specific thesis)",
        page_hint="summary"
    ))
    
    registry.register(Placeholder(
        type=PlaceholderType.BULLETS,
        id="key_points",
        min_items=3, max_items=6, current_items=4,
        description="Key investment points (3-6 bullets, ~20 words each)",
        page_hint="summary"
    ))
    
    registry.register(Placeholder(
        type=PlaceholderType.TEXT,
        id="summary_body",
        min_items=2, max_items=5, current_items=3,
        description="Summary paragraphs explaining thesis (~50 words each)",
        page_hint="summary"
    ))
    
    # Sidebar - company info card
    registry.register(Placeholder(
        type=PlaceholderType.ANALYTICS,
        id="rating_badge",
        min_items=1, max_items=1, current_items=1,
        description="Rating (Overweight/Neutral/Underweight)",
        page_hint="summary"
    ))
    
    registry.register(Placeholder(
        type=PlaceholderType.ANALYTICS,
        id="price_info",
        min_items=4, max_items=8, current_items=6,
        description="Price, target, upside, 52-week range, market cap",
        page_hint="summary"
    ))
    
    registry.register(Placeholder(
        type=PlaceholderType.TEXT,
        id="analysts",
        min_items=1, max_items=4, current_items=2,
        description="Analyst names, phones, emails",
        page_hint="summary"
    ))
    
    registry.register(Placeholder(
        type=PlaceholderType.CHART,
        id="price_chart",
        min_items=0, max_items=1, current_items=1,
        width=140, height=80,
        description="12-month price performance chart",
        page_hint="summary"
    ))
    
    registry.register(Placeholder(
        type=PlaceholderType.TABLE,
        id="price_perf_table",
        min_items=2, max_items=4, current_items=3,
        width=140, height=50,
        description="Price performance (YTD, 1m, 3m, 12m) abs & relative",
        page_hint="summary"
    ))
    
    # Bottom band - EPS and company data
    registry.register(Placeholder(
        type=PlaceholderType.TABLE,
        id="eps_quarterly",
        min_items=4, max_items=8, current_items=6,
        width=320, height=100,
        description="Quarterly EPS table (Q1-Q4 for 2-3 years + estimates)",
        page_hint="summary"
    ))
    
    registry.register(Placeholder(
        type=PlaceholderType.TABLE,
        id="company_data",
        min_items=5, max_items=12, current_items=8,
        width=180, height=100,
        description="Company data snapshot (price, mktcap, shares, P/E, etc.)",
        page_hint="summary"
    ))
    
    # ============================================
    # PAGE 2-3: EXECUTIVE SUMMARY + TOC
    # ============================================
    
    registry.register(Placeholder(
        type=PlaceholderType.TEXT,
        id="exec_summary",
        min_items=3, max_items=8, current_items=5,
        description="Executive summary paragraphs (~100 words each)",
        page_hint="executive"
    ))
    
    registry.register(Placeholder(
        type=PlaceholderType.BULLETS,
        id="valuation_summary",
        min_items=2, max_items=5, current_items=3,
        description="Valuation highlights and price target justification",
        page_hint="executive"
    ))
    
    registry.register(Placeholder(
        type=PlaceholderType.TOC,
        id="toc_entries",
        min_items=5, max_items=15, current_items=8,
        description="Table of contents entries",
        page_hint="toc"
    ))
    
    # ============================================
    # BODY SECTIONS (Pages 4-21 typically)
    # ============================================
    
    # Company Overview section
    registry.register(Placeholder(
        type=PlaceholderType.HEADER,
        id="section_overview",
        min_items=1, max_items=1, current_items=1,
        description="Company/Business Overview section header",
        page_hint="body"
    ))
    
    registry.register(Placeholder(
        type=PlaceholderType.TEXT,
        id="overview_text",
        min_items=4, max_items=12, current_items=6,
        description="Company overview paragraphs (~80 words each)",
        page_hint="body"
    ))
    
    registry.register(Placeholder(
        type=PlaceholderType.CHART,
        id="overview_chart_1",
        min_items=0, max_items=1, current_items=1,
        width=250, height=150,
        description="Key metric chart (subscribers, revenue trend, etc.)",
        page_hint="body"
    ))
    
    registry.register(Placeholder(
        type=PlaceholderType.CHART,
        id="overview_chart_2",
        min_items=0, max_items=1, current_items=1,
        width=250, height=150,
        description="Secondary metric chart (engagement, growth, etc.)",
        page_hint="body"
    ))
    
    # Strategy/Changes section
    registry.register(Placeholder(
        type=PlaceholderType.HEADER,
        id="section_strategy",
        min_items=1, max_items=1, current_items=1,
        description="Strategic Changes/Initiatives section header",
        page_hint="body"
    ))
    
    registry.register(Placeholder(
        type=PlaceholderType.TEXT,
        id="strategy_text",
        min_items=3, max_items=10, current_items=5,
        description="Strategy discussion paragraphs",
        page_hint="body"
    ))
    
    registry.register(Placeholder(
        type=PlaceholderType.BULLETS,
        id="strategy_points",
        min_items=3, max_items=8, current_items=5,
        description="Key strategic initiatives as bullets",
        page_hint="body"
    ))
    
    # Revenue/Business Model section
    registry.register(Placeholder(
        type=PlaceholderType.HEADER,
        id="section_revenue",
        min_items=1, max_items=1, current_items=1,
        description="Revenue/Business Model section header",
        page_hint="body"
    ))
    
    registry.register(Placeholder(
        type=PlaceholderType.TEXT,
        id="revenue_text",
        min_items=3, max_items=8, current_items=4,
        description="Revenue model explanation",
        page_hint="body"
    ))
    
    registry.register(Placeholder(
        type=PlaceholderType.TABLE,
        id="revenue_breakdown",
        min_items=3, max_items=10, current_items=6,
        width=400, height=120,
        description="Revenue breakdown by segment/geography",
        page_hint="body"
    ))
    
    # Industry Overview section
    registry.register(Placeholder(
        type=PlaceholderType.HEADER,
        id="section_industry",
        min_items=1, max_items=1, current_items=1,
        description="Industry Overview section header",
        page_hint="body"
    ))
    
    registry.register(Placeholder(
        type=PlaceholderType.TEXT,
        id="industry_text",
        min_items=4, max_items=12, current_items=6,
        description="Industry analysis paragraphs",
        page_hint="body"
    ))
    
    registry.register(Placeholder(
        type=PlaceholderType.TABLE,
        id="competitor_comparison",
        min_items=4, max_items=15, current_items=8,
        width=500, height=200,
        description="Competitive comparison table",
        page_hint="body"
    ))
    
    registry.register(Placeholder(
        type=PlaceholderType.CHART,
        id="market_share_chart",
        min_items=0, max_items=1, current_items=1,
        width=300, height=180,
        description="Market share or industry trends chart",
        page_hint="body"
    ))
    
    # ============================================
    # INVESTMENT THESIS & VALUATION (Pages 21-27)
    # ============================================
    
    registry.register(Placeholder(
        type=PlaceholderType.HEADER,
        id="section_thesis",
        min_items=1, max_items=1, current_items=1,
        description="Investment Thesis section header",
        page_hint="valuation"
    ))
    
    registry.register(Placeholder(
        type=PlaceholderType.TEXT,
        id="thesis_text",
        min_items=3, max_items=8, current_items=4,
        description="Investment thesis detailed explanation",
        page_hint="valuation"
    ))
    
    registry.register(Placeholder(
        type=PlaceholderType.HEADER,
        id="section_valuation",
        min_items=1, max_items=1, current_items=1,
        description="Valuation Analysis section header",
        page_hint="valuation"
    ))
    
    registry.register(Placeholder(
        type=PlaceholderType.TEXT,
        id="valuation_text",
        min_items=2, max_items=6, current_items=3,
        description="Valuation methodology explanation",
        page_hint="valuation"
    ))
    
    registry.register(Placeholder(
        type=PlaceholderType.TABLE,
        id="dcf_assumptions",
        min_items=5, max_items=15, current_items=10,
        width=450, height=200,
        description="DCF assumptions table (WACC, terminal growth, etc.)",
        page_hint="valuation"
    ))
    
    registry.register(Placeholder(
        type=PlaceholderType.TABLE,
        id="valuation_matrix",
        min_items=4, max_items=12, current_items=8,
        width=450, height=150,
        description="Valuation multiples comparison",
        page_hint="valuation"
    ))
    
    # Financial Statements
    registry.register(Placeholder(
        type=PlaceholderType.TABLE,
        id="income_statement",
        min_items=10, max_items=25, current_items=15,
        width=500, height=300,
        description="Income statement (5 years historical + 3 projected)",
        page_hint="financials"
    ))
    
    registry.register(Placeholder(
        type=PlaceholderType.TABLE,
        id="balance_sheet",
        min_items=8, max_items=20, current_items=12,
        width=500, height=250,
        description="Balance sheet summary",
        page_hint="financials"
    ))
    
    registry.register(Placeholder(
        type=PlaceholderType.TABLE,
        id="cash_flow",
        min_items=6, max_items=15, current_items=10,
        width=500, height=200,
        description="Cash flow statement summary",
        page_hint="financials"
    ))
    
    # ============================================
    # RISKS (Usually before disclosures)
    # ============================================
    
    registry.register(Placeholder(
        type=PlaceholderType.HEADER,
        id="section_risks",
        min_items=1, max_items=1, current_items=1,
        description="Investment Risks section header",
        page_hint="risks"
    ))
    
    registry.register(Placeholder(
        type=PlaceholderType.BULLETS,
        id="risk_factors",
        min_items=4, max_items=12, current_items=6,
        description="Key risk factors (competition, regulatory, macro, etc.)",
        page_hint="risks"
    ))
    
    registry.register(Placeholder(
        type=PlaceholderType.TEXT,
        id="risk_details",
        min_items=2, max_items=6, current_items=3,
        description="Detailed risk discussion paragraphs",
        page_hint="risks"
    ))
    
    # ============================================
    # DISCLOSURES (Required boilerplate)
    # ============================================
    
    registry.register(Placeholder(
        type=PlaceholderType.TEXT,
        id="disclosures",
        min_items=3, max_items=10, current_items=5,
        description="Analyst certification and legal disclosures",
        page_hint="disclosures"
    ))
    
    # ============================================
    # FLEXIBLE SPACERS
    # ============================================
    
    registry.register(Placeholder(
        type=PlaceholderType.SPACER,
        id="spacer_post_summary",
        min_items=6, max_items=40, current_items=12,
        description="Space after summary before bottom band",
        page_hint="summary"
    ))
    
    registry.register(Placeholder(
        type=PlaceholderType.SPACER,
        id="spacer_section",
        min_items=8, max_items=24, current_items=12,
        description="Space between major sections",
        page_hint="body"
    ))
    
    registry.register(Placeholder(
        type=PlaceholderType.SPACER,
        id="spacer_chart",
        min_items=4, max_items=16, current_items=8,
        description="Space around charts/figures",
        page_hint="body"
    ))
    
    return registry


# Backwards compatibility
def create_equity_report_placeholders() -> PlaceholderRegistry:
    """Alias for create_jpm_style_placeholders."""
    return create_jpm_style_placeholders()


# Content density guidelines based on JPM analysis
JPM_CONTENT_GUIDELINES = {
    "words_per_page": 520,
    "bullets_per_report": 234,
    "tables_per_report": 20,
    "figures_per_report": 33,
    "header_font_size": 10.0,
    "body_font_size": 6.1,
    "summary_page": {
        "words": 131,
        "bullets": 4,
        "tables": 2,
        "layout": "2-column with sidebar"
    },
    "body_page": {
        "words": 400,
        "bullets": 8,
        "tables": 1,
        "charts": 1
    }
}


def get_content_guidelines() -> Dict[str, Any]:
    """Get content density guidelines based on JPM report analysis."""
    return JPM_CONTENT_GUIDELINES


if __name__ == "__main__":
    # Demo the placeholder system
    registry = create_jpm_style_placeholders()
    
    print("="*60)
    print("JPM-Style Equity Report Placeholders")
    print("="*60)
    print(registry.get_fillable_summary())
    
    print("\n" + "="*60)
    print("Content Guidelines (from JPM analysis)")
    print("="*60)
    for key, val in JPM_CONTENT_GUIDELINES.items():
        print(f"  {key}: {val}")
