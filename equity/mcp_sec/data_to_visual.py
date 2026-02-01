"""
Data to Visual Converter

LLM-powered tool that transforms raw data into:
  - Markdown tables
  - ASCII charts
  - Mermaid diagrams
  - SVG charts (base64 encoded)

Input: Any structured data (dict, list, DataFrame-like)
Output: Formatted markdown with tables/charts/images
"""

import asyncio
import json
import base64
import os
import re
from typing import Dict, Any, List, Optional, Union, Literal
from dataclasses import dataclass
from datetime import datetime
from openai import AsyncOpenAI


# ============================================================================
# OUTPUT TYPES
# ============================================================================

OutputType = Literal["table", "bar_chart", "line_chart", "pie_chart", "diagram", "auto"]


@dataclass
class VisualOutput:
    """Output from data visualization."""
    output_type: str
    markdown: str
    raw_data: Any = None
    svg: Optional[str] = None  # Base64 encoded SVG
    mermaid: Optional[str] = None  # Mermaid diagram code


# ============================================================================
# ASCII CHART GENERATORS (No dependencies)
# ============================================================================

def generate_ascii_bar_chart(
    data: Dict[str, float],
    title: str = "",
    max_width: int = 40
) -> str:
    """Generate ASCII horizontal bar chart."""
    if not data:
        return "*No data*"
    
    max_val = max(data.values()) if data.values() else 1
    max_label = max(len(str(k)) for k in data.keys())
    
    lines = []
    if title:
        lines.append(f"**{title}**\n")
    
    lines.append("```")
    for label, value in data.items():
        bar_len = int((value / max_val) * max_width) if max_val > 0 else 0
        bar = "█" * bar_len
        lines.append(f"{str(label):>{max_label}} │{bar} {value:,.1f}")
    lines.append("```")
    
    return "\n".join(lines)


def generate_ascii_line_chart(
    data: List[float],
    labels: Optional[List[str]] = None,
    title: str = "",
    height: int = 10,
    width: int = 50
) -> str:
    """Generate ASCII line chart."""
    if not data:
        return "*No data*"
    
    min_val = min(data)
    max_val = max(data)
    range_val = max_val - min_val if max_val != min_val else 1
    
    # Normalize to height
    normalized = [int((v - min_val) / range_val * (height - 1)) for v in data]
    
    lines = []
    if title:
        lines.append(f"**{title}**\n")
    
    lines.append("```")
    
    # Build chart from top to bottom
    for row in range(height - 1, -1, -1):
        line = ""
        for i, norm_val in enumerate(normalized):
            if norm_val == row:
                line += "●"
            elif i > 0 and (
                (normalized[i-1] < row < norm_val) or 
                (norm_val < row < normalized[i-1])
            ):
                line += "│"
            elif norm_val > row:
                line += " "
            else:
                line += " "
        
        # Y-axis label
        y_val = min_val + (row / (height - 1)) * range_val
        lines.append(f"{y_val:>8.1f} │{line}")
    
    # X-axis
    lines.append(" " * 9 + "└" + "─" * len(data))
    
    lines.append("```")
    
    return "\n".join(lines)


def generate_sparkline(data: List[float]) -> str:
    """Generate inline sparkline."""
    if not data:
        return ""
    
    chars = "▁▂▃▄▅▆▇█"
    min_val = min(data)
    max_val = max(data)
    range_val = max_val - min_val if max_val != min_val else 1
    
    result = ""
    for v in data:
        idx = int((v - min_val) / range_val * 7)
        result += chars[min(idx, 7)]
    
    return result


# ============================================================================
# MARKDOWN TABLE GENERATOR
# ============================================================================

def generate_markdown_table(
    data: Union[Dict, List],
    title: str = "",
    headers: Optional[List[str]] = None
) -> str:
    """Generate markdown table from data."""
    
    lines = []
    if title:
        lines.append(f"### {title}\n")
    
    # Handle dict
    if isinstance(data, dict):
        # Check if values are dicts (nested)
        if data and isinstance(list(data.values())[0], dict):
            # Nested dict -> 2D table
            all_keys = set()
            for v in data.values():
                if isinstance(v, dict):
                    all_keys.update(v.keys())
            
            cols = sorted(all_keys)
            headers = headers or [""] + cols
            
            lines.append("| " + " | ".join(str(h) for h in headers) + " |")
            lines.append("|" + "|".join(["---"] * len(headers)) + "|")
            
            for row_key, row_data in data.items():
                row = [str(row_key)]
                for col in cols:
                    val = row_data.get(col, "-") if isinstance(row_data, dict) else "-"
                    row.append(str(val))
                lines.append("| " + " | ".join(row) + " |")
        else:
            # Simple key-value dict
            headers = headers or ["Metric", "Value"]
            lines.append(f"| {headers[0]} | {headers[1]} |")
            lines.append("|---|---|")
            for k, v in data.items():
                lines.append(f"| {k} | {v} |")
    
    # Handle list of dicts
    elif isinstance(data, list) and data and isinstance(data[0], dict):
        all_keys = []
        for item in data:
            for k in item.keys():
                if k not in all_keys:
                    all_keys.append(k)
        
        headers = headers or all_keys
        lines.append("| " + " | ".join(str(h) for h in headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        
        for item in data:
            row = [str(item.get(h, "-")) for h in all_keys]
            lines.append("| " + " | ".join(row) + " |")
    
    # Handle list of lists
    elif isinstance(data, list) and data and isinstance(data[0], list):
        if headers:
            lines.append("| " + " | ".join(str(h) for h in headers) + " |")
            lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        
        for row in data:
            lines.append("| " + " | ".join(str(v) for v in row) + " |")
    
    else:
        lines.append(f"```\n{data}\n```")
    
    return "\n".join(lines)


# ============================================================================
# MERMAID DIAGRAM GENERATOR
# ============================================================================

def generate_mermaid_pie(data: Dict[str, float], title: str = "") -> str:
    """Generate Mermaid pie chart."""
    lines = ["```mermaid"]
    lines.append(f'pie title {title}' if title else "pie")
    
    for label, value in data.items():
        lines.append(f'    "{label}" : {value}')
    
    lines.append("```")
    return "\n".join(lines)


def generate_mermaid_bar(data: Dict[str, float], title: str = "") -> str:
    """Generate Mermaid bar chart (xychart)."""
    lines = ["```mermaid"]
    lines.append("xychart-beta horizontal")
    if title:
        lines.append(f'    title "{title}"')
    
    labels = list(data.keys())
    values = list(data.values())
    
    labels_str = ", ".join(f'"{l}"' for l in labels)
    values_str = ", ".join(str(v) for v in values)
    lines.append(f'    x-axis [{labels_str}]')
    lines.append(f'    bar [{values_str}]')
    
    lines.append("```")
    return "\n".join(lines)


def generate_mermaid_flowchart(nodes: List[Dict[str, str]], title: str = "") -> str:
    """Generate Mermaid flowchart."""
    lines = ["```mermaid"]
    lines.append("flowchart TD")
    
    for node in nodes:
        from_node = node.get("from", "A")
        to_node = node.get("to", "B")
        label = node.get("label", "")
        
        if label:
            lines.append(f"    {from_node} -->|{label}| {to_node}")
        else:
            lines.append(f"    {from_node} --> {to_node}")
    
    lines.append("```")
    return "\n".join(lines)


# ============================================================================
# LLM-POWERED DATA TRANSFORMER
# ============================================================================

class DataToVisualConverter:
    """LLM-powered data to visual converter."""
    
    SYSTEM_PROMPT = """You are a data visualization expert. Your job is to transform raw data into clear, professional visualizations.

When given data, you should:
1. Analyze the data structure
2. Determine the best visualization type
3. Generate the appropriate markdown/chart code

Available output formats:
- **table**: Markdown table for structured data
- **bar_chart**: ASCII or Mermaid bar chart for comparisons
- **line_chart**: ASCII line chart for trends
- **pie_chart**: Mermaid pie chart for proportions
- **sparkline**: Inline mini chart for quick trends
- **diagram**: Mermaid flowchart for processes

Always format numbers nicely (use commas, appropriate decimals).
Add clear titles and labels.
"""
    
    def __init__(self, provider: str = "xai"):
        self.provider = provider
        self.client = self._create_client()
        self.model = self._get_model()
    
    def _create_client(self) -> AsyncOpenAI:
        if self.provider == "xai":
            return AsyncOpenAI(
                api_key=os.environ.get("XAI_API_KEY"),
                base_url="https://api.x.ai/v1"
            )
        return AsyncOpenAI()
    
    def _get_model(self) -> str:
        if self.provider == "xai":
            return "grok-3"
        return "gpt-4o"
    
    async def convert(
        self,
        data: Any,
        output_type: OutputType = "auto",
        title: str = "",
        description: str = ""
    ) -> VisualOutput:
        """Convert data to visual output using LLM."""
        
        # For simple cases, use direct conversion
        if output_type == "table" or (output_type == "auto" and isinstance(data, (dict, list))):
            md = generate_markdown_table(data, title=title)
            return VisualOutput(output_type="table", markdown=md, raw_data=data)
        
        if output_type == "bar_chart" and isinstance(data, dict):
            # Try ASCII first
            ascii_chart = generate_ascii_bar_chart(data, title=title)
            mermaid = generate_mermaid_bar(data, title=title)
            return VisualOutput(
                output_type="bar_chart", 
                markdown=f"{ascii_chart}\n\n{mermaid}",
                mermaid=mermaid,
                raw_data=data
            )
        
        if output_type == "pie_chart" and isinstance(data, dict):
            mermaid = generate_mermaid_pie(data, title=title)
            return VisualOutput(
                output_type="pie_chart",
                markdown=mermaid,
                mermaid=mermaid,
                raw_data=data
            )
        
        if output_type == "line_chart" and isinstance(data, list):
            ascii_chart = generate_ascii_line_chart(data, title=title)
            return VisualOutput(
                output_type="line_chart",
                markdown=ascii_chart,
                raw_data=data
            )
        
        # Use LLM for complex/auto cases
        prompt = f"""Transform this data into a visual representation.

## Data:
```json
{json.dumps(data, indent=2, default=str)[:5000]}
```

## Requested Output: {output_type}
## Title: {title or 'Auto-generate'}
## Description: {description or 'None provided'}

Generate the appropriate markdown output with:
1. A clear title
2. The visualization (table, chart, or diagram)
3. Key insights or summary if relevant

Use markdown tables, ASCII charts, or Mermaid diagrams as appropriate.
Format numbers with commas and appropriate decimals.
"""
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=2000
        )
        
        content = response.choices[0].message.content
        
        # Detect output type from response
        detected_type = output_type
        if "```mermaid" in content:
            # Extract mermaid code
            mermaid_match = re.search(r'```mermaid\n([\s\S]*?)\n```', content)
            mermaid_code = mermaid_match.group(0) if mermaid_match else None
            return VisualOutput(
                output_type="diagram",
                markdown=content,
                mermaid=mermaid_code,
                raw_data=data
            )
        
        return VisualOutput(
            output_type=detected_type,
            markdown=content,
            raw_data=data
        )
    
    def table(self, data: Any, title: str = "") -> str:
        """Sync method: Convert data to markdown table."""
        return generate_markdown_table(data, title=title)
    
    def bar_chart(self, data: Dict[str, float], title: str = "") -> str:
        """Sync method: Convert data to bar chart."""
        return generate_ascii_bar_chart(data, title=title)
    
    def line_chart(self, data: List[float], title: str = "") -> str:
        """Sync method: Convert data to line chart."""
        return generate_ascii_line_chart(data, title=title)
    
    def pie_chart(self, data: Dict[str, float], title: str = "") -> str:
        """Sync method: Convert data to Mermaid pie chart."""
        return generate_mermaid_pie(data, title=title)
    
    def sparkline(self, data: List[float]) -> str:
        """Sync method: Generate sparkline."""
        return generate_sparkline(data)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def data_to_table(data: Any, title: str = "") -> str:
    """Convert data to markdown table."""
    return generate_markdown_table(data, title=title)


def data_to_bar_chart(data: Dict[str, float], title: str = "") -> str:
    """Convert data to ASCII bar chart."""
    return generate_ascii_bar_chart(data, title=title)


def data_to_line_chart(data: List[float], title: str = "") -> str:
    """Convert data to ASCII line chart."""
    return generate_ascii_line_chart(data, title=title)


def data_to_pie_chart(data: Dict[str, float], title: str = "") -> str:
    """Convert data to Mermaid pie chart."""
    return generate_mermaid_pie(data, title=title)


def data_to_sparkline(data: List[float]) -> str:
    """Convert data to sparkline."""
    return generate_sparkline(data)


async def data_to_visual(
    data: Any,
    output_type: OutputType = "auto",
    title: str = "",
    provider: str = "xai"
) -> str:
    """LLM-powered data to visual conversion."""
    converter = DataToVisualConverter(provider=provider)
    result = await converter.convert(data, output_type=output_type, title=title)
    return result.markdown


# ============================================================================
# CLI DEMO
# ============================================================================

async def demo():
    """Demo the data to visual converter."""
    
    print("=" * 60)
    print("DATA TO VISUAL CONVERTER DEMO")
    print("=" * 60)
    
    # Sample data
    revenue_data = {
        "Q1 2025": 94.8,
        "Q2 2025": 85.8,
        "Q3 2025": 89.5,
        "Q4 2025": 124.3
    }
    
    metrics_data = {
        "Revenue": "$394.4B",
        "Net Income": "$101.2B",
        "EPS": "$6.91",
        "Gross Margin": "46.9%",
        "ROE": "151.9%"
    }
    
    eps_trend = [1.61, 1.42, 1.73, 2.65]
    
    segment_share = {
        "iPhone": 52,
        "Services": 22,
        "Mac": 8,
        "iPad": 7,
        "Wearables": 11
    }
    
    # Demo each type
    print("\n### 1. MARKDOWN TABLE ###\n")
    print(data_to_table(metrics_data, title="Key Metrics"))
    
    print("\n### 2. BAR CHART ###\n")
    print(data_to_bar_chart(revenue_data, title="Quarterly Revenue ($B)"))
    
    print("\n### 3. LINE CHART ###\n")
    print(data_to_line_chart(eps_trend, title="EPS Trend"))
    
    print("\n### 4. PIE CHART (Mermaid) ###\n")
    print(data_to_pie_chart(segment_share, title="Revenue by Segment"))
    
    print("\n### 5. SPARKLINE ###\n")
    print(f"EPS Trend: {data_to_sparkline(eps_trend)} (Q1→Q4)")
    
    print("\n### 6. LLM-POWERED AUTO CONVERSION ###\n")
    complex_data = {
        "company": "Apple Inc.",
        "financials": {
            "2024": {"revenue": 383.3, "net_income": 97.0},
            "2025": {"revenue": 394.4, "net_income": 101.2}
        },
        "growth": {"revenue": 2.9, "net_income": 4.3}
    }
    
    result = await data_to_visual(
        complex_data, 
        output_type="auto",
        title="Apple Financial Summary"
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(demo())
