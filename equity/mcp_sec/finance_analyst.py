"""
Finance Analyst Agent

A ReAct-style agent that works like a human equity research analyst:
  1. Creates workspace folder for each analysis
  2. Gathers raw data using tools (saves to folder)
  3. Processes data → charts, tables, analysis paragraphs
  4. Builds valuation models
  5. Generates final report from processed artifacts

The agent THINKS, ACTS, OBSERVES in a loop until the report is complete.
"""

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from openai import AsyncOpenAI

# For chart generation
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ============================================================================
# AGENT TYPES
# ============================================================================

@dataclass
class ToolResult:
    """Result from a tool execution."""
    tool_name: str
    success: bool
    output: Any
    saved_to: Optional[str] = None
    error: Optional[str] = None


@dataclass
class AgentState:
    """Current state of the agent."""
    ticker: str
    workspace: Path
    
    # Data tracking
    raw_data: Dict[str, Any] = field(default_factory=dict)
    processed_data: Dict[str, Any] = field(default_factory=dict)
    charts: List[str] = field(default_factory=list)
    tables: List[str] = field(default_factory=list)
    analysis_paragraphs: List[str] = field(default_factory=list)
    
    # Model outputs
    valuation: Dict[str, Any] = field(default_factory=dict)
    rating: Dict[str, Any] = field(default_factory=dict)
    
    # Execution tracking
    steps: List[Dict[str, Any]] = field(default_factory=list)
    current_step: int = 0
    
    # Final outputs
    report_path: Optional[str] = None


# ============================================================================
# TOOL DEFINITIONS
# ============================================================================

AVAILABLE_TOOLS = """
## Data Gathering Tools
- `gather_market_data(ticker)` - Current price, 52W range, volume, market cap
- `gather_analyst_research(ticker)` - Ratings, price targets, broker coverage
- `gather_earnings_history(ticker)` - EPS actuals vs estimates, beat rate
- `gather_financial_statements(ticker)` - Income, Balance Sheet, Cash Flow
- `gather_peer_comparison(ticker)` - Compare with industry peers
- `gather_technicals(ticker)` - SMA, RSI, MACD, support/resistance
- `gather_news_sentiment(ticker)` - Recent headlines, sentiment score
- `gather_esg_scores(ticker)` - Environmental, Social, Governance
- `gather_guidance(ticker)` - Management outlook from 8-K
- `gather_insider_activity(ticker)` - Insider and institutional trading

## Processing Tools  
- `create_chart(data, chart_type, title, filename)` - Generate PNG chart
- `create_table(data, title, filename)` - Generate MD table file
- `write_analysis(topic, data, filename)` - Write analysis paragraph

## Modeling Tools
- `build_dcf_model(data)` - Discounted Cash Flow valuation
- `build_comparable_model(data)` - Comparable company analysis
- `calculate_fair_value(models)` - Blend models for target price

## Output Tools
- `generate_rating(all_data)` - Generate BUY/HOLD/SELL rating
- `compile_report(state)` - Compile final MD report from artifacts
"""


# ============================================================================
# CHART GENERATOR
# ============================================================================

class ChartGenerator:
    """Generates PNG charts from data."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.charts_dir = output_dir / "charts"
        self.charts_dir.mkdir(exist_ok=True)
    
    def bar_chart(
        self,
        data: Dict[str, float],
        title: str,
        filename: str,
        xlabel: str = "",
        ylabel: str = "",
        color: str = "#2E86AB"
    ) -> str:
        """Create horizontal bar chart."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        labels = list(data.keys())
        values = list(data.values())
        
        bars = ax.barh(labels, values, color=color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                   f'{val:,.1f}', va='center', fontsize=10)
        
        plt.tight_layout()
        path = self.charts_dir / f"{filename}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(path)
    
    def line_chart(
        self,
        data: Dict[str, List[float]],
        labels: List[str],
        title: str,
        filename: str,
        xlabel: str = "",
        ylabel: str = ""
    ) -> str:
        """Create line chart with multiple series."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
        
        for i, (name, values) in enumerate(data.items()):
            ax.plot(labels[:len(values)], values, marker='o', 
                   label=name, color=colors[i % len(colors)], linewidth=2)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = self.charts_dir / f"{filename}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(path)
    
    def pie_chart(
        self,
        data: Dict[str, float],
        title: str,
        filename: str
    ) -> str:
        """Create pie chart."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        labels = list(data.keys())
        values = list(data.values())
        colors = plt.cm.Set3(range(len(labels)))
        
        wedges, texts, autotexts = ax.pie(
            values, labels=labels, autopct='%1.1f%%',
            colors=colors, startangle=90
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        path = self.charts_dir / f"{filename}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(path)
    
    def waterfall_chart(
        self,
        data: Dict[str, float],
        title: str,
        filename: str
    ) -> str:
        """Create waterfall chart for valuation bridge."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        labels = list(data.keys())
        values = list(data.values())
        
        cumulative = 0
        for i, (label, val) in enumerate(zip(labels, values)):
            color = '#2E86AB' if val >= 0 else '#C73E1D'
            if i == len(labels) - 1:  # Last bar (total)
                ax.bar(i, cumulative + val, color='#3B1F2B')
            else:
                ax.bar(i, val, bottom=cumulative, color=color)
                cumulative += val
        
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='black', linewidth=0.5)
        
        plt.tight_layout()
        path = self.charts_dir / f"{filename}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(path)


# ============================================================================
# FINANCE ANALYST AGENT
# ============================================================================

AGENT_SYSTEM_PROMPT = """You are a senior equity research analyst at a top investment bank.

Your job is to analyze a company and produce a professional equity research report.

## Your Workflow:
1. GATHER DATA - Use tools to collect all relevant data
2. PROCESS DATA - Create charts, tables, analysis paragraphs
3. BUILD MODELS - DCF, comparable analysis, valuation
4. GENERATE RATING - Determine BUY/HOLD/SELL with conviction
5. COMPILE REPORT - Assemble final report from artifacts

## Available Tools:
{tools}

## Response Format:
Always respond with a JSON object:
```json
{{
    "thought": "Your reasoning about what to do next",
    "action": "tool_name",
    "action_input": {{"param1": "value1"}},
    "is_final": false
}}
```

When you have completed all analysis and are ready to compile the report:
```json
{{
    "thought": "Analysis complete, ready to compile report",
    "action": "compile_report",
    "action_input": {{}},
    "is_final": true
}}
```

Be thorough - a good equity research report requires comprehensive analysis.
Save intermediate work to files for transparency.
"""


class FinanceAnalystAgent:
    """ReAct-style agent for equity research."""
    
    def __init__(self, provider: str = "xai"):
        self.provider = provider
        self.client = self._create_client()
        self.model = self._get_model()
        self.state: Optional[AgentState] = None
        self.chart_generator: Optional[ChartGenerator] = None
        self.max_steps = 25
    
    def _create_client(self) -> AsyncOpenAI:
        if self.provider == "xai":
            return AsyncOpenAI(
                api_key=os.environ.get("XAI_API_KEY"),
                base_url="https://api.x.ai/v1"
            )
        return AsyncOpenAI()
    
    def _get_model(self) -> str:
        return "grok-3" if self.provider == "xai" else "gpt-4o"
    
    def _create_workspace(self, ticker: str) -> Path:
        """Create workspace folder for analysis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        workspace = Path(f"analysis/{ticker}_{timestamp}")
        workspace.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (workspace / "raw_data").mkdir(exist_ok=True)
        (workspace / "charts").mkdir(exist_ok=True)
        (workspace / "tables").mkdir(exist_ok=True)
        (workspace / "analysis").mkdir(exist_ok=True)
        (workspace / "models").mkdir(exist_ok=True)
        
        return workspace
    
    async def _execute_tool(self, action: str, params: Dict[str, Any]) -> ToolResult:
        """Execute a tool and return result."""
        
        try:
            # Data gathering tools
            if action == "gather_market_data":
        from . import get_market_data
                result = await get_market_data(params.get("ticker", self.state.ticker))
                path = self.state.workspace / "raw_data" / "market_data.md"
                path.write_text(result)
                self.state.raw_data["market_data"] = result
                return ToolResult(action, True, result, str(path))
            
            elif action == "gather_analyst_research":
        from . import get_analyst_research
                result = get_analyst_research(params.get("ticker", self.state.ticker))
                path = self.state.workspace / "raw_data" / "analyst_research.md"
                path.write_text(result)
                self.state.raw_data["analyst_research"] = result
                return ToolResult(action, True, result, str(path))
            
            elif action == "gather_earnings_history":
        from . import get_transcript_list
                result = await get_transcript_list(params.get("ticker", self.state.ticker))
                path = self.state.workspace / "raw_data" / "earnings_history.md"
                path.write_text(result)
                self.state.raw_data["earnings_history"] = result
                return ToolResult(action, True, result, str(path))
            
            elif action == "gather_peer_comparison":
                from . import get_peer_comparison
                result = await get_peer_comparison(params.get("ticker", self.state.ticker))
                path = self.state.workspace / "raw_data" / "peer_comparison.md"
                path.write_text(result)
                self.state.raw_data["peer_comparison"] = result
                return ToolResult(action, True, result, str(path))
            
            elif action == "gather_technicals":
                from . import get_technicals
                result = await get_technicals(params.get("ticker", self.state.ticker))
                path = self.state.workspace / "raw_data" / "technicals.md"
                path.write_text(result)
                self.state.raw_data["technicals"] = result
                return ToolResult(action, True, result, str(path))
            
            elif action == "gather_news_sentiment":
                from . import get_news_sentiment
                result = await get_news_sentiment(params.get("ticker", self.state.ticker))
                path = self.state.workspace / "raw_data" / "news_sentiment.md"
                path.write_text(result)
                self.state.raw_data["news_sentiment"] = result
                return ToolResult(action, True, result, str(path))
            
            elif action == "gather_esg_scores":
        from . import get_esg_score
                result = await get_esg_score(params.get("ticker", self.state.ticker))
                path = self.state.workspace / "raw_data" / "esg_scores.md"
                path.write_text(result)
                self.state.raw_data["esg_scores"] = result
                return ToolResult(action, True, result, str(path))
            
            elif action == "gather_guidance":
        from . import get_company_guidance
                result = await get_company_guidance(params.get("ticker", self.state.ticker))
                path = self.state.workspace / "raw_data" / "guidance.md"
                path.write_text(result)
                self.state.raw_data["guidance"] = result
                return ToolResult(action, True, result, str(path))
            
            elif action == "gather_insider_activity":
                from . import get_insider_trading, get_institutional_holdings
                insider = await get_insider_trading(params.get("ticker", self.state.ticker))
                inst = await get_institutional_holdings(params.get("ticker", self.state.ticker))
                result = f"{insider}\n\n{inst}"
                path = self.state.workspace / "raw_data" / "insider_activity.md"
                path.write_text(result)
                self.state.raw_data["insider_activity"] = result
                return ToolResult(action, True, result, str(path))
            
            # Chart creation
            elif action == "create_chart":
                chart_type = params.get("chart_type", "bar")
                data = params.get("data", {})
                title = params.get("title", "Chart")
                filename = params.get("filename", "chart")
                
                if chart_type == "bar":
                    path = self.chart_generator.bar_chart(data, title, filename)
                elif chart_type == "line":
                    labels = params.get("labels", list(range(len(list(data.values())[0]))))
                    path = self.chart_generator.line_chart(data, labels, title, filename)
                elif chart_type == "pie":
                    path = self.chart_generator.pie_chart(data, title, filename)
                elif chart_type == "waterfall":
                    path = self.chart_generator.waterfall_chart(data, title, filename)
                else:
                    return ToolResult(action, False, None, error=f"Unknown chart type: {chart_type}")
                
                self.state.charts.append(path)
                return ToolResult(action, True, f"Chart saved: {path}", str(path))
            
            # Table creation
            elif action == "create_table":
                from . import data_to_table
                data = params.get("data", {})
                title = params.get("title", "Table")
                filename = params.get("filename", "table")
                
                md = data_to_table(data, title=title)
                path = self.state.workspace / "tables" / f"{filename}.md"
                path.write_text(md)
                self.state.tables.append(str(path))
                return ToolResult(action, True, md, str(path))
        
            # Analysis writing
            elif action == "write_analysis":
                topic = params.get("topic", "Analysis")
                data = params.get("data", "")
                filename = params.get("filename", "analysis")
                
                # Use LLM to write analysis
                analysis = await self._write_analysis_paragraph(topic, data)
                path = self.state.workspace / "analysis" / f"{filename}.md"
                path.write_text(analysis)
                self.state.analysis_paragraphs.append(analysis)
                return ToolResult(action, True, analysis, str(path))
            
            # DCF Model
            elif action == "build_dcf_model":
                model = await self._build_dcf_model(params.get("data", self.state.raw_data))
                path = self.state.workspace / "models" / "dcf_model.json"
                path.write_text(json.dumps(model, indent=2))
                self.state.valuation["dcf"] = model
                return ToolResult(action, True, model, str(path))
            
            # Comparable Model
            elif action == "build_comparable_model":
                model = await self._build_comp_model(params.get("data", self.state.raw_data))
                path = self.state.workspace / "models" / "comparable_model.json"
                path.write_text(json.dumps(model, indent=2))
                self.state.valuation["comparable"] = model
                return ToolResult(action, True, model, str(path))
            
            # Fair Value
            elif action == "calculate_fair_value":
                fair_value = self._calculate_fair_value()
                self.state.valuation["fair_value"] = fair_value
                return ToolResult(action, True, fair_value)
            
            # Rating
            elif action == "generate_rating":
                rating = await self._generate_rating()
                self.state.rating = rating
                path = self.state.workspace / "rating.json"
                path.write_text(json.dumps(rating, indent=2))
                return ToolResult(action, True, rating, str(path))
            
            # Compile Report
            elif action == "compile_report":
                report = await self._compile_report()
                path = self.state.workspace / "REPORT.md"
                path.write_text(report)
                self.state.report_path = str(path)
                return ToolResult(action, True, f"Report saved to {path}", str(path))
            
            else:
                return ToolResult(action, False, None, error=f"Unknown tool: {action}")
                
        except Exception as e:
            return ToolResult(action, False, None, error=str(e))
    
    async def _write_analysis_paragraph(self, topic: str, data: Any) -> str:
        """Use LLM to write analysis paragraph."""
        prompt = f"""Write a professional equity research analysis paragraph about: {topic}

Data to analyze:
{json.dumps(data, indent=2, default=str)[:5000] if isinstance(data, (dict, list)) else str(data)[:5000]}

Write 2-3 paragraphs of professional analysis. Be specific with numbers.
Include investment implications.
"""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a senior equity research analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )
        return response.choices[0].message.content
    
    async def _build_dcf_model(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build DCF valuation model."""
        prompt = f"""Build a DCF (Discounted Cash Flow) model based on this data:

{json.dumps(data, indent=2, default=str)[:8000]}

Return a JSON object with:
{{
    "assumptions": {{
        "revenue_growth_rate": <float>,
        "terminal_growth_rate": <float>,
        "wacc": <float>,
        "projection_years": <int>
    }},
    "projections": [
        {{"year": 1, "revenue": <float>, "fcf": <float>}},
        ...
    ],
    "terminal_value": <float>,
    "enterprise_value": <float>,
    "equity_value": <float>,
    "fair_value_per_share": <float>
}}
"""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a financial modeling expert. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1500
        )
        
        content = response.choices[0].message.content
        try:
            match = re.search(r'\{[\s\S]*\}', content)
            if match:
                return json.loads(match.group())
            except:
            pass
        return {"fair_value_per_share": 0, "error": "Could not build model"}
    
    async def _build_comp_model(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build comparable company model."""
        prompt = f"""Build a comparable company analysis based on this data:

{json.dumps(data, indent=2, default=str)[:8000]}

Return a JSON object with:
{{
    "peer_multiples": {{
        "pe_ratio": {{"min": <float>, "median": <float>, "max": <float>}},
        "ev_ebitda": {{"min": <float>, "median": <float>, "max": <float>}}
    }},
    "target_metrics": {{
        "eps": <float>,
        "ebitda": <float>
    }},
    "implied_values": {{
        "pe_low": <float>,
        "pe_median": <float>,
        "pe_high": <float>,
        "ev_ebitda_median": <float>
    }},
    "fair_value_per_share": <float>
}}
"""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a financial modeling expert. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1000
        )
        
        content = response.choices[0].message.content
        try:
            match = re.search(r'\{[\s\S]*\}', content)
            if match:
                return json.loads(match.group())
        except:
            pass
        return {"fair_value_per_share": 0, "error": "Could not build model"}
    
    def _calculate_fair_value(self) -> Dict[str, Any]:
        """Blend valuation models."""
        dcf_val = self.state.valuation.get("dcf", {}).get("fair_value_per_share", 0)
        comp_val = self.state.valuation.get("comparable", {}).get("fair_value_per_share", 0)
        
        # Weighted average (60% DCF, 40% comps)
        values = []
        if dcf_val > 0:
            values.append(("dcf", dcf_val, 0.6))
        if comp_val > 0:
            values.append(("comparable", comp_val, 0.4))
        
        if not values:
            return {"blended_fair_value": 0, "method": "none"}
        
        # Normalize weights
        total_weight = sum(v[2] for v in values)
        blended = sum(v[1] * (v[2] / total_weight) for v in values)
        
        return {
            "dcf_value": dcf_val,
            "comparable_value": comp_val,
            "blended_fair_value": blended,
            "method": "weighted_average"
        }
    
    async def _generate_rating(self) -> Dict[str, Any]:
        """Generate investment rating."""
        prompt = f"""Based on this analysis, generate an investment rating:

## Valuation:
{json.dumps(self.state.valuation, indent=2, default=str)}

## Raw Data Summary:
{json.dumps({k: v[:500] if isinstance(v, str) else v for k, v in self.state.raw_data.items()}, indent=2, default=str)[:5000]}

Return JSON:
{{
    "rating": "BUY" | "HOLD" | "SELL",
    "conviction": "HIGH" | "MEDIUM" | "LOW",
    "price_target": <float>,
    "current_price": <float>,
    "upside_pct": <float>,
    "thesis": "<2-3 sentence investment thesis>",
    "catalysts": ["<catalyst 1>", "<catalyst 2>", "<catalyst 3>"],
    "risks": ["<risk 1>", "<risk 2>", "<risk 3>"]
}}
"""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a senior equity research analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=800
        )
        
        content = response.choices[0].message.content
        try:
            match = re.search(r'\{[\s\S]*\}', content)
            if match:
                return json.loads(match.group())
        except:
            pass
        return {"rating": "HOLD", "conviction": "LOW"}
    
    async def _compile_report(self) -> str:
        """Compile final report from all artifacts."""
        sections = []
        
        # Header
        rating = self.state.rating
        sections.append(f"""# {self.state.ticker} Equity Research Report
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*

---

## Investment Rating

| Metric | Value |
|:-------|:------|
| **Rating** | **{rating.get('rating', 'N/A')}** |
| **Conviction** | {rating.get('conviction', 'N/A')} |
| **Price Target** | ${rating.get('price_target', 0):,.2f} |
| **Current Price** | ${rating.get('current_price', 0):,.2f} |
| **Upside** | {rating.get('upside_pct', 0):+.1f}% |

### Investment Thesis
{rating.get('thesis', '')}

### Catalysts
""")
        for c in rating.get('catalysts', []):
            sections.append(f"- {c}")
        
        sections.append("\n### Risks")
        for r in rating.get('risks', []):
            sections.append(f"- {r}")
        
        sections.append("\n---\n")
        
        # Include charts
        if self.state.charts:
            sections.append("## Charts\n")
            for chart_path in self.state.charts:
                rel_path = Path(chart_path).relative_to(self.state.workspace)
                sections.append(f"![Chart]({rel_path})\n")
        
        # Include analysis paragraphs
        if self.state.analysis_paragraphs:
            sections.append("## Analysis\n")
            for para in self.state.analysis_paragraphs:
                sections.append(para)
            sections.append("\n---\n")
        
        # Include raw data sections
        sections.append("## Data Appendix\n")
        for name, data in self.state.raw_data.items():
            if isinstance(data, str):
                sections.append(f"### {name.replace('_', ' ').title()}\n")
                # Truncate long sections
                sections.append(data[:2000] if len(data) > 2000 else data)
                sections.append("\n---\n")
        
        # Valuation
        sections.append("## Valuation Summary\n")
        sections.append(f"```json\n{json.dumps(self.state.valuation, indent=2, default=str)}\n```\n")
        
        return "\n".join(sections)
    
    async def _get_next_action(self) -> Dict[str, Any]:
        """Ask LLM for next action."""
        # Build context
        context = f"""
## Current State
- Ticker: {self.state.ticker}
- Workspace: {self.state.workspace}
- Step: {self.state.current_step} / {self.max_steps}

## Data Gathered:
{list(self.state.raw_data.keys())}

## Charts Created:
{len(self.state.charts)} charts

## Tables Created:
{len(self.state.tables)} tables

## Analysis Written:
{len(self.state.analysis_paragraphs)} paragraphs

## Valuation Models:
{list(self.state.valuation.keys())}

## Rating Generated:
{"Yes" if self.state.rating else "No"}

## Recent Steps:
{json.dumps(self.state.steps[-5:], indent=2, default=str) if self.state.steps else "None yet"}
"""
        
        prompt = f"""What should I do next to complete the equity research report for {self.state.ticker}?

{context}

Decide on ONE action to take. Remember:
1. First gather all necessary data
2. Then process into charts and analysis
3. Build valuation models
4. Generate rating
5. Finally compile the report

Respond with JSON only.
        """
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": AGENT_SYSTEM_PROMPT.format(tools=AVAILABLE_TOOLS)},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        content = response.choices[0].message.content
        
        # Parse JSON
        try:
            match = re.search(r'\{[\s\S]*\}', content)
            if match:
                return json.loads(match.group())
        except:
            pass
        
        return {
            "thought": "Failed to parse action",
            "action": "compile_report",
            "action_input": {},
            "is_final": True
        }
    
    async def analyze(self, ticker: str) -> str:
        """Run full analysis for a ticker."""
        ticker = ticker.upper()
        
        print(f"\n{'='*60}")
        print(f"🤖 FINANCE ANALYST AGENT - Analyzing {ticker}")
        print(f"{'='*60}\n")
        
        # Create workspace
        workspace = self._create_workspace(ticker)
        print(f"📁 Created workspace: {workspace}\n")
        
        # Initialize state
        self.state = AgentState(ticker=ticker, workspace=workspace)
        self.chart_generator = ChartGenerator(workspace)
        
        # Agent loop
        while self.state.current_step < self.max_steps:
            self.state.current_step += 1
            print(f"--- Step {self.state.current_step} ---")
            
            # Get next action from LLM
            action_response = await self._get_next_action()
            
            thought = action_response.get("thought", "")
            action = action_response.get("action", "")
            action_input = action_response.get("action_input", {})
            is_final = action_response.get("is_final", False)
            
            print(f"💭 Thought: {thought}")
            print(f"🔧 Action: {action}")
            
            # Execute action
            result = await self._execute_tool(action, action_input)
            
            if result.success:
                print(f"✅ Success: {result.saved_to or 'Done'}")
            else:
                print(f"❌ Error: {result.error}")
            
            # Track step
            self.state.steps.append({
                "step": self.state.current_step,
                "thought": thought,
                "action": action,
                "success": result.success,
                "saved_to": result.saved_to
            })
            
            # Check if done
            if is_final or action == "compile_report":
                print(f"\n{'='*60}")
                print(f"✅ ANALYSIS COMPLETE")
                print(f"📄 Report: {self.state.report_path}")
                print(f"📁 Workspace: {self.state.workspace}")
                print(f"{'='*60}\n")
                break
            
            print()
        
        # Save execution log
        log_path = self.state.workspace / "execution_log.json"
        log_path.write_text(json.dumps(self.state.steps, indent=2))
        
        return self.state.report_path or str(workspace)


# ============================================================================
# CLI
# ============================================================================

async def main():
    import sys
    
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    
    agent = FinanceAnalystAgent(provider="xai")
    report_path = await agent.analyze(ticker)
    
    print(f"\n📄 Report saved to: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
