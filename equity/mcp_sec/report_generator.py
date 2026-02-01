"""
Equity Research Report Generator

End-to-end pipeline:
  1. Gather all data from 13 MCP tools
  2. Use LLM as Finance Analyst to synthesize
  3. Generate structured MD report with proper layout

Report Structure:
  - Page 1: Executive Summary (Rating, Target, Thesis)
  - Page 2-3: Investment Thesis + Key Metrics
  - Page 4-5: Financial Analysis (Tables, Charts)
  - Page 6: Valuation Analysis
  - Page 7: Risk Factors
  - Page 8: Appendix (Data Sources)
"""

import asyncio
import json
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from openai import AsyncOpenAI


# ============================================================================
# REPORT STRUCTURE DEFINITION
# ============================================================================

@dataclass
class ReportSection:
    """A section of the equity research report."""
    name: str
    required: bool  # Must-have vs Good-to-have
    data_sources: List[str]
    content: str = ""


REPORT_STRUCTURE = {
    # PAGE 1: EXECUTIVE SUMMARY (Front Page)
    "executive_summary": ReportSection(
        name="Executive Summary",
        required=True,
        data_sources=["analyst_research", "market_data", "earnings_transcripts"],
    ),
    
    # MUST-HAVE SECTIONS
    "investment_rating": ReportSection(
        name="Investment Rating",
        required=True,
        data_sources=["analyst_research", "technicals"],
    ),
    "price_target": ReportSection(
        name="Price Target & Valuation",
        required=True,
        data_sources=["analyst_research", "analyzer", "market_data"],
    ),
    "investment_thesis": ReportSection(
        name="Investment Thesis",
        required=True,
        data_sources=["guidance_transcripts", "news_sentiment", "earnings_transcripts"],
    ),
    "financial_summary": ReportSection(
        name="Financial Summary",
        required=True,
        data_sources=["server", "analyst_research"],  # Income, Balance, Cash Flow
    ),
    "earnings_analysis": ReportSection(
        name="Earnings Analysis",
        required=True,
        data_sources=["earnings_transcripts", "earnings_peers"],
    ),
    "valuation_analysis": ReportSection(
        name="Valuation Analysis",
        required=True,
        data_sources=["analyzer", "earnings_peers"],
    ),
    "risk_factors": ReportSection(
        name="Risk Factors",
        required=True,
        data_sources=["news_sentiment", "guidance_transcripts", "esg_scores"],
    ),
    
    # GOOD-TO-HAVE SECTIONS
    "technical_analysis": ReportSection(
        name="Technical Analysis",
        required=False,
        data_sources=["technicals"],
    ),
    "peer_comparison": ReportSection(
        name="Peer Comparison",
        required=False,
        data_sources=["earnings_peers"],
    ),
    "insider_activity": ReportSection(
        name="Insider & Institutional Activity",
        required=False,
        data_sources=["insider_holdings"],
    ),
    "esg_analysis": ReportSection(
        name="ESG Analysis",
        required=False,
        data_sources=["esg_scores"],
    ),
    "management_guidance": ReportSection(
        name="Management Guidance",
        required=False,
        data_sources=["guidance_transcripts"],
    ),
    "news_sentiment": ReportSection(
        name="News & Sentiment",
        required=False,
        data_sources=["news_sentiment"],
    ),
}


# ============================================================================
# DATA GATHERING
# ============================================================================

@dataclass
class GatheredData:
    """All gathered data for report generation."""
    ticker: str
    company_name: str
    gathered_at: str
    
    # Raw data from each tool
    financial_statements: Dict[str, Any] = field(default_factory=dict)
    market_data: Dict[str, Any] = field(default_factory=dict)
    analyst_research: Dict[str, Any] = field(default_factory=dict)
    earnings_history: Dict[str, Any] = field(default_factory=dict)
    earnings_calendar: Dict[str, Any] = field(default_factory=dict)
    peer_comparison: Dict[str, Any] = field(default_factory=dict)
    technicals: Dict[str, Any] = field(default_factory=dict)
    news_sentiment: Dict[str, Any] = field(default_factory=dict)
    esg_scores: Dict[str, Any] = field(default_factory=dict)
    guidance: Dict[str, Any] = field(default_factory=dict)
    insider_trading: Dict[str, Any] = field(default_factory=dict)
    institutional_holdings: Dict[str, Any] = field(default_factory=dict)
    full_analysis: Dict[str, Any] = field(default_factory=dict)


class DataGatherer:
    """Gathers all data for equity research report."""
    
    def __init__(self):
        pass
    
    async def gather_all(self, ticker: str) -> GatheredData:
        """Gather all data from all tools."""
        ticker = ticker.upper()
        
        print(f"📊 Gathering data for {ticker}...")
        
        # Import all tools
        from . import (
            get_market_data,
            get_analyst_research,
            get_transcript_list,
            get_earnings,
            get_peer_comparison,
            get_technicals,
            get_news_sentiment,
            get_esg_score,
            get_company_guidance,
            get_insider_trading,
            get_institutional_holdings,
            analyze_ticker,
        )
        from .server import SECClient
        
        data = GatheredData(
            ticker=ticker,
            company_name="",
            gathered_at=datetime.now().isoformat()
        )
        
        # Gather in parallel where possible
        print("  → Fetching market data & analyst research...")
        try:
            data.market_data = {"md": await get_market_data(ticker)}
        except Exception as e:
            print(f"    ⚠️ Market data: {e}")
        
        try:
            data.analyst_research = {"md": get_analyst_research(ticker)}
        except Exception as e:
            print(f"    ⚠️ Analyst research: {e}")
        
        print("  → Fetching earnings data...")
        try:
            data.earnings_history = {"md": await get_transcript_list(ticker)}
        except Exception as e:
            print(f"    ⚠️ Earnings history: {e}")
        
        try:
            data.earnings_calendar = {"md": await get_earnings(ticker)}
        except Exception as e:
            print(f"    ⚠️ Earnings calendar: {e}")
        
        print("  → Fetching peer comparison...")
        try:
            data.peer_comparison = {"md": await get_peer_comparison(ticker)}
        except Exception as e:
            print(f"    ⚠️ Peer comparison: {e}")
        
        print("  → Fetching technical analysis...")
        try:
            data.technicals = {"md": await get_technicals(ticker)}
        except Exception as e:
            print(f"    ⚠️ Technicals: {e}")
        
        print("  → Fetching news & sentiment...")
        try:
            data.news_sentiment = {"md": await get_news_sentiment(ticker)}
        except Exception as e:
            print(f"    ⚠️ News sentiment: {e}")
        
        print("  → Fetching ESG scores...")
        try:
            data.esg_scores = {"md": await get_esg_score(ticker)}
        except Exception as e:
            print(f"    ⚠️ ESG: {e}")
        
        print("  → Fetching management guidance...")
        try:
            data.guidance = {"md": await get_company_guidance(ticker)}
        except Exception as e:
            print(f"    ⚠️ Guidance: {e}")
        
        print("  → Fetching insider activity...")
        try:
            data.insider_trading = {"md": await get_insider_trading(ticker)}
        except Exception as e:
            print(f"    ⚠️ Insider trading: {e}")
        
        try:
            data.institutional_holdings = {"md": await get_institutional_holdings(ticker)}
        except Exception as e:
            print(f"    ⚠️ Institutional holdings: {e}")
        
        print("  → Fetching financial statements...")
        try:
            sec = SECClient()
            cik = await sec.get_cik(ticker)
            if cik:
                statements = await sec.get_financial_statements(cik)
                data.financial_statements = {
                    "income": statements.get("income_statement", {}),
                    "balance": statements.get("balance_sheet", {}),
                    "cashflow": statements.get("cash_flow", {}),
                }
        except Exception as e:
            print(f"    ⚠️ Financial statements: {e}")
        
        print("  → Running full analysis...")
        try:
            data.full_analysis = {"md": await analyze_ticker(ticker)}
        except Exception as e:
            print(f"    ⚠️ Full analysis: {e}")
        
        print(f"✅ Data gathering complete for {ticker}")
        
        return data


# ============================================================================
# LLM FINANCE ANALYST AGENT
# ============================================================================

ANALYST_SYSTEM_PROMPT = """You are a senior equity research analyst at a top investment bank. 
Your role is to analyze company data and write professional equity research reports.

Your writing style:
- Professional, concise, data-driven
- Clear investment thesis with supporting evidence
- Objective analysis of risks and opportunities
- Actionable recommendations with price targets

When generating report sections:
1. Use the provided data to support your analysis
2. Include specific numbers, percentages, and comparisons
3. Highlight key insights that matter to investors
4. Be balanced - acknowledge both positives and negatives
5. Use markdown formatting for tables and emphasis
"""


class FinanceAnalystAgent:
    """LLM-powered finance analyst for report generation."""
    
    def __init__(self, provider: str = "xai"):
        self.provider = provider
        self.client = self._create_client()
        self.model = self._get_model()
    
    def _create_client(self) -> AsyncOpenAI:
        """Create LLM client."""
        if self.provider == "xai":
            api_key = os.environ.get("XAI_API_KEY")
            return AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.x.ai/v1"
            )
        elif self.provider == "openai":
            return AsyncOpenAI()
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _get_model(self) -> str:
        """Get model name."""
        if self.provider == "xai":
            return "grok-3"
        return "gpt-4o"
    
    async def generate_section(
        self,
        section_name: str,
        section_prompt: str,
        data: Dict[str, Any]
    ) -> str:
        """Generate a single report section."""
        
        user_prompt = f"""Generate the "{section_name}" section of an equity research report.

## Available Data:
{json.dumps(data, indent=2, default=str)[:15000]}

## Section Requirements:
{section_prompt}

Write the section in markdown format. Include tables where appropriate.
Be specific with numbers and percentages from the data.
"""
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    
    async def generate_rating(self, data: GatheredData) -> Dict[str, Any]:
        """Generate investment rating and price target."""
        
        prompt = f"""Based on the following data, provide an investment rating.

## Analyst Consensus:
{data.analyst_research.get('md', 'N/A')[:3000]}

## Technical Signals:
{data.technicals.get('md', 'N/A')[:2000]}

## Earnings History:
{data.earnings_history.get('md', 'N/A')[:2000]}

## Full Analysis:
{data.full_analysis.get('md', 'N/A')[:3000]}

Respond in JSON format:
{{
    "rating": "BUY" | "HOLD" | "SELL",
    "conviction": "HIGH" | "MEDIUM" | "LOW",
    "price_target": <float>,
    "current_price": <float>,
    "upside_pct": <float>,
    "key_thesis": "<1-2 sentence thesis>",
    "bull_case": "<1 sentence>",
    "bear_case": "<1 sentence>",
    "catalysts": ["<catalyst 1>", "<catalyst 2>", "<catalyst 3>"],
    "risks": ["<risk 1>", "<risk 2>", "<risk 3>"]
}}
"""
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1000
        )
        
        content = response.choices[0].message.content
        
        # Parse JSON from response
        try:
            # Find JSON in response
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return {
            "rating": "HOLD",
            "conviction": "MEDIUM",
            "price_target": 0,
            "current_price": 0,
            "upside_pct": 0,
            "key_thesis": "Analysis pending",
            "catalysts": [],
            "risks": []
        }
    
    async def generate_executive_summary(
        self,
        data: GatheredData,
        rating: Dict[str, Any]
    ) -> str:
        """Generate the executive summary front page."""
        
        prompt = f"""Generate an executive summary for {data.ticker} equity research report.

## Investment Rating:
{json.dumps(rating, indent=2)}

## Key Data:
- Analyst Research: {data.analyst_research.get('md', 'N/A')[:2000]}
- Market Data: {data.market_data.get('md', 'N/A')[:1000]}
- Earnings: {data.earnings_history.get('md', 'N/A')[:1000]}

Create a professional executive summary with:
1. Company header (ticker, name, sector)
2. Rating box (Rating, Target, Upside)
3. Key investment thesis (2-3 sentences)
4. Key metrics table
5. Catalysts and risks bullets

Use markdown formatting. This is the FRONT PAGE of the report.
"""
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        
        return response.choices[0].message.content


# ============================================================================
# REPORT GENERATOR
# ============================================================================

class EquityReportGenerator:
    """Generates complete equity research reports."""
    
    def __init__(self, provider: str = "xai"):
        self.gatherer = DataGatherer()
        self.analyst = FinanceAnalystAgent(provider=provider)
    
    async def generate_report(
        self,
        ticker: str,
        include_optional: bool = True,
        output_path: Optional[str] = None
    ) -> str:
        """Generate complete equity research report."""
        
        ticker = ticker.upper()
        
        # Step 1: Gather all data
        data = await self.gatherer.gather_all(ticker)
        
        # Step 2: Generate investment rating
        print("\n🤖 Analyzing data with AI...")
        rating = await self.analyst.generate_rating(data)
        print(f"  → Rating: {rating.get('rating')} | Target: ${rating.get('price_target', 0):.2f}")
        
        # Step 3: Generate executive summary
        print("  → Generating executive summary...")
        exec_summary = await self.analyst.generate_executive_summary(data, rating)
        
        # Step 4: Build full report
        print("  → Building full report...")
        
        sections = []
        
        # Front page
        sections.append(exec_summary)
        sections.append("\n---\n")
        
        # Must-have sections
        must_have_sections = [
            ("Investment Thesis", data.guidance.get("md", "") + "\n" + data.news_sentiment.get("md", "")),
            ("Earnings Analysis", data.earnings_history.get("md", "") + "\n" + data.earnings_calendar.get("md", "")),
            ("Financial Summary", data.full_analysis.get("md", "")),
            ("Valuation Analysis", data.analyst_research.get("md", "")),
            ("Risk Factors", "See ESG and News sections below"),
        ]
        
        for section_name, content in must_have_sections:
            sections.append(f"\n## {section_name}\n")
            if content:
                sections.append(content[:3000])  # Truncate if too long
            sections.append("\n---\n")
        
        # Good-to-have sections
        if include_optional:
            optional_sections = [
                ("Technical Analysis", data.technicals.get("md", "")),
                ("Peer Comparison", data.peer_comparison.get("md", "")),
                ("Insider & Institutional Activity", 
                 data.insider_trading.get("md", "") + "\n" + data.institutional_holdings.get("md", "")),
                ("ESG Analysis", data.esg_scores.get("md", "")),
                ("News & Sentiment", data.news_sentiment.get("md", "")),
            ]
            
            for section_name, content in optional_sections:
                if content:
                    sections.append(f"\n## {section_name}\n")
                    sections.append(content[:3000])
                    sections.append("\n---\n")
        
        # Appendix
        sections.append("\n## Appendix\n")
        sections.append(f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        sections.append(f"**Data Sources:** SEC EDGAR, NASDAQ, Yahoo Finance\n")
        sections.append(f"**Analyst:** AI-Powered Equity Research Agent\n")
        
        # Combine all sections
        full_report = "\n".join(sections)
        
        # Save if path provided
        if output_path:
            with open(output_path, "w") as f:
                f.write(full_report)
            print(f"\n✅ Report saved to: {output_path}")
        
        return full_report


# ============================================================================
# CLI
# ============================================================================

async def main():
    import sys
    
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    output = sys.argv[2] if len(sys.argv) > 2 else f"examples/{ticker}_report.md"
    
    generator = EquityReportGenerator(provider="xai")
    report = await generator.generate_report(ticker, output_path=output)
    
    print(f"\n📄 Report Preview (first 2000 chars):\n")
    print(report[:2000])


if __name__ == "__main__":
    asyncio.run(main())
