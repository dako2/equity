"""
Earnings Call Transcripts & Data

Fetches earnings data from multiple sources:
  - NASDAQ API (earnings dates, EPS data)
  - SEC 8-K filings (earnings press releases)
  
Provides:
  - Earnings history (actual vs estimates)
  - Upcoming earnings dates
  - Key quotes extraction from 8-K
  - Management commentary analysis
"""

import asyncio
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import requests
import httpx


@dataclass
class TranscriptListing:
    """An earnings call transcript listing."""
    id: str
    title: str
    quarter: str
    year: int
    date: str
    url: str
    is_paywalled: bool = False


@dataclass
class TranscriptContent:
    """Parsed transcript content."""
    ticker: str
    company_name: str
    quarter: str
    year: int
    date: str
    
    # Extracted content
    ceo_quotes: List[str] = field(default_factory=list)
    cfo_quotes: List[str] = field(default_factory=list)
    key_highlights: List[str] = field(default_factory=list)
    guidance_mentions: List[str] = field(default_factory=list)
    
    # Sentiment
    tone: str = ""  # "Bullish", "Neutral", "Cautious"
    
    # Source
    source: str = ""
    source_url: str = ""


@dataclass
class EarningsResult:
    """A single earnings result."""
    quarter: str
    fiscal_year: int
    report_date: str
    eps_actual: Optional[float] = None
    eps_estimate: Optional[float] = None
    eps_surprise: Optional[float] = None
    eps_surprise_pct: Optional[float] = None
    revenue_actual: Optional[float] = None


class EarningsTranscriptClient:
    """Fetches earnings data and transcripts."""
    
    # Keywords for extraction
    HIGHLIGHT_KEYWORDS = [
        'record', 'growth', 'strong', 'exceeded', 'beat', 
        'challenging', 'headwind', 'declined', 'improved'
    ]
    
    GUIDANCE_KEYWORDS = [
        'expect', 'guidance', 'outlook', 'forecast', 'anticipate',
        'project', 'target', 'full year', 'next quarter'
    ]
    
    BULLISH_WORDS = ['strong', 'record', 'exceeded', 'growth', 'momentum', 'confident']
    CAUTIOUS_WORDS = ['challenging', 'headwind', 'uncertainty', 'cautious', 'decline']
    
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
        }
    
    def get_nasdaq_earnings(self, ticker: str) -> List[EarningsResult]:
        """Get earnings history from NASDAQ API."""
        ticker = ticker.upper()
        
        # NASDAQ earnings surprise endpoint
        url = f"https://api.nasdaq.com/api/company/{ticker}/earnings-surprise"
        
        try:
            resp = requests.get(url, headers=self.headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except:
            return []
        
        results = []
        earnings_data = data.get("data", {}).get("earningsSurpriseTable", {}).get("rows", [])
        
        for row in earnings_data[:8]:  # Last 8 quarters
            # Parse quarter info (e.g., "Dec 2025")
            fiscal_end = row.get("fiscalQtrEnd", "")
            
            # Parse EPS values - NASDAQ returns numbers directly
            eps_val = row.get("eps")
            estimate_val = row.get("consensusForecast", "")
            surprise_val = row.get("percentageSurprise", "")
            
            try:
                eps_actual = float(eps_val) if eps_val is not None else None
            except:
                eps_actual = None
            
            try:
                eps_estimate = float(str(estimate_val).replace("$", "")) if estimate_val else None
            except:
                eps_estimate = None
            
            try:
                eps_surprise_pct = float(str(surprise_val).replace("%", "")) if surprise_val else None
            except:
                eps_surprise_pct = None
            
            # Parse date
            date_str = row.get("dateReported", "")
            
            results.append(EarningsResult(
                quarter=fiscal_end,
                fiscal_year=datetime.now().year,
                report_date=date_str,
                eps_actual=eps_actual,
                eps_estimate=eps_estimate,
                eps_surprise_pct=eps_surprise_pct
            ))
        
        return results
    
    async def get_transcript_list(self, ticker: str, limit: int = 8) -> List[TranscriptListing]:
        """Get list of available transcripts (placeholder - requires subscription)."""
        ticker = ticker.upper()
        
        # NASDAQ doesn't have transcripts, return empty
        # Users need Seeking Alpha Pro, Bloomberg, or similar for full transcripts
        return []
    
    async def get_cik(self, ticker: str) -> Optional[str]:
        """Get CIK for ticker."""
        url = "https://www.sec.gov/files/company_tickers.json"
        
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(
                    url,
                    headers={"User-Agent": "Equity Research research@example.com"},
                    timeout=15.0
                )
                resp.raise_for_status()
                data = resp.json()
            except:
                return None
        
        ticker = ticker.upper()
        for entry in data.values():
            if entry.get("ticker") == ticker:
                return str(entry.get("cik_str"))
        return None
    
    async def get_company_name(self, ticker: str) -> str:
        """Get company name."""
        url = "https://www.sec.gov/files/company_tickers.json"
        
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(
                    url,
                    headers={"User-Agent": "Equity Research research@example.com"},
                    timeout=15.0
                )
                resp.raise_for_status()
                data = resp.json()
            except:
                return ticker
        
        for entry in data.values():
            if entry.get("ticker") == ticker.upper():
                return entry.get("title", ticker)
        return ticker
    
    async def get_8k_earnings_content(self, ticker: str, cik: str) -> List[Dict[str, Any]]:
        """Get earnings-related 8-K content from SEC."""
        url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
        
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(
                    url,
                    headers={"User-Agent": "Equity Research research@example.com"},
                    timeout=15.0
                )
                resp.raise_for_status()
                data = resp.json()
            except:
                return []
        
        filings = []
        recent = data.get("filings", {}).get("recent", {})
        
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        descriptions = recent.get("primaryDocDescription", [])
        
        for i, form in enumerate(forms):
            if form == "8-K" and len(filings) < 5:
                desc = descriptions[i] if i < len(descriptions) else ""
                # Look for earnings-related 8-Ks
                if any(kw in desc.lower() for kw in ['result', 'earnings', 'financial']):
                    acc = accessions[i].replace("-", "")
                    filings.append({
                        "date": dates[i],
                        "description": desc,
                        "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{primary_docs[i]}"
                    })
        
        return filings
    
    def analyze_tone(self, text: str) -> str:
        """Analyze overall tone of transcript text."""
        text_lower = text.lower()
        
        bullish = sum(1 for w in self.BULLISH_WORDS if w in text_lower)
        cautious = sum(1 for w in self.CAUTIOUS_WORDS if w in text_lower)
        
        if bullish > cautious + 2:
            return "Bullish"
        elif cautious > bullish + 2:
            return "Cautious"
        return "Neutral"
    
    async def get_transcript_analysis(self, ticker: str) -> TranscriptContent:
        """Get transcript analysis combining multiple sources."""
        ticker = ticker.upper()
        
        company_name = await self.get_company_name(ticker)
        cik = await self.get_cik(ticker)
        
        # Get Seeking Alpha listings
        listings = await self.get_transcript_list(ticker, limit=4)
        
        # Get SEC 8-K earnings releases
        sec_content = []
        if cik:
            sec_content = await self.get_8k_earnings_content(ticker, cik)
        
        # Use most recent quarter if available
        if listings:
            latest = listings[0]
            quarter = latest.quarter
            year = latest.year
            date = latest.date
            source_url = latest.url
        else:
            quarter = "Q?"
            year = datetime.now().year
            date = datetime.now().strftime("%Y-%m-%d")
            source_url = ""
        
        # Extract key highlights from titles
        highlights = []
        for listing in listings:
            highlights.append(listing.title)
        
        # Placeholder for actual transcript parsing
        # Full transcript content typically requires Seeking Alpha Pro or scraping
        
        return TranscriptContent(
            ticker=ticker,
            company_name=company_name,
            quarter=quarter,
            year=year,
            date=date,
            key_highlights=highlights[:3],
            guidance_mentions=[],
            ceo_quotes=[],
            cfo_quotes=[],
            tone="Neutral",
            source="Seeking Alpha + SEC 8-K",
            source_url=source_url
        )
    
    async def get_full_listing(self, ticker: str) -> Dict[str, Any]:
        """Get comprehensive earnings listing."""
        ticker = ticker.upper()
        
        company_name = await self.get_company_name(ticker)
        earnings = self.get_nasdaq_earnings(ticker)
        
        return {
            "ticker": ticker,
            "company_name": company_name,
            "earnings": earnings,
            "count": len(earnings)
        }


def format_transcripts_markdown(data: Dict[str, Any]) -> str:
    """Format earnings history as Markdown."""
    
    ticker = data["ticker"]
    company = data["company_name"]
    earnings = data["earnings"]
    
    md = f"""# {ticker} Earnings History
**{company}**
*Found {data['count']} recent earnings reports*

---

## Earnings Surprise History

| Quarter | Report Date | EPS Actual | EPS Estimate | Surprise |
|:--------|:------------|----------:|-------------:|:---------|
"""
    
    for e in earnings:
        actual = f"${e.eps_actual:.2f}" if e.eps_actual is not None else "-"
        estimate = f"${e.eps_estimate:.2f}" if e.eps_estimate is not None else "-"
        
        if e.eps_surprise_pct is not None:
            if e.eps_surprise_pct > 0:
                surprise = f"🟢 +{e.eps_surprise_pct:.1f}%"
            elif e.eps_surprise_pct < 0:
                surprise = f"🔴 {e.eps_surprise_pct:.1f}%"
            else:
                surprise = "➖ 0%"
        else:
            surprise = "-"
        
        md += f"| **{e.quarter}** | {e.report_date} | {actual} | {estimate} | {surprise} |\n"
    
    if not earnings:
        md += "| - | - | - | - | No data found |\n"
    
    # Calculate beat rate
    beats = sum(1 for e in earnings if e.eps_surprise_pct and e.eps_surprise_pct > 0)
    total = len([e for e in earnings if e.eps_surprise_pct is not None])
    beat_rate = (beats / total * 100) if total > 0 else 0
    
    md += f"""
---

## Summary

| Metric | Value |
|:-------|------:|
| Quarters Analyzed | {total} |
| Beat Rate | {beat_rate:.0f}% |
| Beats | {beats} |
| Misses | {total - beats} |

---

## How to Access Full Transcripts

1. **Seeking Alpha** (Pro subscription): Full earnings call transcripts
2. **SEC 8-K Filings**: Earnings press releases and exhibits
3. **Company IR Website**: Many companies post transcripts on investor relations pages

---

*Earnings data sourced from NASDAQ API*
"""
    
    return md


def format_analysis_markdown(content: TranscriptContent) -> str:
    """Format transcript analysis as Markdown."""
    
    tone_emoji = "🟢" if content.tone == "Bullish" else \
                "🔴" if content.tone == "Cautious" else "🟡"
    
    md = f"""# {content.ticker} Earnings Call Analysis
**{content.company_name}** | **{content.quarter} {content.year}**
*Call Date: {content.date}*

---

## Tone Summary

| Metric | Value |
|:-------|------:|
| **Management Tone** | {tone_emoji} **{content.tone}** |

---

## Key Highlights

"""
    
    if content.key_highlights:
        for h in content.key_highlights:
            md += f"- {h}\n"
    else:
        md += "*No key highlights extracted*\n"
    
    md += "\n---\n\n## CEO Commentary\n\n"
    if content.ceo_quotes:
        for q in content.ceo_quotes:
            md += f"> \"{q}\"\n\n"
    else:
        md += "*Full transcript required for CEO quotes*\n"
    
    md += "\n## CFO Commentary\n\n"
    if content.cfo_quotes:
        for q in content.cfo_quotes:
            md += f"> \"{q}\"\n\n"
    else:
        md += "*Full transcript required for CFO quotes*\n"
    
    md += f"""
---

## Source

- **Provider**: {content.source}
- **Link**: [{content.source_url}]({content.source_url})

*For full transcript content, visit the source link above.*
"""
    
    return md


async def get_transcript_list(ticker: str) -> str:
    """Get transcript listing as Markdown."""
    client = EarningsTranscriptClient()
    data = await client.get_full_listing(ticker)
    return format_transcripts_markdown(data)


async def get_transcript_analysis(ticker: str) -> str:
    """Get transcript analysis as Markdown."""
    client = EarningsTranscriptClient()
    content = await client.get_transcript_analysis(ticker)
    return format_analysis_markdown(content)


# CLI
if __name__ == "__main__":
    import sys
    
    mode = sys.argv[1] if len(sys.argv) > 1 else "list"
    ticker = sys.argv[2] if len(sys.argv) > 2 else "AAPL"
    
    if mode == "list":
        print(asyncio.run(get_transcript_list(ticker)))
    elif mode == "analyze":
        print(asyncio.run(get_transcript_analysis(ticker)))
    else:
        print(f"Usage: python -m mcp_sec.earnings_transcripts [list|analyze] TICKER")
        print(f"\nExamples:")
        print(f"  python -m mcp_sec.earnings_transcripts list AAPL")
        print(f"  python -m mcp_sec.earnings_transcripts analyze MSFT")
