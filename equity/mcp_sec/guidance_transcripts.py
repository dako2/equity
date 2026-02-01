"""
Guidance & Earnings Transcripts

Parses 8-K filings and earnings-related content:
  - Management guidance extraction
  - Key metrics and outlook
  - Recent earnings highlights
"""

import asyncio
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import httpx


@dataclass
class Guidance:
    """Company guidance data."""
    ticker: str
    company_name: str
    last_updated: str
    fiscal_year: str = ""
    fiscal_quarter: str = ""
    
    # Revenue
    revenue_low: float = 0
    revenue_high: float = 0
    revenue_growth: str = ""
    
    # EPS
    eps_low: float = 0
    eps_high: float = 0
    eps_growth: str = ""
    
    # Margins
    gross_margin_guidance: str = ""
    operating_margin_guidance: str = ""
    
    # Other metrics
    capex_guidance: str = ""
    free_cash_flow: str = ""
    
    # Qualitative
    outlook: str = ""  # Positive, Neutral, Cautious
    key_initiatives: List[str] = field(default_factory=list)
    headwinds: List[str] = field(default_factory=list)
    tailwinds: List[str] = field(default_factory=list)
    
    # Source
    source_date: str = ""
    source_filing: str = ""


@dataclass  
class EarningsHighlight:
    """Key earnings highlights."""
    ticker: str
    period: str
    date: str
    
    # Results vs Expectations
    revenue_actual: float = 0
    revenue_estimate: float = 0
    revenue_beat: bool = False
    
    eps_actual: float = 0
    eps_estimate: float = 0
    eps_beat: bool = False
    
    # Key metrics
    gross_margin: float = 0
    operating_margin: float = 0
    
    # YoY changes
    revenue_yoy: float = 0
    eps_yoy: float = 0
    
    # Management commentary
    key_points: List[str] = field(default_factory=list)


class GuidanceParser:
    """Parses company guidance from SEC filings."""
    
    # Guidance keywords
    REVENUE_PATTERNS = [
        r'revenue.*?(\$[\d,.]+)\s*(?:to|[-–])\s*(\$[\d,.]+)\s*(billion|million)?',
        r'expects?\s+revenue.*?(\$[\d,.]+)\s*(billion|million)?',
        r'guidance.*?revenue.*?(\d+(?:\.\d+)?)\s*%'
    ]
    
    EPS_PATTERNS = [
        r'eps.*?(\$[\d.]+)\s*(?:to|[-–])\s*(\$[\d.]+)',
        r'earnings\s+per\s+share.*?(\$[\d.]+)\s*(?:to|[-–])\s*(\$[\d.]+)',
        r'diluted\s+eps.*?(\$[\d.]+)'
    ]
    
    OUTLOOK_POSITIVE = ['strong', 'robust', 'confident', 'accelerating', 'optimistic', 'exceeding']
    OUTLOOK_CAUTIOUS = ['cautious', 'uncertain', 'challenging', 'headwind', 'softening', 'slowing']
    
    def __init__(self):
        self.headers = {
            "User-Agent": "Equity Research Tool research@example.com"
        }
    
    async def get_cik(self, ticker: str) -> Optional[str]:
        """Get CIK for ticker."""
        url = "https://www.sec.gov/files/company_tickers.json"
        
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(url, headers=self.headers, timeout=15.0)
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
                resp = await client.get(url, headers=self.headers, timeout=15.0)
                resp.raise_for_status()
                data = resp.json()
            except:
                return ticker
        
        for entry in data.values():
            if entry.get("ticker") == ticker.upper():
                return entry.get("title", ticker)
        return ticker
    
    async def get_recent_8k(self, cik: str) -> List[Dict[str, Any]]:
        """Get recent 8-K filings."""
        url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
        
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(url, headers=self.headers, timeout=15.0)
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
            if form == "8-K" and len(filings) < 10:
                acc = accessions[i].replace("-", "")
                filings.append({
                    "date": dates[i],
                    "accession": accessions[i],
                    "doc": primary_docs[i] if i < len(primary_docs) else "",
                    "desc": descriptions[i] if i < len(descriptions) else "",
                    "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{primary_docs[i]}"
                })
        
        return filings
    
    async def fetch_8k_content(self, url: str) -> str:
        """Fetch 8-K filing content."""
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(url, headers=self.headers, timeout=30.0)
                resp.raise_for_status()
                return resp.text
            except:
                return ""
    
    def extract_guidance(self, content: str, ticker: str, company_name: str, source_info: Dict) -> Guidance:
        """Extract guidance from filing content."""
        content_lower = content.lower()
        
        # Determine outlook
        positive_count = sum(1 for w in self.OUTLOOK_POSITIVE if w in content_lower)
        cautious_count = sum(1 for w in self.OUTLOOK_CAUTIOUS if w in content_lower)
        
        if positive_count > cautious_count + 2:
            outlook = "Positive"
        elif cautious_count > positive_count + 2:
            outlook = "Cautious"
        else:
            outlook = "Neutral"
        
        # Extract revenue guidance
        revenue_low, revenue_high = 0, 0
        for pattern in self.REVENUE_PATTERNS:
            match = re.search(pattern, content_lower)
            if match:
                groups = match.groups()
                try:
                    if len(groups) >= 2 and groups[1]:
                        val1 = float(re.sub(r'[$,]', '', groups[0]))
                        val2 = float(re.sub(r'[$,]', '', groups[1]))
                        revenue_low, revenue_high = min(val1, val2), max(val1, val2)
                        break
                except:
                    pass
        
        # Extract EPS guidance
        eps_low, eps_high = 0, 0
        for pattern in self.EPS_PATTERNS:
            match = re.search(pattern, content_lower)
            if match:
                groups = match.groups()
                try:
                    val1 = float(re.sub(r'[$,]', '', groups[0]))
                    if len(groups) >= 2 and groups[1]:
                        val2 = float(re.sub(r'[$,]', '', groups[1]))
                        eps_low, eps_high = min(val1, val2), max(val1, val2)
                    else:
                        eps_low = eps_high = val1
                    break
                except:
                    pass
        
        # Extract key initiatives
        initiatives = []
        initiative_patterns = [
            r'(launch\w*\s+\w+(?:\s+\w+)?)',
            r'(invest\w*\s+in\s+\w+(?:\s+\w+)?)',
            r'(expand\w*\s+\w+(?:\s+\w+)?)'
        ]
        for pattern in initiative_patterns:
            matches = re.findall(pattern, content_lower)
            initiatives.extend(matches[:2])
        
        # Extract headwinds/tailwinds
        headwinds = []
        tailwinds = []
        
        if 'inflation' in content_lower or 'cost pressure' in content_lower:
            headwinds.append("Cost pressures")
        if 'supply chain' in content_lower:
            headwinds.append("Supply chain challenges")
        if 'competition' in content_lower:
            headwinds.append("Competitive pressure")
        if 'demand' in content_lower and 'strong' in content_lower:
            tailwinds.append("Strong demand")
        if 'market share' in content_lower and 'gain' in content_lower:
            tailwinds.append("Market share gains")
        if 'innovation' in content_lower or 'new product' in content_lower:
            tailwinds.append("Product innovation")
        
        return Guidance(
            ticker=ticker,
            company_name=company_name,
            last_updated=datetime.now().isoformat(),
            fiscal_year=str(datetime.now().year),
            revenue_low=revenue_low,
            revenue_high=revenue_high,
            eps_low=eps_low,
            eps_high=eps_high,
            outlook=outlook,
            key_initiatives=initiatives[:3],
            headwinds=headwinds[:3],
            tailwinds=tailwinds[:3],
            source_date=source_info.get("date", ""),
            source_filing=source_info.get("url", "")
        )
    
    async def get_guidance(self, ticker: str) -> Guidance:
        """Get company guidance from recent 8-K filings."""
        ticker = ticker.upper()
        
        cik = await self.get_cik(ticker)
        company_name = await self.get_company_name(ticker)
        
        if not cik:
            return Guidance(
                ticker=ticker,
                company_name=company_name,
                last_updated=datetime.now().isoformat(),
                outlook="Unknown - No SEC data"
            )
        
        filings = await self.get_recent_8k(cik)
        
        # Look for earnings-related 8-K
        for filing in filings:
            desc = filing.get("desc", "").lower()
            if any(kw in desc for kw in ['results', 'earnings', 'financial']):
                content = await self.fetch_8k_content(filing["url"])
                if content:
                    return self.extract_guidance(content, ticker, company_name, filing)
        
        # Fallback: try first 8-K
        if filings:
            content = await self.fetch_8k_content(filings[0]["url"])
            if content:
                return self.extract_guidance(content, ticker, company_name, filings[0])
        
        return Guidance(
            ticker=ticker,
            company_name=company_name,
            last_updated=datetime.now().isoformat(),
            outlook="No recent guidance found"
        )


def format_guidance_markdown(g: Guidance) -> str:
    """Format guidance as Markdown."""
    
    outlook_emoji = "🟢" if g.outlook == "Positive" else \
                   "🔴" if g.outlook == "Cautious" else "🟡"
    
    md = f"""# {g.ticker} Management Guidance
**{g.company_name}** | *FY{g.fiscal_year}*
*Updated: {g.last_updated}*

---

## Outlook Summary

| Metric | Value |
|:-------|------:|
| **Management Outlook** | {outlook_emoji} **{g.outlook}** |

---

## Financial Guidance

### Revenue
"""
    
    if g.revenue_low and g.revenue_high:
        md += f"| Range | ${g.revenue_low:,.0f}B - ${g.revenue_high:,.0f}B |\n"
        md += f"|:------|------:|\n"
    else:
        md += "*No specific revenue guidance provided*\n"
    
    md += "\n### Earnings Per Share (EPS)\n"
    
    if g.eps_low and g.eps_high:
        md += f"| Range | ${g.eps_low:.2f} - ${g.eps_high:.2f} |\n"
        md += f"|:------|------:|\n"
    elif g.eps_low:
        md += f"| Estimate | ${g.eps_low:.2f} |\n"
        md += f"|:---------|------:|\n"
    else:
        md += "*No specific EPS guidance provided*\n"
    
    md += "\n---\n\n## Key Initiatives\n\n"
    
    if g.key_initiatives:
        for init in g.key_initiatives:
            md += f"- {init.title()}\n"
    else:
        md += "*No specific initiatives mentioned*\n"
    
    md += "\n---\n\n## Tailwinds & Headwinds\n\n"
    md += "### Tailwinds 📈\n"
    if g.tailwinds:
        for t in g.tailwinds:
            md += f"- ✅ {t}\n"
    else:
        md += "*None identified*\n"
    
    md += "\n### Headwinds 📉\n"
    if g.headwinds:
        for h in g.headwinds:
            md += f"- ⚠️ {h}\n"
    else:
        md += "*None identified*\n"
    
    md += f"""
---

## Source

| Filed | Link |
|:------|:-----|
| {g.source_date} | [View 8-K]({g.source_filing}) |

*Guidance extracted from SEC 8-K filings. Always verify with official company communications.*
"""
    
    return md


async def get_company_guidance(ticker: str) -> str:
    """Get company guidance as Markdown."""
    parser = GuidanceParser()
    guidance = await parser.get_guidance(ticker)
    return format_guidance_markdown(guidance)


# CLI
if __name__ == "__main__":
    import sys
    
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(asyncio.run(get_company_guidance(ticker)))
