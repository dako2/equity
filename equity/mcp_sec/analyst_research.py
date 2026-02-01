"""
Analyst Research (NASDAQ)

Aggregates analyst data from NASDAQ including:
  - Analyst ratings (Buy/Hold/Sell consensus)
  - Price targets
  - Financial ratios
  - Broker coverage list (including Zacks)
"""

import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import requests


@dataclass
class AnalystRating:
    """Analyst rating summary."""
    broker_name: str
    rating: str = ""


@dataclass
class FinancialRatios:
    """Key financial ratios."""
    period: str
    # Liquidity
    current_ratio: float = 0
    quick_ratio: float = 0
    cash_ratio: float = 0
    # Profitability
    gross_margin: float = 0
    operating_margin: float = 0
    profit_margin: float = 0
    roe: float = 0


@dataclass
class AnalystResearch:
    """Complete analyst research data."""
    ticker: str
    company_name: str
    last_updated: str
    
    # Ratings
    consensus_rating: str = ""  # Buy, Hold, Sell
    num_analysts: int = 0
    brokers: List[str] = field(default_factory=list)
    
    # Price Target
    current_price: float = 0
    one_year_target: float = 0
    upside_pct: float = 0
    
    # Stock Info
    sector: str = ""
    industry: str = ""
    market_cap: float = 0
    
    # Dividend
    dividend_yield: float = 0
    annual_dividend: float = 0
    ex_dividend_date: str = ""
    
    # Ratios (latest)
    ratios: Optional[FinancialRatios] = None
    
    # Sources
    data_source: str = "NASDAQ"


class AnalystResearchClient:
    """Fetches analyst research from NASDAQ."""
    
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
        }
        self.base_url = "https://api.nasdaq.com/api"
    
    def get_analyst_ratings(self, ticker: str) -> Dict[str, Any]:
        """Get analyst ratings."""
        url = f"{self.base_url}/analyst/{ticker}/ratings"
        
        try:
            resp = requests.get(url, headers=self.headers, timeout=15)
            resp.raise_for_status()
            return resp.json().get("data", {})
        except:
            return {}
    
    def get_stock_summary(self, ticker: str) -> Dict[str, Any]:
        """Get stock summary including price target."""
        url = f"{self.base_url}/quote/{ticker}/summary?assetclass=stocks"
        
        try:
            resp = requests.get(url, headers=self.headers, timeout=15)
            resp.raise_for_status()
            return resp.json().get("data", {})
        except:
            return {}
    
    def get_quote_info(self, ticker: str) -> Dict[str, Any]:
        """Get current quote info."""
        url = f"{self.base_url}/quote/{ticker}/info?assetclass=stocks"
        
        try:
            resp = requests.get(url, headers=self.headers, timeout=15)
            resp.raise_for_status()
            return resp.json().get("data", {})
        except:
            return {}
    
    def get_financial_ratios(self, ticker: str) -> Optional[FinancialRatios]:
        """Get financial ratios."""
        url = f"{self.base_url}/company/{ticker}/financials?frequency=1"
        
        try:
            resp = requests.get(url, headers=self.headers, timeout=15)
            resp.raise_for_status()
            data = resp.json().get("data", {})
        except:
            return None
        
        ratios_table = data.get("financialRatiosTable", {})
        headers = ratios_table.get("headers", {})
        rows = ratios_table.get("rows", [])
        
        # Get latest period
        period = headers.get("value2", "")
        
        # Parse ratios
        ratios_dict = {}
        for row in rows:
            label = row.get("value1", "").lower().replace(" ", "_")
            value = row.get("value2", "")
            if value and "%" in str(value):
                try:
                    ratios_dict[label] = float(value.replace("%", ""))
                except:
                    pass
        
        return FinancialRatios(
            period=period,
            current_ratio=ratios_dict.get("current_ratio", 0),
            quick_ratio=ratios_dict.get("quick_ratio", 0),
            cash_ratio=ratios_dict.get("cash_ratio", 0),
            gross_margin=ratios_dict.get("gross_margin", 0),
            operating_margin=ratios_dict.get("operating_margin", 0),
            profit_margin=ratios_dict.get("profit_margin", 0),
            roe=ratios_dict.get("after_tax_roe", 0)
        )
    
    def get_research(self, ticker: str) -> AnalystResearch:
        """Get complete analyst research."""
        ticker = ticker.upper()
        
        # Get all data
        ratings = self.get_analyst_ratings(ticker)
        summary = self.get_stock_summary(ticker)
        quote = self.get_quote_info(ticker)
        ratios = self.get_financial_ratios(ticker)
        
        # Parse ratings
        consensus = ratings.get("meanRatingType", "")
        brokers = ratings.get("brokerNames", [])
        
        # Parse summary
        summary_data = summary.get("summaryData", {})
        
        def parse_value(key: str, remove_chars: str = "$,%") -> str:
            val = summary_data.get(key, {}).get("value", "")
            for c in remove_chars:
                val = str(val).replace(c, "")
            return val
        
        try:
            target = float(parse_value("OneYrTarget"))
        except:
            target = 0
        
        try:
            market_cap = float(parse_value("MarketCap", ",").replace(",", ""))
        except:
            market_cap = 0
        
        try:
            dividend_yield = float(parse_value("Yield"))
        except:
            dividend_yield = 0
        
        try:
            annual_div = float(parse_value("AnnualizedDividend"))
        except:
            annual_div = 0
        
        # Parse quote
        primary = quote.get("primaryData", {})
        try:
            current_price = float(primary.get("lastSalePrice", "0").replace("$", "").replace(",", ""))
        except:
            current_price = 0
        
        company_name = quote.get("companyName", ticker)
        
        # Calculate upside
        if current_price > 0 and target > 0:
            upside = ((target - current_price) / current_price) * 100
        else:
            upside = 0
        
        # Parse summary string to get analyst count
        summary_str = ratings.get("ratingsSummary", "")
        import re
        match = re.search(r'(\d+)\s+analysts', summary_str)
        num_analysts = int(match.group(1)) if match else len(brokers)
        
        return AnalystResearch(
            ticker=ticker,
            company_name=company_name,
            last_updated=datetime.now().isoformat(),
            consensus_rating=consensus,
            num_analysts=num_analysts,
            brokers=brokers,
            current_price=current_price,
            one_year_target=target,
            upside_pct=upside,
            sector=summary_data.get("Sector", {}).get("value", ""),
            industry=summary_data.get("Industry", {}).get("value", ""),
            market_cap=market_cap,
            dividend_yield=dividend_yield,
            annual_dividend=annual_div,
            ex_dividend_date=summary_data.get("ExDividendDate", {}).get("value", ""),
            ratios=ratios,
            data_source="NASDAQ"
        )


def format_research_markdown(r: AnalystResearch) -> str:
    """Format analyst research as Markdown."""
    
    # Rating emoji
    if "buy" in r.consensus_rating.lower() or "strong" in r.consensus_rating.lower():
        rating_emoji = "🟢"
    elif "sell" in r.consensus_rating.lower():
        rating_emoji = "🔴"
    else:
        rating_emoji = "🟡"
    
    # Upside emoji
    if r.upside_pct > 15:
        upside_emoji = "📈"
    elif r.upside_pct < -10:
        upside_emoji = "📉"
    else:
        upside_emoji = "➖"
    
    md = f"""# {r.ticker} Analyst Research
**{r.company_name}**
*Updated: {r.last_updated}*

---

## Analyst Consensus

| Metric | Value |
|:-------|------:|
| **Consensus Rating** | {rating_emoji} **{r.consensus_rating}** |
| Analysts Covering | {r.num_analysts} |

---

## Price Target

| Metric | Value |
|:-------|------:|
| Current Price | ${r.current_price:,.2f} |
| **1-Year Target** | **${r.one_year_target:,.2f}** |
| Implied Upside | {upside_emoji} {r.upside_pct:+.1f}% |

---

## Company Profile

| Metric | Value |
|:-------|------:|
| Sector | {r.sector} |
| Industry | {r.industry} |
| Market Cap | ${r.market_cap/1e9:,.1f}B |

---

## Dividend

| Metric | Value |
|:-------|------:|
| Annual Dividend | ${r.annual_dividend:.2f} |
| Dividend Yield | {r.dividend_yield:.2f}% |
| Ex-Dividend Date | {r.ex_dividend_date} |

"""
    
    if r.ratios:
        md += f"""---

## Financial Ratios (Period: {r.ratios.period})

### Profitability

| Ratio | Value |
|:------|------:|
| Gross Margin | {r.ratios.gross_margin:.1f}% |
| Operating Margin | {r.ratios.operating_margin:.1f}% |
| Net Profit Margin | {r.ratios.profit_margin:.1f}% |
| Return on Equity | {r.ratios.roe:.1f}% |

### Liquidity

| Ratio | Value |
|:------|------:|
| Current Ratio | {r.ratios.current_ratio:.1f}% |
| Quick Ratio | {r.ratios.quick_ratio:.1f}% |
| Cash Ratio | {r.ratios.cash_ratio:.1f}% |

"""
    
    md += """---

## Broker Coverage

"""
    # Split brokers into columns
    if r.brokers:
        # Show first 10 brokers
        for broker in r.brokers[:15]:
            zacks_tag = " ⭐" if "ZACKS" in broker.upper() else ""
            md += f"- {broker}{zacks_tag}\n"
        if len(r.brokers) > 15:
            md += f"- *...and {len(r.brokers) - 15} more*\n"
    else:
        md += "*No broker coverage data*\n"
    
    md += f"""
---

*Data sourced from {r.data_source}*
"""
    
    return md


def get_analyst_research(ticker: str) -> str:
    """Get analyst research as Markdown."""
    client = AnalystResearchClient()
    research = client.get_research(ticker)
    return format_research_markdown(research)


# CLI
if __name__ == "__main__":
    import sys
    
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(get_analyst_research(ticker))
