"""
Market Data Provider

Fetches real-time stock prices, trading metrics, and market data
to complement SEC financial statements.

Data includes:
  - Current price, change, volume
  - 52-week high/low
  - Market cap, shares outstanding
  - Beta, dividend yield
  - Analyst ratings and price targets
"""

import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import httpx


@dataclass
class MarketData:
    """Real-time market data for a stock."""
    ticker: str
    company_name: str
    
    # Price
    current_price: float = 0
    previous_close: float = 0
    change: float = 0
    change_pct: float = 0
    
    # Trading
    open: float = 0
    day_high: float = 0
    day_low: float = 0
    volume: int = 0
    avg_volume: int = 0
    
    # 52-Week Range
    week_52_high: float = 0
    week_52_low: float = 0
    
    # Market Cap
    market_cap: float = 0
    shares_outstanding: float = 0
    float_shares: float = 0
    
    # Valuation (from market)
    pe_ratio: float = 0
    forward_pe: float = 0
    peg_ratio: float = 0
    price_to_book: float = 0
    price_to_sales: float = 0
    
    # Dividends
    dividend_rate: float = 0
    dividend_yield: float = 0
    ex_dividend_date: str = ""
    
    # Risk
    beta: float = 0
    
    # Analyst
    target_mean: float = 0
    target_high: float = 0
    target_low: float = 0
    recommendation: str = ""
    num_analysts: int = 0
    
    # Metadata
    currency: str = "USD"
    exchange: str = ""
    last_updated: str = ""


class MarketDataProvider:
    """Fetches real-time market data."""
    
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
        }
    
    async def get_quote(self, ticker: str) -> MarketData:
        """Get real-time quote and market data for a ticker."""
        ticker = ticker.upper()
        data = MarketData(ticker=ticker, company_name=ticker)
        
        # Try Yahoo Finance API
        try:
            quote = await self._fetch_yahoo_quote(ticker)
            if quote:
                data = self._parse_yahoo_quote(ticker, quote)
        except Exception as e:
            print(f"Warning: Yahoo Finance error for {ticker}: {e}")
        
        data.last_updated = datetime.now().isoformat()
        return data
    
    async def _fetch_yahoo_quote(self, ticker: str) -> Dict[str, Any]:
        """Fetch quote from Yahoo Finance using chart API (no auth required)."""
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        
        async with httpx.AsyncClient() as client:
            # Get 1-day data for current price
            resp = await client.get(
                url, 
                params={"interval": "1d", "range": "1d"},
                headers=self.headers,
                timeout=15.0
            )
            resp.raise_for_status()
            chart_data = resp.json()
            
            # Get 1-year data for 52-week high/low
            try:
                year_resp = await client.get(
                    url,
                    params={"interval": "1d", "range": "1y"},
                    headers=self.headers,
                    timeout=15.0
                )
                if year_resp.status_code == 200:
                    year_data = year_resp.json()
                    return {"chart": chart_data, "year": year_data}
            except:
                pass
            
            return {"chart": chart_data}
    
    def _parse_yahoo_quote(self, ticker: str, data: Dict[str, Any]) -> MarketData:
        """Parse Yahoo Finance response from chart APIs."""
        md = MarketData(ticker=ticker, company_name=ticker)
        
        # Parse daily chart data
        chart = data.get("chart", {})
        result = chart.get("chart", chart).get("result", [{}])[0]
        meta = result.get("meta", {})
        
        # Basic price from chart API
        md.current_price = meta.get("regularMarketPrice", 0)
        md.previous_close = meta.get("previousClose", meta.get("chartPreviousClose", 0))
        md.exchange = meta.get("exchangeName", "")
        md.currency = meta.get("currency", "USD")
        md.company_name = meta.get("shortName", meta.get("symbol", ticker))
        
        if md.current_price and md.previous_close:
            md.change = md.current_price - md.previous_close
            md.change_pct = md.change / md.previous_close if md.previous_close else 0
        
        # Day data from indicators
        indicators = result.get("indicators", {}).get("quote", [{}])[0]
        if indicators:
            high = [h for h in indicators.get("high", []) if h]
            low = [l for l in indicators.get("low", []) if l]
            vol = [v for v in indicators.get("volume", []) if v]
            opens = [o for o in indicators.get("open", []) if o]
            
            if high:
                md.day_high = max(high)
            if low:
                md.day_low = min(low)
            if vol:
                md.volume = sum(vol)
            if opens:
                md.open = opens[0]
        
        # Parse 1-year data for 52-week high/low
        year_data = data.get("year", {})
        year_result = year_data.get("chart", year_data).get("result", [{}])[0]
        year_indicators = year_result.get("indicators", {}).get("quote", [{}])[0]
        
        if year_indicators:
            year_high = [h for h in year_indicators.get("high", []) if h]
            year_low = [l for l in year_indicators.get("low", []) if l]
            year_vol = [v for v in year_indicators.get("volume", []) if v]
            
            if year_high:
                md.week_52_high = max(year_high)
            if year_low:
                md.week_52_low = min(year_low)
            if year_vol:
                md.avg_volume = int(sum(year_vol) / len(year_vol))
        
        # Estimate market cap from price and shares (if we have SEC data)
        # This will be filled in by the analyzer
        
        return md
    
    async def get_multiple_quotes(self, tickers: list) -> Dict[str, MarketData]:
        """Get quotes for multiple tickers."""
        tasks = [self.get_quote(t) for t in tickers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            t: r for t, r in zip(tickers, results) 
            if isinstance(r, MarketData)
        }


def format_market_data_markdown(data: MarketData) -> str:
    """Format market data as Markdown."""
    
    def fmt(val, fmt_type="num"):
        if val is None or val == 0:
            return "-"
        if fmt_type == "money":
            if abs(val) >= 1e12:
                return f"${val/1e12:.2f}T"
            elif abs(val) >= 1e9:
                return f"${val/1e9:.2f}B"
            elif abs(val) >= 1e6:
                return f"${val/1e6:.1f}M"
            else:
                return f"${val:,.2f}"
        elif fmt_type == "pct":
            return f"{val*100:.2f}%" if abs(val) < 1 else f"{val:.2f}%"
        elif fmt_type == "vol":
            if val >= 1e6:
                return f"{val/1e6:.1f}M"
            elif val >= 1e3:
                return f"{val/1e3:.1f}K"
            return f"{val:,.0f}"
        elif fmt_type == "price":
            return f"${val:,.2f}"
        else:
            return f"{val:,.2f}"
    
    change_sign = "+" if data.change >= 0 else ""
    change_color = "🟢" if data.change >= 0 else "🔴"
    
    return f"""# {data.ticker} Market Data
**{data.company_name}** | {data.exchange}  
*Last Updated: {data.last_updated}*

---

## Current Quote

| Metric | Value |
|:-------|------:|
| **Price** | **{fmt(data.current_price, "price")}** |
| Change | {change_color} {change_sign}{fmt(data.change, "price")} ({change_sign}{fmt(data.change_pct, "pct")}) |
| Previous Close | {fmt(data.previous_close, "price")} |
| Open | {fmt(data.open, "price")} |
| Day Range | {fmt(data.day_low, "price")} - {fmt(data.day_high, "price")} |
| Volume | {fmt(data.volume, "vol")} |
| Avg Volume | {fmt(data.avg_volume, "vol")} |

## 52-Week Range

| Metric | Value |
|:-------|------:|
| 52-Week High | {fmt(data.week_52_high, "price")} |
| 52-Week Low | {fmt(data.week_52_low, "price")} |
| % from High | {fmt((data.current_price/data.week_52_high - 1) if data.week_52_high else 0, "pct")} |
| % from Low | {fmt((data.current_price/data.week_52_low - 1) if data.week_52_low else 0, "pct")} |

## Market Cap & Shares

| Metric | Value |
|:-------|------:|
| Market Cap | {fmt(data.market_cap, "money")} |
| Shares Outstanding | {fmt(data.shares_outstanding, "vol")} |
| Float | {fmt(data.float_shares, "vol")} |

## Valuation Multiples

| Metric | Value |
|:-------|------:|
| P/E (TTM) | {fmt(data.pe_ratio)}x |
| Forward P/E | {fmt(data.forward_pe)}x |
| PEG Ratio | {fmt(data.peg_ratio)} |
| P/B Ratio | {fmt(data.price_to_book)}x |
| P/S Ratio | {fmt(data.price_to_sales)}x |

## Dividends

| Metric | Value |
|:-------|------:|
| Dividend Rate | {fmt(data.dividend_rate, "price")}/share |
| Dividend Yield | {fmt(data.dividend_yield, "pct")} |
| Ex-Dividend Date | {data.ex_dividend_date or "-"} |

## Risk & Beta

| Metric | Value |
|:-------|------:|
| Beta | {fmt(data.beta)} |

## Analyst Ratings

| Metric | Value |
|:-------|------:|
| Recommendation | **{data.recommendation.upper() if data.recommendation else "-"}** |
| Target (Mean) | {fmt(data.target_mean, "price")} |
| Target (High) | {fmt(data.target_high, "price")} |
| Target (Low) | {fmt(data.target_low, "price")} |
| Upside to Target | {fmt((data.target_mean/data.current_price - 1) if data.current_price and data.target_mean else 0, "pct")} |
| # of Analysts | {data.num_analysts} |

---
"""


async def get_market_data(ticker: str) -> str:
    """Get market data for a ticker and return as Markdown."""
    provider = MarketDataProvider()
    data = await provider.get_quote(ticker)
    return format_market_data_markdown(data)


# CLI
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python market_data.py <TICKER>")
        print("Example: python market_data.py AAPL")
        sys.exit(1)
    
    ticker = sys.argv[1]
    result = asyncio.run(get_market_data(ticker))
    print(result)
