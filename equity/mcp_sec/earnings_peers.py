"""
Earnings Calendar & Peer Comparison

Provides:
  - Earnings dates and estimates
  - Industry peer identification
  - Comparative analysis across peers
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import httpx

from .server import sec_client, SEC_EDGAR_BASE, USER_AGENT


# Industry peer mapping (SIC codes to common peers)
INDUSTRY_PEERS = {
    # Technology
    "3571": ["AAPL", "MSFT", "GOOGL", "META", "AMZN"],  # Electronic Computers
    "7370": ["MSFT", "CRM", "ORCL", "SAP", "ADBE"],  # Computer Programming Services
    "7372": ["MSFT", "ADBE", "CRM", "ORCL", "INTU"],  # Prepackaged Software
    "3674": ["NVDA", "AMD", "INTC", "QCOM", "AVGO"],  # Semiconductors
    
    # Financials
    "6021": ["JPM", "BAC", "WFC", "C", "USB"],  # Commercial Banks
    "6211": ["GS", "MS", "JPM", "SCHW", "RJF"],  # Security Brokers
    "6282": ["BLK", "BX", "KKR", "APO", "ARES"],  # Investment Advisers
    
    # Healthcare
    "2834": ["JNJ", "PFE", "MRK", "ABBV", "LLY"],  # Pharmaceutical Preparations
    "3841": ["MDT", "ABT", "SYK", "BSX", "EW"],  # Surgical Instruments
    
    # Consumer
    "5331": ["WMT", "TGT", "COST", "DG", "DLTR"],  # Variety Stores
    "5812": ["MCD", "SBUX", "CMG", "YUM", "DRI"],  # Eating Places
    "5411": ["KR", "WMT", "COST", "ACI", "SFM"],  # Grocery Stores
    
    # Industrial
    "3711": ["F", "GM", "TSLA", "TM", "HMC"],  # Motor Vehicles
    "4512": ["DAL", "UAL", "AAL", "LUV", "JBLU"],  # Air Transportation
    "3721": ["BA", "LMT", "RTX", "NOC", "GD"],  # Aircraft
    
    # Energy
    "1311": ["XOM", "CVX", "COP", "EOG", "PXD"],  # Crude Petroleum
    "4911": ["NEE", "DUK", "SO", "D", "AEP"],  # Electric Services
}

# Common company -> SIC mapping
COMPANY_SIC = {
    "AAPL": "3571", "MSFT": "7372", "GOOGL": "7370", "AMZN": "5961", "META": "7370",
    "NVDA": "3674", "AMD": "3674", "INTC": "3674", "TSLA": "3711", "JPM": "6021",
    "BAC": "6021", "WFC": "6021", "GS": "6211", "MS": "6211", "JNJ": "2834",
    "PFE": "2834", "MRK": "2834", "WMT": "5331", "COST": "5331", "TGT": "5331",
    "XOM": "1311", "CVX": "1311", "BA": "3721", "LMT": "3721"
}


@dataclass
class EarningsInfo:
    """Earnings information for a company."""
    ticker: str
    company_name: str
    next_earnings_date: str
    fiscal_quarter: str
    
    # Estimates
    eps_estimate: float
    eps_actual: Optional[float]
    revenue_estimate: float
    revenue_actual: Optional[float]
    
    # Historical
    last_eps: float
    last_eps_surprise: float
    last_eps_surprise_pct: float
    
    # Beat/Miss history
    beats_last_4: int
    misses_last_4: int


@dataclass
class PeerMetrics:
    """Comparative metrics for a peer."""
    ticker: str
    company_name: str
    market_cap: float
    
    # Valuation
    pe_ratio: float
    forward_pe: float
    ev_ebitda: float
    ps_ratio: float
    
    # Profitability
    gross_margin: float
    operating_margin: float
    net_margin: float
    roe: float
    
    # Growth
    revenue_growth: float
    eps_growth: float


class EarningsPeersClient:
    """Client for earnings and peer comparison data."""
    
    def __init__(self):
        self.headers = {
            "User-Agent": USER_AGENT,
            "Accept": "application/json"
        }
    
    async def get_earnings_info(self, ticker: str) -> EarningsInfo:
        """
        Get earnings information for a ticker.
        
        Note: Actual earnings dates require specialized data providers.
        This uses SEC filing patterns to estimate.
        """
        ticker = ticker.upper()
        
        # Get company info
        cik = await sec_client.get_cik(ticker)
        company_name = ticker
        
        if cik:
            filings = await sec_client.get_company_filings(cik, "10-K", limit=1)
            if filings:
                company_name = filings[0].company_name
        
        # Estimate next earnings based on typical patterns
        # Most companies report ~45 days after quarter end
        now = datetime.now()
        
        # Determine current fiscal quarter
        if now.month in [1, 2]:
            next_q = "Q1"
            est_date = now.replace(month=4, day=15)
        elif now.month in [3, 4, 5]:
            next_q = "Q2"
            est_date = now.replace(month=7, day=15)
        elif now.month in [6, 7, 8]:
            next_q = "Q3"
            est_date = now.replace(month=10, day=15)
        else:
            next_q = "Q4"
            est_date = now.replace(year=now.year + 1, month=1, day=15)
        
        # Get last earnings from SEC 10-Q
        last_eps = 0.0
        if cik:
            facts = await sec_client.get_company_facts(cik)
            statements = sec_client.extract_financial_statements(facts, "10-Q", 1)
            income = statements.get("income_statement", [])
            if income:
                last_eps = income[0].data.get("EPS_Diluted", 0) or 0
        
        return EarningsInfo(
            ticker=ticker,
            company_name=company_name,
            next_earnings_date=est_date.strftime("%Y-%m-%d") + " (Est.)",
            fiscal_quarter=next_q,
            eps_estimate=last_eps * 1.05 if last_eps else 0,  # Placeholder
            eps_actual=None,
            revenue_estimate=0,
            revenue_actual=None,
            last_eps=last_eps,
            last_eps_surprise=0,
            last_eps_surprise_pct=0,
            beats_last_4=0,
            misses_last_4=0
        )
    
    async def find_peers(self, ticker: str, limit: int = 5) -> List[str]:
        """Find industry peers for a ticker."""
        ticker = ticker.upper()
        
        # Check if we have a known SIC code
        sic = COMPANY_SIC.get(ticker)
        
        if sic and sic in INDUSTRY_PEERS:
            peers = [p for p in INDUSTRY_PEERS[sic] if p != ticker]
            return peers[:limit]
        
        # Try to get SIC from SEC
        cik = await sec_client.get_cik(ticker)
        if cik:
            try:
                url = f"{SEC_EDGAR_BASE}/submissions/CIK{cik.zfill(10)}.json"
                async with httpx.AsyncClient() as client:
                    resp = await client.get(url, headers=self.headers, timeout=15.0)
                    if resp.status_code == 200:
                        data = resp.json()
                        sic = data.get("sic", "")
                        
                        if sic and sic in INDUSTRY_PEERS:
                            peers = [p for p in INDUSTRY_PEERS[sic] if p != ticker]
                            return peers[:limit]
            except:
                pass
        
        # Default peers for unknown companies
        return ["SPY", "QQQ"][:limit]
    
    async def get_peer_comparison(
        self,
        ticker: str,
        peers: Optional[List[str]] = None,
        limit: int = 5
    ) -> Tuple[PeerMetrics, List[PeerMetrics]]:
        """
        Get comparative metrics for a ticker and its peers.
        
        Returns: (target_metrics, [peer_metrics])
        """
        ticker = ticker.upper()
        
        if peers is None:
            peers = await self.find_peers(ticker, limit)
        
        # Get metrics for target
        target = await self._get_metrics(ticker)
        
        # Get metrics for peers
        peer_metrics = []
        for peer in peers[:limit]:
            try:
                pm = await self._get_metrics(peer)
                peer_metrics.append(pm)
            except:
                pass
        
        return target, peer_metrics
    
    async def _get_metrics(self, ticker: str) -> PeerMetrics:
        """Get metrics for a single ticker."""
        from .analyzer import FinancialAnalyzer
        
        analyzer = FinancialAnalyzer()
        
        try:
            analysis = await analyzer.analyze(ticker)
            m = analysis.metrics
            g = analysis.growth
            v = analysis.valuation
            
            return PeerMetrics(
                ticker=ticker,
                company_name=m.company_name,
                market_cap=v.market_cap,
                pe_ratio=v.pe_ratio,
                forward_pe=v.forward_pe,
                ev_ebitda=v.ev_ebitda,
                ps_ratio=v.ps_ratio,
                gross_margin=m.gross_margin,
                operating_margin=m.operating_margin,
                net_margin=m.net_margin,
                roe=m.roe,
                revenue_growth=g.revenue_growth,
                eps_growth=g.eps_growth
            )
        except Exception as e:
            # Return empty metrics on error
            return PeerMetrics(
                ticker=ticker,
                company_name=ticker,
                market_cap=0,
                pe_ratio=0,
                forward_pe=0,
                ev_ebitda=0,
                ps_ratio=0,
                gross_margin=0,
                operating_margin=0,
                net_margin=0,
                roe=0,
                revenue_growth=0,
                eps_growth=0
            )


def format_earnings_markdown(info: EarningsInfo) -> str:
    """Format earnings info as Markdown."""
    
    return f"""# {info.ticker} Earnings
**{info.company_name}**

## Next Earnings

| Metric | Value |
|:-------|------:|
| **Next Report** | {info.next_earnings_date} |
| Fiscal Quarter | {info.fiscal_quarter} |
| EPS Estimate | ${info.eps_estimate:.2f} |
| Last EPS | ${info.last_eps:.2f} |

## Historical Performance

| Metric | Value |
|:-------|------:|
| Beats (Last 4Q) | {info.beats_last_4} |
| Misses (Last 4Q) | {info.misses_last_4} |
| Avg Surprise % | {info.last_eps_surprise_pct:.1f}% |

*Note: Earnings dates are estimates. Check investor relations for confirmed dates.*
"""


def format_peer_comparison_markdown(
    target: PeerMetrics,
    peers: List[PeerMetrics]
) -> str:
    """Format peer comparison as Markdown."""
    
    def fmt_num(val, fmt_type="num"):
        if val is None or val == 0:
            return "-"
        if fmt_type == "money":
            if abs(val) >= 1e12:
                return f"${val/1e12:.1f}T"
            elif abs(val) >= 1e9:
                return f"${val/1e9:.0f}B"
            elif abs(val) >= 1e6:
                return f"${val/1e6:.0f}M"
            return f"${val:,.0f}"
        elif fmt_type == "pct":
            return f"{val*100:.1f}%"
        elif fmt_type == "ratio":
            return f"{val:.1f}x"
        return f"{val:.2f}"
    
    all_companies = [target] + peers
    
    output = [
        f"# {target.ticker} Peer Comparison",
        f"**{target.company_name}** vs Industry Peers\n",
        "## Valuation Multiples\n",
        "| Company | Mkt Cap | P/E | Fwd P/E | EV/EBITDA | P/S |",
        "|:--------|--------:|----:|--------:|----------:|----:|"
    ]
    
    for c in all_companies:
        name = f"**{c.ticker}**" if c.ticker == target.ticker else c.ticker
        output.append(
            f"| {name} | {fmt_num(c.market_cap, 'money')} | "
            f"{fmt_num(c.pe_ratio, 'ratio')} | {fmt_num(c.forward_pe, 'ratio')} | "
            f"{fmt_num(c.ev_ebitda, 'ratio')} | {fmt_num(c.ps_ratio, 'ratio')} |"
        )
    
    output.extend([
        "\n## Profitability Margins\n",
        "| Company | Gross | Operating | Net | ROE |",
        "|:--------|------:|----------:|----:|----:|"
    ])
    
    for c in all_companies:
        name = f"**{c.ticker}**" if c.ticker == target.ticker else c.ticker
        output.append(
            f"| {name} | {fmt_num(c.gross_margin, 'pct')} | "
            f"{fmt_num(c.operating_margin, 'pct')} | {fmt_num(c.net_margin, 'pct')} | "
            f"{fmt_num(c.roe, 'pct')} |"
        )
    
    output.extend([
        "\n## Growth Rates\n",
        "| Company | Revenue Growth | EPS Growth |",
        "|:--------|---------------:|-----------:|"
    ])
    
    for c in all_companies:
        name = f"**{c.ticker}**" if c.ticker == target.ticker else c.ticker
        output.append(
            f"| {name} | {fmt_num(c.revenue_growth, 'pct')} | "
            f"{fmt_num(c.eps_growth, 'pct')} |"
        )
    
    # Calculate averages
    peer_pe_avg = sum(p.pe_ratio for p in peers if p.pe_ratio) / max(len([p for p in peers if p.pe_ratio]), 1)
    peer_margin_avg = sum(p.net_margin for p in peers if p.net_margin) / max(len([p for p in peers if p.net_margin]), 1)
    
    output.extend([
        "\n## Summary\n",
        f"- **{target.ticker} P/E**: {fmt_num(target.pe_ratio, 'ratio')} vs Peer Avg {fmt_num(peer_pe_avg, 'ratio')}",
        f"- **{target.ticker} Net Margin**: {fmt_num(target.net_margin, 'pct')} vs Peer Avg {fmt_num(peer_margin_avg, 'pct')}",
        "",
        "*Data from SEC 10-K filings*"
    ])
    
    return "\n".join(output)


# Async handlers
async def get_earnings(ticker: str) -> str:
    """Get earnings info as Markdown."""
    client = EarningsPeersClient()
    info = await client.get_earnings_info(ticker)
    return format_earnings_markdown(info)


async def get_peers(ticker: str, limit: int = 5) -> str:
    """Get peer list."""
    client = EarningsPeersClient()
    peers = await client.find_peers(ticker, limit)
    return f"# {ticker.upper()} Industry Peers\n\n" + "\n".join([f"- {p}" for p in peers])


async def get_peer_comparison(ticker: str, limit: int = 4) -> str:
    """Get full peer comparison as Markdown."""
    client = EarningsPeersClient()
    target, peers = await client.get_peer_comparison(ticker, limit=limit)
    return format_peer_comparison_markdown(target, peers)


# CLI
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python earnings_peers.py <earnings|peers|compare> <TICKER>")
        print("Example: python earnings_peers.py compare AAPL")
        sys.exit(1)
    
    cmd = sys.argv[1]
    ticker = sys.argv[2] if len(sys.argv) > 2 else "AAPL"
    
    if cmd == "earnings":
        print(asyncio.run(get_earnings(ticker)))
    elif cmd == "peers":
        print(asyncio.run(get_peers(ticker)))
    elif cmd == "compare":
        print(asyncio.run(get_peer_comparison(ticker)))
    else:
        print(f"Unknown command: {cmd}")
