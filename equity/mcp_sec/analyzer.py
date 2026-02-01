"""
SEC Financial Analyzer

Calculates key metrics, ratios, and valuation from SEC financial data.
Produces analysis ready for equity research reports.

Metrics:
  - Profitability: Gross Margin, Operating Margin, Net Margin, ROE, ROA
  - Growth: Revenue Growth, EPS Growth, Net Income Growth
  - Valuation: P/E, Forward P/E, EV/EBITDA, P/B, PEG
  - Health: Debt/Equity, Current Ratio, Interest Coverage
"""

import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import httpx

from .server import SECClient, sec_client


@dataclass
class FinancialMetrics:
    """Calculated financial metrics for a company."""
    ticker: str
    company_name: str
    fiscal_year: int
    period_end: str
    
    # Income Statement
    revenue: float = 0
    gross_profit: float = 0
    operating_income: float = 0
    net_income: float = 0
    eps_basic: float = 0
    eps_diluted: float = 0
    shares_outstanding: float = 0
    
    # Balance Sheet
    total_assets: float = 0
    total_liabilities: float = 0
    total_equity: float = 0
    cash: float = 0
    long_term_debt: float = 0
    current_assets: float = 0
    current_liabilities: float = 0
    
    # Cash Flow
    operating_cash_flow: float = 0
    capex: float = 0
    free_cash_flow: float = 0
    dividends_paid: float = 0
    share_repurchases: float = 0
    
    # Calculated Ratios
    gross_margin: float = 0
    operating_margin: float = 0
    net_margin: float = 0
    roe: float = 0
    roa: float = 0
    debt_to_equity: float = 0
    current_ratio: float = 0


@dataclass
class GrowthMetrics:
    """Year-over-year growth metrics."""
    revenue_growth: float = 0
    gross_profit_growth: float = 0
    operating_income_growth: float = 0
    net_income_growth: float = 0
    eps_growth: float = 0


@dataclass 
class ValuationMetrics:
    """Valuation ratios and fair value estimates."""
    current_price: float = 0
    market_cap: float = 0
    enterprise_value: float = 0
    
    # Multiples
    pe_ratio: float = 0
    forward_pe: float = 0
    peg_ratio: float = 0
    ps_ratio: float = 0
    pb_ratio: float = 0
    ev_ebitda: float = 0
    ev_revenue: float = 0
    
    # Yields
    dividend_yield: float = 0
    fcf_yield: float = 0
    earnings_yield: float = 0
    
    # Fair Value Estimates
    dcf_value: float = 0
    comparable_value: float = 0
    fair_value: float = 0
    upside_pct: float = 0


@dataclass
class EquityAnalysis:
    """Complete equity analysis for research report."""
    ticker: str
    company_name: str
    analysis_date: str
    
    # Current Period Metrics
    metrics: FinancialMetrics
    
    # Historical (prior year)
    prior_metrics: Optional[FinancialMetrics]
    
    # Growth
    growth: GrowthMetrics
    
    # Valuation
    valuation: ValuationMetrics
    
    # Investment Summary
    rating: str  # "Buy", "Hold", "Sell"
    price_target: float
    investment_thesis: List[str]
    key_risks: List[str]


class FinancialAnalyzer:
    """Analyzes SEC financial data and produces research-ready metrics."""
    
    def __init__(self):
        self.sec_client = sec_client
    
    async def get_current_price(self, ticker: str) -> Dict[str, float]:
        """Get current stock price and market data from Yahoo Finance API."""
        # Use a simple quote endpoint
        try:
            async with httpx.AsyncClient() as client:
                # Try Yahoo Finance chart API
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
                resp = await client.get(url, timeout=10.0, headers={
                    "User-Agent": "Mozilla/5.0"
                })
                data = resp.json()
                
                result = data.get("chart", {}).get("result", [{}])[0]
                meta = result.get("meta", {})
                
                return {
                    "price": meta.get("regularMarketPrice", 0),
                    "previous_close": meta.get("previousClose", 0),
                    "market_cap": meta.get("marketCap", 0),
                    "currency": meta.get("currency", "USD")
                }
        except Exception as e:
            print(f"Warning: Could not fetch price for {ticker}: {e}")
            return {"price": 0, "previous_close": 0, "market_cap": 0}
    
    async def analyze(self, ticker: str, form_type: str = "10-K") -> EquityAnalysis:
        """
        Perform complete equity analysis for a ticker.
        
        Returns EquityAnalysis with metrics, growth, valuation, and investment summary.
        """
        ticker = ticker.upper()
        
        # Get CIK and company info
        cik = await self.sec_client.get_cik(ticker)
        if not cik:
            raise ValueError(f"Could not find CIK for {ticker}")
        
        filings = await self.sec_client.get_company_filings(cik, form_type, limit=1)
        company_name = filings[0].company_name if filings else ticker
        
        # Get financial statements
        facts = await self.sec_client.get_company_facts(cik)
        statements = self.sec_client.extract_financial_statements(facts, form_type, 2)
        
        # Extract current and prior period
        income = statements.get("income_statement", [])
        balance = statements.get("balance_sheet", [])
        cashflow = statements.get("cash_flow", [])
        
        # Build current metrics
        current = self._build_metrics(
            ticker, company_name,
            income[0] if income else None,
            balance[0] if balance else None,
            cashflow[0] if cashflow else None
        )
        
        # Build prior metrics
        prior = None
        if len(income) > 1 or len(balance) > 1 or len(cashflow) > 1:
            prior = self._build_metrics(
                ticker, company_name,
                income[1] if len(income) > 1 else None,
                balance[1] if len(balance) > 1 else None,
                cashflow[1] if len(cashflow) > 1 else None
            )
        
        # Calculate growth
        growth = self._calculate_growth(current, prior)
        
        # Get market data and calculate valuation
        market_data = await self.get_current_price(ticker)
        valuation = self._calculate_valuation(current, growth, market_data)
        
        # Generate investment summary
        rating, price_target, thesis, risks = self._generate_investment_summary(
            current, growth, valuation
        )
        
        return EquityAnalysis(
            ticker=ticker,
            company_name=company_name,
            analysis_date=datetime.now().strftime("%Y-%m-%d"),
            metrics=current,
            prior_metrics=prior,
            growth=growth,
            valuation=valuation,
            rating=rating,
            price_target=price_target,
            investment_thesis=thesis,
            key_risks=risks
        )
    
    def _build_metrics(
        self,
        ticker: str,
        company_name: str,
        income: Any,
        balance: Any,
        cashflow: Any
    ) -> FinancialMetrics:
        """Build FinancialMetrics from statement data."""
        m = FinancialMetrics(
            ticker=ticker,
            company_name=company_name,
            fiscal_year=income.fiscal_year if income else 0,
            period_end=income.period_end if income else ""
        )
        
        if income:
            d = income.data
            # Try to get revenue from various fields
            m.revenue = d.get("Revenues", d.get("RevenueFromContractWithCustomerExcludingAssessedTax", 0)) or 0
            m.gross_profit = d.get("GrossProfit", 0) or 0
            m.operating_income = d.get("OperatingIncome", 0) or 0
            m.net_income = d.get("NetIncome", 0) or 0
            m.eps_basic = d.get("EPS_Basic", 0) or 0
            m.eps_diluted = d.get("EPS_Diluted", 0) or 0
            m.shares_outstanding = d.get("SharesOutstanding_Diluted", 0) or 0
            
            # If revenue is 0, try to calculate from gross profit + cost
            if m.revenue == 0 and m.gross_profit > 0:
                cost = d.get("CostOfRevenue", 0) or 0
                m.revenue = m.gross_profit + cost
        
        if balance:
            d = balance.data
            m.total_assets = d.get("TotalAssets", 0) or 0
            m.total_liabilities = d.get("TotalLiabilities", 0) or 0
            m.total_equity = d.get("TotalEquity", 0) or 0
            m.cash = d.get("CashAndEquivalents", 0) or 0
            m.long_term_debt = d.get("LongTermDebt", 0) or 0
            m.current_assets = d.get("TotalCurrentAssets", 0) or 0
            m.current_liabilities = d.get("TotalCurrentLiabilities", 0) or 0
        
        if cashflow:
            d = cashflow.data
            m.operating_cash_flow = d.get("CashFromOperations", 0) or 0
            m.capex = abs(d.get("CapitalExpenditures", 0) or 0)
            m.free_cash_flow = m.operating_cash_flow - m.capex
            m.dividends_paid = abs(d.get("DividendsPaid", 0) or 0)
            m.share_repurchases = abs(d.get("StockRepurchases", 0) or 0)
        
        # Calculate ratios
        if m.revenue > 0:
            m.gross_margin = m.gross_profit / m.revenue
            m.operating_margin = m.operating_income / m.revenue
            m.net_margin = m.net_income / m.revenue
        
        if m.total_equity > 0:
            m.roe = m.net_income / m.total_equity
            m.debt_to_equity = m.long_term_debt / m.total_equity
        
        if m.total_assets > 0:
            m.roa = m.net_income / m.total_assets
        
        if m.current_liabilities > 0:
            m.current_ratio = m.current_assets / m.current_liabilities
        
        return m
    
    def _calculate_growth(
        self,
        current: FinancialMetrics,
        prior: Optional[FinancialMetrics]
    ) -> GrowthMetrics:
        """Calculate YoY growth rates."""
        g = GrowthMetrics()
        
        if prior is None:
            return g
        
        def pct_change(curr, prev):
            if prev and prev != 0:
                return (curr - prev) / abs(prev)
            return 0
        
        g.revenue_growth = pct_change(current.revenue, prior.revenue)
        g.gross_profit_growth = pct_change(current.gross_profit, prior.gross_profit)
        g.operating_income_growth = pct_change(current.operating_income, prior.operating_income)
        g.net_income_growth = pct_change(current.net_income, prior.net_income)
        g.eps_growth = pct_change(current.eps_diluted, prior.eps_diluted)
        
        return g
    
    def _calculate_valuation(
        self,
        metrics: FinancialMetrics,
        growth: GrowthMetrics,
        market_data: Dict[str, float]
    ) -> ValuationMetrics:
        """Calculate valuation metrics."""
        v = ValuationMetrics()
        
        v.current_price = market_data.get("price", 0)
        v.market_cap = market_data.get("market_cap", 0)
        
        # If no market cap from API, estimate from shares * price
        if v.market_cap == 0 and v.current_price > 0 and metrics.shares_outstanding > 0:
            v.market_cap = v.current_price * metrics.shares_outstanding
        
        # Enterprise Value = Market Cap + Debt - Cash
        v.enterprise_value = v.market_cap + metrics.long_term_debt - metrics.cash
        
        # P/E Ratio
        if metrics.eps_diluted > 0 and v.current_price > 0:
            v.pe_ratio = v.current_price / metrics.eps_diluted
        
        # Forward P/E (estimate next year EPS with growth)
        if metrics.eps_diluted > 0 and growth.eps_growth != 0:
            forward_eps = metrics.eps_diluted * (1 + growth.eps_growth)
            if forward_eps > 0:
                v.forward_pe = v.current_price / forward_eps
        
        # PEG Ratio
        if v.pe_ratio > 0 and growth.eps_growth > 0:
            v.peg_ratio = v.pe_ratio / (growth.eps_growth * 100)
        
        # P/S Ratio
        if metrics.revenue > 0 and v.market_cap > 0:
            v.ps_ratio = v.market_cap / metrics.revenue
        
        # P/B Ratio
        if metrics.total_equity > 0 and v.market_cap > 0:
            v.pb_ratio = v.market_cap / metrics.total_equity
        
        # EV/EBITDA (approximate EBITDA as operating income + D&A, assume 10% of revenue)
        ebitda = metrics.operating_income * 1.15  # Rough approximation
        if ebitda > 0:
            v.ev_ebitda = v.enterprise_value / ebitda
        
        # EV/Revenue
        if metrics.revenue > 0:
            v.ev_revenue = v.enterprise_value / metrics.revenue
        
        # Yields
        if v.market_cap > 0:
            v.dividend_yield = metrics.dividends_paid / v.market_cap
            v.fcf_yield = metrics.free_cash_flow / v.market_cap
        
        if v.pe_ratio > 0:
            v.earnings_yield = 1 / v.pe_ratio
        
        # Fair Value Estimates
        v.dcf_value = self._dcf_valuation(metrics, growth)
        v.comparable_value = self._comparable_valuation(metrics, v.pe_ratio)
        v.fair_value = (v.dcf_value * 0.5 + v.comparable_value * 0.5)
        
        if v.current_price > 0:
            v.upside_pct = (v.fair_value / v.current_price - 1)
        
        return v
    
    def _dcf_valuation(self, metrics: FinancialMetrics, growth: GrowthMetrics) -> float:
        """Simple DCF valuation."""
        if metrics.free_cash_flow <= 0:
            return 0
        
        fcf = metrics.free_cash_flow
        growth_rate = min(max(growth.net_income_growth, 0.03), 0.25)  # 3-25%
        discount_rate = 0.10
        terminal_growth = 0.03
        
        # 5-year FCF projection
        total_pv = 0
        for year in range(1, 6):
            future_fcf = fcf * ((1 + growth_rate) ** year)
            pv = future_fcf / ((1 + discount_rate) ** year)
            total_pv += pv
        
        # Terminal value
        terminal_fcf = fcf * ((1 + growth_rate) ** 5) * (1 + terminal_growth)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth)
        terminal_pv = terminal_value / ((1 + discount_rate) ** 5)
        
        total_value = total_pv + terminal_pv
        
        if metrics.shares_outstanding > 0:
            return total_value / metrics.shares_outstanding
        return 0
    
    def _comparable_valuation(self, metrics: FinancialMetrics, pe_ratio: float) -> float:
        """Comparable company valuation using sector median P/E."""
        sector_pe = 22  # Assumed sector median
        
        if metrics.eps_diluted > 0:
            return metrics.eps_diluted * sector_pe
        return 0
    
    def _generate_investment_summary(
        self,
        metrics: FinancialMetrics,
        growth: GrowthMetrics,
        valuation: ValuationMetrics
    ) -> tuple:
        """Generate rating, price target, thesis, and risks."""
        
        # Determine rating based on upside
        upside = valuation.upside_pct
        if upside >= 0.20:
            rating = "Buy"
        elif upside >= -0.10:
            rating = "Hold"
        else:
            rating = "Sell"
        
        price_target = valuation.fair_value
        
        # Generate thesis points
        thesis = []
        if metrics.net_margin > 0.15:
            thesis.append(f"Strong profitability with {metrics.net_margin:.1%} net margin")
        if growth.revenue_growth > 0.10:
            thesis.append(f"Robust revenue growth of {growth.revenue_growth:.1%} YoY")
        if growth.eps_growth > 0.15:
            thesis.append(f"EPS growth of {growth.eps_growth:.1%} demonstrates earnings power")
        if metrics.roe > 0.15:
            thesis.append(f"High return on equity of {metrics.roe:.1%}")
        if metrics.free_cash_flow > 0:
            thesis.append(f"Generating ${metrics.free_cash_flow/1e9:.1f}B in free cash flow")
        if valuation.peg_ratio > 0 and valuation.peg_ratio < 1.5:
            thesis.append(f"Attractive PEG ratio of {valuation.peg_ratio:.2f}")
        
        if not thesis:
            thesis = ["Market position and fundamentals warrant current valuation"]
        
        # Generate risk points
        risks = []
        if metrics.debt_to_equity > 1.0:
            risks.append(f"Elevated leverage with {metrics.debt_to_equity:.2f}x debt/equity")
        if growth.revenue_growth < 0:
            risks.append("Revenue decline indicates business challenges")
        if valuation.pe_ratio > 35:
            risks.append(f"Premium valuation at {valuation.pe_ratio:.1f}x P/E")
        if metrics.net_margin < 0.05:
            risks.append("Thin profit margins limit earnings flexibility")
        
        risks.append("Macroeconomic uncertainty could impact demand")
        risks.append("Competitive pressures in core markets")
        
        return rating, price_target, thesis[:4], risks[:4]


def format_analysis_markdown(analysis: EquityAnalysis) -> str:
    """Format equity analysis as Markdown for research report."""
    
    def fmt_num(n, is_pct=False, is_money=False, decimals=2):
        if n is None or n == 0:
            return "-"
        if is_pct:
            return f"{n*100:.1f}%"
        if is_money:
            if abs(n) >= 1e9:
                return f"${n/1e9:.2f}B"
            elif abs(n) >= 1e6:
                return f"${n/1e6:.1f}M"
            else:
                return f"${n:,.2f}"
        return f"{n:.{decimals}f}"
    
    m = analysis.metrics
    g = analysis.growth
    v = analysis.valuation
    
    output = f"""# {analysis.ticker} Equity Analysis
**{analysis.company_name}**  
*Analysis Date: {analysis.analysis_date}*

---

## Investment Summary

| Metric | Value |
|:-------|------:|
| **Rating** | **{analysis.rating}** |
| **Current Price** | {fmt_num(v.current_price, is_money=True)} |
| **Price Target** | {fmt_num(analysis.price_target, is_money=True)} |
| **Upside** | {fmt_num(v.upside_pct, is_pct=True)} |
| **Market Cap** | {fmt_num(v.market_cap, is_money=True)} |

### Investment Thesis
"""
    
    for point in analysis.investment_thesis:
        output += f"- {point}\n"
    
    output += """
### Key Risks
"""
    for risk in analysis.key_risks:
        output += f"- {risk}\n"
    
    output += f"""
---

## Financial Metrics (FY{m.fiscal_year})

### Profitability

| Metric | Value |
|:-------|------:|
| Revenue | {fmt_num(m.revenue, is_money=True)} |
| Gross Profit | {fmt_num(m.gross_profit, is_money=True)} |
| Operating Income | {fmt_num(m.operating_income, is_money=True)} |
| Net Income | {fmt_num(m.net_income, is_money=True)} |
| EPS (Diluted) | ${m.eps_diluted:.2f} |

### Margins & Returns

| Metric | Value |
|:-------|------:|
| Gross Margin | {fmt_num(m.gross_margin, is_pct=True)} |
| Operating Margin | {fmt_num(m.operating_margin, is_pct=True)} |
| Net Margin | {fmt_num(m.net_margin, is_pct=True)} |
| ROE | {fmt_num(m.roe, is_pct=True)} |
| ROA | {fmt_num(m.roa, is_pct=True)} |

### Growth (YoY)

| Metric | Growth |
|:-------|-------:|
| Revenue | {fmt_num(g.revenue_growth, is_pct=True)} |
| Operating Income | {fmt_num(g.operating_income_growth, is_pct=True)} |
| Net Income | {fmt_num(g.net_income_growth, is_pct=True)} |
| EPS | {fmt_num(g.eps_growth, is_pct=True)} |

---

## Valuation

| Metric | Value |
|:-------|------:|
| P/E Ratio | {fmt_num(v.pe_ratio)}x |
| Forward P/E | {fmt_num(v.forward_pe)}x |
| PEG Ratio | {fmt_num(v.peg_ratio)} |
| EV/EBITDA | {fmt_num(v.ev_ebitda)}x |
| P/S Ratio | {fmt_num(v.ps_ratio)}x |
| P/B Ratio | {fmt_num(v.pb_ratio)}x |

### Yields

| Metric | Value |
|:-------|------:|
| Earnings Yield | {fmt_num(v.earnings_yield, is_pct=True)} |
| FCF Yield | {fmt_num(v.fcf_yield, is_pct=True)} |
| Dividend Yield | {fmt_num(v.dividend_yield, is_pct=True)} |

### Fair Value Estimates

| Method | Value |
|:-------|------:|
| DCF Valuation | {fmt_num(v.dcf_value, is_money=True)} |
| Comparable Analysis | {fmt_num(v.comparable_value, is_money=True)} |
| **Blended Fair Value** | **{fmt_num(v.fair_value, is_money=True)}** |

---

## Balance Sheet Highlights

| Metric | Value |
|:-------|------:|
| Total Assets | {fmt_num(m.total_assets, is_money=True)} |
| Total Liabilities | {fmt_num(m.total_liabilities, is_money=True)} |
| Total Equity | {fmt_num(m.total_equity, is_money=True)} |
| Cash | {fmt_num(m.cash, is_money=True)} |
| Long-term Debt | {fmt_num(m.long_term_debt, is_money=True)} |
| Debt/Equity | {fmt_num(m.debt_to_equity)}x |
| Current Ratio | {fmt_num(m.current_ratio)}x |

## Cash Flow

| Metric | Value |
|:-------|------:|
| Operating Cash Flow | {fmt_num(m.operating_cash_flow, is_money=True)} |
| Capital Expenditures | {fmt_num(m.capex, is_money=True)} |
| Free Cash Flow | {fmt_num(m.free_cash_flow, is_money=True)} |
| Dividends Paid | {fmt_num(m.dividends_paid, is_money=True)} |
| Share Repurchases | {fmt_num(m.share_repurchases, is_money=True)} |

---

*Data sourced from SEC EDGAR filings. Fair value estimates are illustrative.*
"""
    
    return output


async def analyze_ticker(ticker: str, form_type: str = "10-K") -> str:
    """Analyze a ticker and return Markdown report."""
    analyzer = FinancialAnalyzer()
    analysis = await analyzer.analyze(ticker, form_type)
    return format_analysis_markdown(analysis)


# CLI
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyzer.py <TICKER>")
        print("Example: python analyzer.py AAPL")
        sys.exit(1)
    
    ticker = sys.argv[1]
    form_type = sys.argv[2] if len(sys.argv) > 2 else "10-K"
    
    result = asyncio.run(analyze_ticker(ticker, form_type))
    print(result)
