#!/usr/bin/env python3
"""
SEC EDGAR MCP Server

Provides tools to fetch and parse SEC filings (10-K, 10-Q) and extract
the three main financial statements:
  1. Income Statement (Statement of Operations)
  2. Balance Sheet (Statement of Financial Position)  
  3. Cash Flow Statement (Statement of Cash Flows)

Run with: python -m mcp_sec.server
"""

import asyncio
import json
import re
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import httpx

# MCP SDK imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️  MCP SDK not installed. Run: pip install mcp")


# SEC EDGAR API configuration
SEC_EDGAR_BASE = "https://data.sec.gov"
SEC_SEARCH_BASE = "https://efts.sec.gov/LATEST/search-index"
SEC_COMPANY_TICKERS = "https://www.sec.gov/files/company_tickers.json"

# User agent required by SEC
USER_AGENT = "EquityResearchBot/1.0 (contact@example.com)"


@dataclass
class FinancialStatement:
    """Represents a parsed financial statement."""
    statement_type: str  # "income", "balance", "cashflow"
    period_end: str
    period_type: str  # "annual" or "quarterly"
    fiscal_year: int
    fiscal_quarter: Optional[int]
    currency: str
    data: Dict[str, Any]  # Line items with values


@dataclass
class FilingInfo:
    """Information about an SEC filing."""
    cik: str
    company_name: str
    form_type: str  # "10-K" or "10-Q"
    filing_date: str
    accession_number: str
    primary_document: str
    fiscal_year_end: str


class SECClient:
    """Client for SEC EDGAR API."""
    
    def __init__(self):
        self.headers = {
            "User-Agent": USER_AGENT,
            "Accept": "application/json"
        }
        self._cik_cache: Dict[str, str] = {}
    
    async def get_cik(self, ticker: str) -> Optional[str]:
        """Get CIK number for a ticker symbol."""
        ticker = ticker.upper()
        
        if ticker in self._cik_cache:
            return self._cik_cache[ticker]
        
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(
                    SEC_COMPANY_TICKERS,
                    headers=self.headers,
                    timeout=30.0
                )
                resp.raise_for_status()
                data = resp.json()
                
                # Search for ticker
                for entry in data.values():
                    if entry.get("ticker", "").upper() == ticker:
                        cik = str(entry["cik_str"]).zfill(10)
                        self._cik_cache[ticker] = cik
                        return cik
                
                return None
            except Exception as e:
                print(f"Error fetching CIK: {e}")
                return None
    
    async def get_company_filings(
        self,
        cik: str,
        form_type: str = "10-K",
        limit: int = 5
    ) -> List[FilingInfo]:
        """Get list of filings for a company."""
        cik = cik.zfill(10)
        url = f"{SEC_EDGAR_BASE}/submissions/CIK{cik}.json"
        
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(url, headers=self.headers, timeout=30.0)
                resp.raise_for_status()
                data = resp.json()
                
                filings = []
                recent = data.get("filings", {}).get("recent", {})
                
                forms = recent.get("form", [])
                dates = recent.get("filingDate", [])
                accessions = recent.get("accessionNumber", [])
                primary_docs = recent.get("primaryDocument", [])
                
                for i, form in enumerate(forms):
                    if form == form_type and len(filings) < limit:
                        filings.append(FilingInfo(
                            cik=cik,
                            company_name=data.get("name", ""),
                            form_type=form,
                            filing_date=dates[i] if i < len(dates) else "",
                            accession_number=accessions[i] if i < len(accessions) else "",
                            primary_document=primary_docs[i] if i < len(primary_docs) else "",
                            fiscal_year_end=data.get("fiscalYearEnd", "")
                        ))
                
                return filings
                
            except Exception as e:
                print(f"Error fetching filings: {e}")
                return []
    
    async def get_company_facts(self, cik: str) -> Dict[str, Any]:
        """Get company facts (XBRL data) from SEC."""
        cik = cik.zfill(10)
        url = f"{SEC_EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
        
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(url, headers=self.headers, timeout=60.0)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                print(f"Error fetching company facts: {e}")
                return {}
    
    def extract_financial_statements(
        self,
        facts: Dict[str, Any],
        form_type: str = "10-K",
        num_periods: int = 4
    ) -> Dict[str, List[FinancialStatement]]:
        """
        Extract the three main financial statements from company facts.
        
        Returns dict with keys: "income_statement", "balance_sheet", "cash_flow"
        """
        us_gaap = facts.get("facts", {}).get("us-gaap", {})
        
        # Define the key line items for each statement
        income_statement_items = {
            "Revenues": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", 
                        "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"],
            "CostOfRevenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
            "GrossProfit": ["GrossProfit"],
            "OperatingExpenses": ["OperatingExpenses", "OperatingCostsAndExpenses"],
            "ResearchAndDevelopment": ["ResearchAndDevelopmentExpense"],
            "SellingGeneralAdmin": ["SellingGeneralAndAdministrativeExpense"],
            "OperatingIncome": ["OperatingIncomeLoss"],
            "InterestExpense": ["InterestExpense", "InterestExpenseDebt"],
            "InterestIncome": ["InterestIncome", "InvestmentIncomeInterest"],
            "OtherIncome": ["OtherNonoperatingIncomeExpense", "NonoperatingIncomeExpense"],
            "IncomeBeforeTax": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                               "IncomeLossFromContinuingOperationsBeforeIncomeTaxes"],
            "IncomeTaxExpense": ["IncomeTaxExpenseBenefit"],
            "NetIncome": ["NetIncomeLoss", "ProfitLoss"],
            "EPS_Basic": ["EarningsPerShareBasic"],
            "EPS_Diluted": ["EarningsPerShareDiluted"],
            "SharesOutstanding_Basic": ["WeightedAverageNumberOfSharesOutstandingBasic"],
            "SharesOutstanding_Diluted": ["WeightedAverageNumberOfDilutedSharesOutstanding"]
        }
        
        balance_sheet_items = {
            # Assets
            "CashAndEquivalents": ["CashAndCashEquivalentsAtCarryingValue", "Cash"],
            "ShortTermInvestments": ["ShortTermInvestments", "MarketableSecuritiesCurrent"],
            "AccountsReceivable": ["AccountsReceivableNetCurrent", "AccountsReceivableNet"],
            "Inventory": ["InventoryNet", "InventoryFinishedGoods"],
            "PrepaidExpenses": ["PrepaidExpenseAndOtherAssetsCurrent", "PrepaidExpenseCurrent"],
            "TotalCurrentAssets": ["AssetsCurrent"],
            "PropertyPlantEquipment": ["PropertyPlantAndEquipmentNet"],
            "Goodwill": ["Goodwill"],
            "IntangibleAssets": ["IntangibleAssetsNetExcludingGoodwill", "FiniteLivedIntangibleAssetsNet"],
            "LongTermInvestments": ["LongTermInvestments", "MarketableSecuritiesNoncurrent"],
            "OtherAssets": ["OtherAssetsNoncurrent"],
            "TotalAssets": ["Assets"],
            # Liabilities
            "AccountsPayable": ["AccountsPayableCurrent"],
            "ShortTermDebt": ["ShortTermBorrowings", "DebtCurrent"],
            "AccruedLiabilities": ["AccruedLiabilitiesCurrent"],
            "DeferredRevenue": ["DeferredRevenueCurrent", "ContractWithCustomerLiabilityCurrent"],
            "TotalCurrentLiabilities": ["LiabilitiesCurrent"],
            "LongTermDebt": ["LongTermDebt", "LongTermDebtNoncurrent"],
            "DeferredTaxLiabilities": ["DeferredIncomeTaxLiabilitiesNet"],
            "OtherLiabilities": ["OtherLiabilitiesNoncurrent"],
            "TotalLiabilities": ["Liabilities"],
            # Equity
            "CommonStock": ["CommonStockValue"],
            "RetainedEarnings": ["RetainedEarningsAccumulatedDeficit"],
            "AOCI": ["AccumulatedOtherComprehensiveIncomeLossNetOfTax"],
            "TreasuryStock": ["TreasuryStockValue"],
            "TotalEquity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
            "TotalLiabilitiesAndEquity": ["LiabilitiesAndStockholdersEquity"]
        }
        
        cash_flow_items = {
            # Operating Activities
            "NetIncome_CF": ["NetIncomeLoss", "ProfitLoss"],
            "DepreciationAmortization": ["DepreciationDepletionAndAmortization", "Depreciation"],
            "StockBasedCompensation": ["ShareBasedCompensation"],
            "DeferredIncomeTax": ["DeferredIncomeTaxExpenseBenefit"],
            "ChangeInReceivables": ["IncreaseDecreaseInAccountsReceivable"],
            "ChangeInInventory": ["IncreaseDecreaseInInventories"],
            "ChangeInPayables": ["IncreaseDecreaseInAccountsPayable"],
            "OtherOperating": ["OtherOperatingActivitiesCashFlowStatement"],
            "CashFromOperations": ["NetCashProvidedByUsedInOperatingActivities"],
            # Investing Activities
            "CapitalExpenditures": ["PaymentsToAcquirePropertyPlantAndEquipment"],
            "Acquisitions": ["PaymentsToAcquireBusinessesNetOfCashAcquired"],
            "InvestmentPurchases": ["PaymentsToAcquireInvestments", "PaymentsToAcquireMarketableSecurities"],
            "InvestmentSales": ["ProceedsFromSaleOfInvestments", "ProceedsFromMaturitiesPaydownsAndCallsOfAvailableForSaleSecurities"],
            "OtherInvesting": ["PaymentsForProceedsFromOtherInvestingActivities"],
            "CashFromInvesting": ["NetCashProvidedByUsedInInvestingActivities"],
            # Financing Activities
            "DebtProceeds": ["ProceedsFromIssuanceOfLongTermDebt"],
            "DebtRepayments": ["RepaymentsOfLongTermDebt"],
            "StockRepurchases": ["PaymentsForRepurchaseOfCommonStock"],
            "DividendsPaid": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
            "StockIssuance": ["ProceedsFromIssuanceOfCommonStock"],
            "OtherFinancing": ["ProceedsFromPaymentsForOtherFinancingActivities"],
            "CashFromFinancing": ["NetCashProvidedByUsedInFinancingActivities"],
            # Summary
            "NetChangeInCash": ["CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect",
                               "CashAndCashEquivalentsPeriodIncreaseDecrease"],
            "BeginningCash": ["CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsIncludingDisposalGroupAndDiscontinuedOperations"],
            "EndingCash": ["CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"]
        }
        
        def extract_values(item_mapping: Dict[str, List[str]], form: str) -> List[Dict[str, Any]]:
            """Extract values for all items in a statement."""
            # Collect all periods first
            periods = {}  # {(fiscal_year, fiscal_period): {item: value}}
            
            for label, xbrl_tags in item_mapping.items():
                for tag in xbrl_tags:
                    if tag in us_gaap:
                        units = us_gaap[tag].get("units", {})
                        # Try USD first, then shares
                        values = units.get("USD", units.get("USD/shares", units.get("shares", [])))
                        
                        for entry in values:
                            # Filter by form type
                            entry_form = entry.get("form", "")
                            if form == "10-K" and entry_form != "10-K":
                                continue
                            if form == "10-Q" and entry_form != "10-Q":
                                continue
                            
                            fy = entry.get("fy")
                            fp = entry.get("fp", "FY")
                            end_date = entry.get("end", "")
                            val = entry.get("val")
                            
                            if fy and val is not None:
                                key = (fy, fp, end_date)
                                if key not in periods:
                                    periods[key] = {"fiscal_year": fy, "fiscal_period": fp, "end_date": end_date}
                                if label not in periods[key]:  # Don't overwrite
                                    periods[key][label] = val
                        break  # Found a match, don't try other tags
            
            # Sort by date descending and limit
            sorted_periods = sorted(
                periods.values(),
                key=lambda x: (x.get("fiscal_year", 0), x.get("end_date", "")),
                reverse=True
            )[:num_periods]
            
            return sorted_periods
        
        # Build statements
        income_data = extract_values(income_statement_items, form_type)
        balance_data = extract_values(balance_sheet_items, form_type)
        cashflow_data = extract_values(cash_flow_items, form_type)
        
        def to_statements(data: List[Dict], stmt_type: str) -> List[FinancialStatement]:
            statements = []
            for period in data:
                fy = period.pop("fiscal_year", 0)
                fp = period.pop("fiscal_period", "FY")
                end_date = period.pop("end_date", "")
                
                statements.append(FinancialStatement(
                    statement_type=stmt_type,
                    period_end=end_date,
                    period_type="annual" if fp == "FY" else "quarterly",
                    fiscal_year=fy,
                    fiscal_quarter=int(fp[1]) if fp.startswith("Q") else None,
                    currency="USD",
                    data=period
                ))
            return statements
        
        return {
            "income_statement": to_statements(income_data, "income"),
            "balance_sheet": to_statements(balance_data, "balance"),
            "cash_flow": to_statements(cashflow_data, "cashflow")
        }


# Global SEC client
sec_client = SECClient()


def format_number(value: Any) -> str:
    """Format a number for display."""
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        if abs(value) >= 1_000_000_000:
            return f"${value/1_000_000_000:.2f}B"
        elif abs(value) >= 1_000_000:
            return f"${value/1_000_000:.2f}M"
        elif abs(value) >= 1_000:
            return f"${value/1_000:.2f}K"
        else:
            return f"${value:,.2f}"
    return str(value)


def format_statement_table(statement: FinancialStatement, title: str) -> str:
    """Format a financial statement as a text table."""
    lines = [
        f"\n{'='*60}",
        f"  {title}",
        f"  Period: {statement.period_end} ({statement.period_type.upper()})",
        f"  Fiscal Year: {statement.fiscal_year}" + (f" Q{statement.fiscal_quarter}" if statement.fiscal_quarter else ""),
        f"{'='*60}",
    ]
    
    max_label_len = max(len(k) for k in statement.data.keys()) if statement.data else 20
    
    for label, value in statement.data.items():
        formatted = format_number(value)
        lines.append(f"  {label:<{max_label_len}} : {formatted:>15}")
    
    lines.append(f"{'='*60}\n")
    return "\n".join(lines)


def format_statements_markdown(
    statements: Dict[str, List[FinancialStatement]],
    ticker: str,
    form_type: str
) -> str:
    """
    Format all three financial statements as Markdown tables.
    
    Creates side-by-side period comparison tables.
    """
    output = []
    
    # Header
    output.append(f"# {ticker.upper()} Financial Statements")
    output.append(f"*Source: SEC EDGAR {form_type} Filings*\n")
    
    # === INCOME STATEMENT ===
    income_stmts = statements.get("income_statement", [])
    if income_stmts:
        output.append("## Income Statement\n")
        
        # Build header row with periods
        periods = [f"FY{s.fiscal_year}" + (f" Q{s.fiscal_quarter}" if s.fiscal_quarter else "") for s in income_stmts]
        header = "| Line Item | " + " | ".join(periods) + " |"
        separator = "|:---|" + "|".join([":---:" for _ in periods]) + "|"
        
        output.append(header)
        output.append(separator)
        
        # Get all unique line items across periods
        all_items = []
        for stmt in income_stmts:
            for key in stmt.data.keys():
                if key not in all_items:
                    all_items.append(key)
        
        # Build rows
        for item in all_items:
            row_values = []
            for stmt in income_stmts:
                val = stmt.data.get(item)
                row_values.append(format_number(val) if val is not None else "-")
            output.append(f"| **{item}** | " + " | ".join(row_values) + " |")
        
        output.append("")
    
    # === BALANCE SHEET ===
    balance_stmts = statements.get("balance_sheet", [])
    if balance_stmts:
        output.append("## Balance Sheet\n")
        
        periods = [f"FY{s.fiscal_year}" + (f" Q{s.fiscal_quarter}" if s.fiscal_quarter else "") for s in balance_stmts]
        header = "| Line Item | " + " | ".join(periods) + " |"
        separator = "|:---|" + "|".join([":---:" for _ in periods]) + "|"
        
        output.append(header)
        output.append(separator)
        
        all_items = []
        for stmt in balance_stmts:
            for key in stmt.data.keys():
                if key not in all_items:
                    all_items.append(key)
        
        # Group by category
        asset_items = [i for i in all_items if any(x in i for x in ["Cash", "Receivable", "Inventory", "Asset", "Investment", "Property", "Goodwill", "Intangible", "Prepaid"])]
        liability_items = [i for i in all_items if any(x in i for x in ["Payable", "Debt", "Liabilit", "Deferred", "Accrued"])]
        equity_items = [i for i in all_items if any(x in i for x in ["Equity", "Stock", "Retained", "AOCI", "Treasury"])]
        
        # Assets section
        if asset_items:
            output.append("| **ASSETS** | " + " | ".join(["" for _ in periods]) + " |")
            for item in asset_items:
                row_values = []
                for stmt in balance_stmts:
                    val = stmt.data.get(item)
                    row_values.append(format_number(val) if val is not None else "-")
                output.append(f"| {item} | " + " | ".join(row_values) + " |")
        
        # Liabilities section
        if liability_items:
            output.append("| **LIABILITIES** | " + " | ".join(["" for _ in periods]) + " |")
            for item in liability_items:
                row_values = []
                for stmt in balance_stmts:
                    val = stmt.data.get(item)
                    row_values.append(format_number(val) if val is not None else "-")
                output.append(f"| {item} | " + " | ".join(row_values) + " |")
        
        # Equity section
        if equity_items:
            output.append("| **EQUITY** | " + " | ".join(["" for _ in periods]) + " |")
            for item in equity_items:
                row_values = []
                for stmt in balance_stmts:
                    val = stmt.data.get(item)
                    row_values.append(format_number(val) if val is not None else "-")
                output.append(f"| {item} | " + " | ".join(row_values) + " |")
        
        output.append("")
    
    # === CASH FLOW STATEMENT ===
    cashflow_stmts = statements.get("cash_flow", [])
    if cashflow_stmts:
        output.append("## Cash Flow Statement\n")
        
        periods = [f"FY{s.fiscal_year}" + (f" Q{s.fiscal_quarter}" if s.fiscal_quarter else "") for s in cashflow_stmts]
        header = "| Line Item | " + " | ".join(periods) + " |"
        separator = "|:---|" + "|".join([":---:" for _ in periods]) + "|"
        
        output.append(header)
        output.append(separator)
        
        all_items = []
        for stmt in cashflow_stmts:
            for key in stmt.data.keys():
                if key not in all_items:
                    all_items.append(key)
        
        # Group by activity type
        operating_items = [i for i in all_items if any(x in i for x in ["NetIncome", "Depreciation", "StockBased", "Deferred", "Change", "Operating", "Other"]) and "Investing" not in i and "Financing" not in i]
        investing_items = [i for i in all_items if any(x in i for x in ["Capital", "Acquisition", "Investment", "Investing"])]
        financing_items = [i for i in all_items if any(x in i for x in ["Debt", "Stock", "Dividend", "Financing"])]
        summary_items = [i for i in all_items if any(x in i for x in ["NetChange", "Beginning", "Ending"])]
        
        # Operating
        if operating_items:
            output.append("| **OPERATING ACTIVITIES** | " + " | ".join(["" for _ in periods]) + " |")
            for item in operating_items:
                row_values = []
                for stmt in cashflow_stmts:
                    val = stmt.data.get(item)
                    row_values.append(format_number(val) if val is not None else "-")
                output.append(f"| {item} | " + " | ".join(row_values) + " |")
        
        # Investing
        if investing_items:
            output.append("| **INVESTING ACTIVITIES** | " + " | ".join(["" for _ in periods]) + " |")
            for item in investing_items:
                row_values = []
                for stmt in cashflow_stmts:
                    val = stmt.data.get(item)
                    row_values.append(format_number(val) if val is not None else "-")
                output.append(f"| {item} | " + " | ".join(row_values) + " |")
        
        # Financing
        if financing_items:
            output.append("| **FINANCING ACTIVITIES** | " + " | ".join(["" for _ in periods]) + " |")
            for item in financing_items:
                row_values = []
                for stmt in cashflow_stmts:
                    val = stmt.data.get(item)
                    row_values.append(format_number(val) if val is not None else "-")
                output.append(f"| {item} | " + " | ".join(row_values) + " |")
        
        # Summary
        if summary_items:
            output.append("| **CASH SUMMARY** | " + " | ".join(["" for _ in periods]) + " |")
            for item in summary_items:
                row_values = []
                for stmt in cashflow_stmts:
                    val = stmt.data.get(item)
                    row_values.append(format_number(val) if val is not None else "-")
                output.append(f"| {item} | " + " | ".join(row_values) + " |")
        
        output.append("")
    
    return "\n".join(output)


# MCP Tool Handlers

async def handle_get_company_info(ticker: str) -> str:
    """Get basic company information from SEC."""
    cik = await sec_client.get_cik(ticker)
    if not cik:
        return f"Could not find CIK for ticker: {ticker}"
    
    filings = await sec_client.get_company_filings(cik, "10-K", limit=1)
    if not filings:
        return f"No filings found for {ticker}"
    
    filing = filings[0]
    return json.dumps({
        "ticker": ticker.upper(),
        "cik": cik,
        "company_name": filing.company_name,
        "fiscal_year_end": filing.fiscal_year_end,
        "latest_10k_date": filing.filing_date
    }, indent=2)


async def handle_get_filings(
    ticker: str,
    form_type: str = "10-K",
    limit: int = 5
) -> str:
    """Get list of SEC filings for a company."""
    cik = await sec_client.get_cik(ticker)
    if not cik:
        return f"Could not find CIK for ticker: {ticker}"
    
    filings = await sec_client.get_company_filings(cik, form_type, limit)
    
    if not filings:
        return f"No {form_type} filings found for {ticker}"
    
    result = [f"\n{form_type} Filings for {ticker.upper()}\n" + "="*40]
    for f in filings:
        result.append(f"  {f.filing_date}: {f.accession_number}")
    
    return "\n".join(result)


async def handle_get_financial_statements(
    ticker: str,
    form_type: str = "10-K",
    num_periods: int = 4,
    output_format: str = "markdown"
) -> str:
    """
    Get the three main financial statements for a company.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        form_type: "10-K" for annual or "10-Q" for quarterly
        num_periods: Number of periods to retrieve
        output_format: "markdown" or "text"
    
    Returns:
        Formatted financial statements (Income, Balance Sheet, Cash Flow)
    """
    cik = await sec_client.get_cik(ticker)
    if not cik:
        return f"Could not find CIK for ticker: {ticker}"
    
    facts = await sec_client.get_company_facts(cik)
    if not facts:
        return f"Could not retrieve company facts for {ticker}"
    
    statements = sec_client.extract_financial_statements(facts, form_type, num_periods)
    
    # Use markdown format by default
    if output_format == "markdown":
        return format_statements_markdown(statements, ticker, form_type)
    
    # Legacy text format
    output = [
        f"\n{'#'*70}",
        f"  FINANCIAL STATEMENTS: {ticker.upper()}",
        f"  Form: {form_type} | Periods: {num_periods}",
        f"{'#'*70}"
    ]
    
    # Income Statements
    output.append("\n" + "="*70)
    output.append("  INCOME STATEMENTS")
    output.append("="*70)
    for stmt in statements.get("income_statement", []):
        output.append(format_statement_table(stmt, f"Income Statement - {stmt.fiscal_year}"))
    
    # Balance Sheets
    output.append("\n" + "="*70)
    output.append("  BALANCE SHEETS")
    output.append("="*70)
    for stmt in statements.get("balance_sheet", []):
        output.append(format_statement_table(stmt, f"Balance Sheet - {stmt.fiscal_year}"))
    
    # Cash Flow Statements
    output.append("\n" + "="*70)
    output.append("  CASH FLOW STATEMENTS")
    output.append("="*70)
    for stmt in statements.get("cash_flow", []):
        output.append(format_statement_table(stmt, f"Cash Flow Statement - {stmt.fiscal_year}"))
    
    return "\n".join(output)


async def handle_get_income_statement(ticker: str, form_type: str = "10-K", num_periods: int = 4) -> str:
    """Get income statement only in Markdown format."""
    cik = await sec_client.get_cik(ticker)
    if not cik:
        return f"Could not find CIK for ticker: {ticker}"
    
    facts = await sec_client.get_company_facts(cik)
    statements = sec_client.extract_financial_statements(facts, form_type, num_periods)
    income_stmts = statements.get("income_statement", [])
    
    if not income_stmts:
        return f"No income statement data found for {ticker}"
    
    output = [f"# {ticker.upper()} Income Statement", f"*Source: SEC EDGAR {form_type}*\n"]
    
    periods = [f"FY{s.fiscal_year}" + (f" Q{s.fiscal_quarter}" if s.fiscal_quarter else "") for s in income_stmts]
    output.append("| Line Item | " + " | ".join(periods) + " |")
    output.append("|:---|" + "|".join([":---:" for _ in periods]) + "|")
    
    all_items = []
    for stmt in income_stmts:
        for key in stmt.data.keys():
            if key not in all_items:
                all_items.append(key)
    
    for item in all_items:
        row_values = []
        for stmt in income_stmts:
            val = stmt.data.get(item)
            row_values.append(format_number(val) if val is not None else "-")
        output.append(f"| **{item}** | " + " | ".join(row_values) + " |")
    
    return "\n".join(output)


async def handle_get_balance_sheet(ticker: str, form_type: str = "10-K", num_periods: int = 4) -> str:
    """Get balance sheet only in Markdown format."""
    cik = await sec_client.get_cik(ticker)
    if not cik:
        return f"Could not find CIK for ticker: {ticker}"
    
    facts = await sec_client.get_company_facts(cik)
    statements = sec_client.extract_financial_statements(facts, form_type, num_periods)
    balance_stmts = statements.get("balance_sheet", [])
    
    if not balance_stmts:
        return f"No balance sheet data found for {ticker}"
    
    output = [f"# {ticker.upper()} Balance Sheet", f"*Source: SEC EDGAR {form_type}*\n"]
    
    periods = [f"FY{s.fiscal_year}" + (f" Q{s.fiscal_quarter}" if s.fiscal_quarter else "") for s in balance_stmts]
    output.append("| Line Item | " + " | ".join(periods) + " |")
    output.append("|:---|" + "|".join([":---:" for _ in periods]) + "|")
    
    all_items = []
    for stmt in balance_stmts:
        for key in stmt.data.keys():
            if key not in all_items:
                all_items.append(key)
    
    for item in all_items:
        row_values = []
        for stmt in balance_stmts:
            val = stmt.data.get(item)
            row_values.append(format_number(val) if val is not None else "-")
        output.append(f"| **{item}** | " + " | ".join(row_values) + " |")
    
    return "\n".join(output)


async def handle_get_cash_flow(ticker: str, form_type: str = "10-K", num_periods: int = 4) -> str:
    """Get cash flow statement only in Markdown format."""
    cik = await sec_client.get_cik(ticker)
    if not cik:
        return f"Could not find CIK for ticker: {ticker}"
    
    facts = await sec_client.get_company_facts(cik)
    statements = sec_client.extract_financial_statements(facts, form_type, num_periods)
    cashflow_stmts = statements.get("cash_flow", [])
    
    if not cashflow_stmts:
        return f"No cash flow data found for {ticker}"
    
    output = [f"# {ticker.upper()} Cash Flow Statement", f"*Source: SEC EDGAR {form_type}*\n"]
    
    periods = [f"FY{s.fiscal_year}" + (f" Q{s.fiscal_quarter}" if s.fiscal_quarter else "") for s in cashflow_stmts]
    output.append("| Line Item | " + " | ".join(periods) + " |")
    output.append("|:---|" + "|".join([":---:" for _ in periods]) + "|")
    
    all_items = []
    for stmt in cashflow_stmts:
        for key in stmt.data.keys():
            if key not in all_items:
                all_items.append(key)
    
    for item in all_items:
        row_values = []
        for stmt in cashflow_stmts:
            val = stmt.data.get(item)
            row_values.append(format_number(val) if val is not None else "-")
        output.append(f"| **{item}** | " + " | ".join(row_values) + " |")
    
    return "\n".join(output)


async def handle_compare_companies(tickers: List[str], metric: str = "NetIncome") -> str:
    """Compare a metric across multiple companies."""
    results = []
    
    for ticker in tickers:
        cik = await sec_client.get_cik(ticker)
        if not cik:
            results.append({"ticker": ticker, "error": "CIK not found"})
            continue
        
        facts = await sec_client.get_company_facts(cik)
        statements = sec_client.extract_financial_statements(facts, "10-K", 1)
        
        income = statements.get("income_statement", [])
        if income:
            value = income[0].data.get(metric, "N/A")
            results.append({
                "ticker": ticker,
                "fiscal_year": income[0].fiscal_year,
                metric: format_number(value) if isinstance(value, (int, float)) else value
            })
    
    output = [f"\n  Comparison: {metric}\n" + "="*50]
    for r in results:
        if "error" in r:
            output.append(f"  {r['ticker']}: {r['error']}")
        else:
            output.append(f"  {r['ticker']}: {r.get(metric, 'N/A')} (FY{r.get('fiscal_year', 'N/A')})")
    
    return "\n".join(output)


# MCP Server Definition

if MCP_AVAILABLE:
    server = Server("sec-edgar")
    
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available SEC tools."""
        return [
            Tool(
                name="sec_get_company_info",
                description="Get basic company information from SEC EDGAR (CIK, name, fiscal year end)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "Stock ticker symbol (e.g., AAPL, MSFT)"
                        }
                    },
                    "required": ["ticker"]
                }
            ),
            Tool(
                name="sec_get_filings",
                description="Get list of SEC filings (10-K, 10-Q) for a company",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string", "description": "Stock ticker symbol"},
                        "form_type": {"type": "string", "enum": ["10-K", "10-Q"], "default": "10-K"},
                        "limit": {"type": "integer", "default": 5}
                    },
                    "required": ["ticker"]
                }
            ),
            Tool(
                name="sec_get_financial_statements",
                description="Get all three main financial statements (Income Statement, Balance Sheet, Cash Flow) from 10-K or 10-Q filings",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string", "description": "Stock ticker symbol"},
                        "form_type": {"type": "string", "enum": ["10-K", "10-Q"], "default": "10-K"},
                        "num_periods": {"type": "integer", "default": 4, "description": "Number of periods to retrieve"}
                    },
                    "required": ["ticker"]
                }
            ),
            Tool(
                name="sec_get_income_statement",
                description="Get income statement (revenues, expenses, net income) from SEC filings",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "form_type": {"type": "string", "enum": ["10-K", "10-Q"], "default": "10-K"},
                        "num_periods": {"type": "integer", "default": 4}
                    },
                    "required": ["ticker"]
                }
            ),
            Tool(
                name="sec_get_balance_sheet",
                description="Get balance sheet (assets, liabilities, equity) from SEC filings",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "form_type": {"type": "string", "enum": ["10-K", "10-Q"], "default": "10-K"},
                        "num_periods": {"type": "integer", "default": 4}
                    },
                    "required": ["ticker"]
                }
            ),
            Tool(
                name="sec_get_cash_flow",
                description="Get cash flow statement (operating, investing, financing) from SEC filings",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "form_type": {"type": "string", "enum": ["10-K", "10-Q"], "default": "10-K"},
                        "num_periods": {"type": "integer", "default": 4}
                    },
                    "required": ["ticker"]
                }
            ),
            Tool(
                name="sec_compare_companies",
                description="Compare a financial metric across multiple companies",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "tickers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of ticker symbols"
                        },
                        "metric": {
                            "type": "string",
                            "description": "Metric to compare (e.g., NetIncome, Revenues, TotalAssets)",
                            "default": "NetIncome"
                        }
                    },
                    "required": ["tickers"]
                }
            )
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """Handle tool calls."""
        try:
            if name == "sec_get_company_info":
                result = await handle_get_company_info(arguments["ticker"])
            elif name == "sec_get_filings":
                result = await handle_get_filings(
                    arguments["ticker"],
                    arguments.get("form_type", "10-K"),
                    arguments.get("limit", 5)
                )
            elif name == "sec_get_financial_statements":
                result = await handle_get_financial_statements(
                    arguments["ticker"],
                    arguments.get("form_type", "10-K"),
                    arguments.get("num_periods", 4)
                )
            elif name == "sec_get_income_statement":
                result = await handle_get_income_statement(
                    arguments["ticker"],
                    arguments.get("form_type", "10-K"),
                    arguments.get("num_periods", 4)
                )
            elif name == "sec_get_balance_sheet":
                result = await handle_get_balance_sheet(
                    arguments["ticker"],
                    arguments.get("form_type", "10-K"),
                    arguments.get("num_periods", 4)
                )
            elif name == "sec_get_cash_flow":
                result = await handle_get_cash_flow(
                    arguments["ticker"],
                    arguments.get("form_type", "10-K"),
                    arguments.get("num_periods", 4)
                )
            elif name == "sec_compare_companies":
                result = await handle_compare_companies(
                    arguments["tickers"],
                    arguments.get("metric", "NetIncome")
                )
            else:
                result = f"Unknown tool: {name}"
            
            return [TextContent(type="text", text=result)]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    async def main():
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())


# CLI for testing
async def test_cli():
    """Test the SEC tools from command line."""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python server.py <command> <ticker> [options]")
        print("\nCommands:")
        print("  info <ticker>           - Get company info")
        print("  filings <ticker>        - List filings")
        print("  all <ticker>            - Get all financial statements")
        print("  income <ticker>         - Get income statement")
        print("  balance <ticker>        - Get balance sheet")
        print("  cashflow <ticker>       - Get cash flow statement")
        print("  compare <t1,t2,...>     - Compare companies")
        return
    
    command = sys.argv[1]
    ticker = sys.argv[2]
    
    if command == "info":
        print(await handle_get_company_info(ticker))
    elif command == "filings":
        form_type = sys.argv[3] if len(sys.argv) > 3 else "10-K"
        print(await handle_get_filings(ticker, form_type))
    elif command == "all":
        form_type = sys.argv[3] if len(sys.argv) > 3 else "10-K"
        print(await handle_get_financial_statements(ticker, form_type))
    elif command == "income":
        print(await handle_get_income_statement(ticker))
    elif command == "balance":
        print(await handle_get_balance_sheet(ticker))
    elif command == "cashflow":
        print(await handle_get_cash_flow(ticker))
    elif command == "compare":
        tickers = ticker.split(",")
        metric = sys.argv[3] if len(sys.argv) > 3 else "NetIncome"
        print(await handle_compare_companies(tickers, metric))


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] != "--mcp":
        # CLI testing mode
        asyncio.run(test_cli())
    elif MCP_AVAILABLE:
        # MCP server mode
        asyncio.run(main())
    else:
        print("MCP SDK not available. Install with: pip install mcp")
