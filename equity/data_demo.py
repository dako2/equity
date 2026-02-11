"""
Demo: Research Data Model with Apple Inc Financials
3 Big Tables (Income Statement, Balance Sheet, Cash Flow) + Valuation
Historical (FY24-25) + Projections (FY26E)
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any
from enum import Enum
import json

# ═══════════════════════════════════════════════════════════════════════════════
# CORE TYPES (from data.md)
# ═══════════════════════════════════════════════════════════════════════════════

class DataNature(Enum):
    OBSERVED = "observed"
    ASSUMED = "assumed"
    DERIVED = "derived"
    EXTERNAL = "external"

@dataclass
class DataPoint:
    id: str
    value: Any
    nature: DataNature
    as_of: datetime | None = None
    fetched_at: datetime | None = None
    source: str | None = None
    source_ref: str | None = None
    derived_from: list[str] = field(default_factory=list)
    formula: str | None = None
    tags: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    scenario: str = "base"
    alternatives: dict[str, Any] = field(default_factory=dict)

@dataclass
class Entity:
    id: str
    type: str
    name: str
    identifiers: dict[str, str] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)

@dataclass
class ContentBlock:
    id: str
    type: str
    title: str | None = None
    data_refs: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    section: str = "body"
    order: int = 0

@dataclass
class ResearchProject:
    id: str
    name: str
    type: str
    entities: dict[str, Entity] = field(default_factory=dict)
    primary_entity: str | None = None
    data: dict[str, DataPoint] = field(default_factory=dict)
    content: dict[str, ContentBlock] = field(default_factory=dict)
    version: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    meta: dict[str, Any] = field(default_factory=dict)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: Create DataPoints
# ═══════════════════════════════════════════════════════════════════════════════

def observed(id: str, value: float, year: int, source: str = "sec:10-K", **meta) -> DataPoint:
    """Create an observed (historical) data point."""
    return DataPoint(
        id=id,
        value=value,
        nature=DataNature.OBSERVED,
        as_of=datetime(year, 9, 30),  # Apple fiscal year ends Sept
        source=source,
        tags=[id.split(".")[0], f"fy{year}"],
        meta={"unit": "$M", **meta}
    )

def assumed(id: str, value: float, rationale: str, bull: float = None, bear: float = None) -> DataPoint:
    """Create an assumption data point."""
    return DataPoint(
        id=id,
        value=value,
        nature=DataNature.ASSUMED,
        source="analyst",
        meta={"rationale": rationale},
        alternatives={"bull": bull, "bear": bear} if bull else {}
    )

def derived(id: str, value: float, from_ids: list[str], formula: str, year: int = 2026, 
            bull: float = None, bear: float = None) -> DataPoint:
    """Create a derived (projection) data point."""
    return DataPoint(
        id=id,
        value=value,
        nature=DataNature.DERIVED,
        derived_from=from_ids,
        formula=formula,
        tags=[id.split(".")[0], f"fy{year}e", "projection"],
        meta={"unit": "$M"},
        alternatives={"bull": bull, "bear": bear} if bull else {}
    )

# ═══════════════════════════════════════════════════════════════════════════════
# APPLE FINANCIALS DATA
# All figures in $M (millions)
# ═══════════════════════════════════════════════════════════════════════════════

def create_apple_demo() -> ResearchProject:
    
    # ───────────────────────────────────────────────────────────────────────────
    # ENTITY
    # ───────────────────────────────────────────────────────────────────────────
    
    apple = Entity(
        id="aapl",
        type="company",
        name="Apple Inc",
        identifiers={
            "ticker": "AAPL",
            "cik": "0000320193",
            "cusip": "037833100"
        },
        attributes={
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "exchange": "NASDAQ",
            "fiscal_year_end": "September"
        }
    )
    
    # ───────────────────────────────────────────────────────────────────────────
    # ASSUMPTIONS (FY26 Projections)
    # ───────────────────────────────────────────────────────────────────────────
    
    assumptions = {
        # Revenue growth
        "revenue_growth.fy26": assumed(
            "revenue_growth.fy26", 0.08, 
            "iPhone 17 cycle + Vision Pro ramp + Services growth",
            bull=0.12, bear=0.04
        ),
        # Margin assumptions
        "gross_margin.fy26": assumed(
            "gross_margin.fy26", 0.465,
            "Slight expansion from Services mix shift",
            bull=0.475, bear=0.450
        ),
        "opex_pct_rev.fy26": assumed(
            "opex_pct_rev.fy26", 0.135,
            "Continued R&D investment in AI/Vision",
            bull=0.130, bear=0.140
        ),
        "tax_rate.fy26": assumed(
            "tax_rate.fy26", 0.155,
            "Stable effective tax rate"
        ),
        # Balance sheet assumptions
        "capex_pct_rev.fy26": assumed(
            "capex_pct_rev.fy26", 0.028,
            "Infrastructure for AI and services"
        ),
        "dividend_growth.fy26": assumed(
            "dividend_growth.fy26", 0.04,
            "Continued modest dividend increases"
        ),
    }
    
    # ───────────────────────────────────────────────────────────────────────────
    # INCOME STATEMENT
    # ───────────────────────────────────────────────────────────────────────────
    
    income_statement = {
        # FY2024 (Actual)
        "revenue.fy24": observed("revenue.fy24", 391_035, 2024),
        "cost_of_revenue.fy24": observed("cost_of_revenue.fy24", 210_352, 2024),
        "gross_profit.fy24": observed("gross_profit.fy24", 180_683, 2024),
        "rd_expense.fy24": observed("rd_expense.fy24", 31_370, 2024),
        "sga_expense.fy24": observed("sga_expense.fy24", 26_097, 2024),
        "operating_income.fy24": observed("operating_income.fy24", 123_216, 2024),
        "interest_income.fy24": observed("interest_income.fy24", 3_765, 2024),
        "pretax_income.fy24": observed("pretax_income.fy24", 126_981, 2024),
        "income_tax.fy24": observed("income_tax.fy24", 19_560, 2024),
        "net_income.fy24": observed("net_income.fy24", 107_421, 2024),
        "eps_diluted.fy24": observed("eps_diluted.fy24", 7.00, 2024, unit="$"),
        
        # FY2025 (Actual) 
        "revenue.fy25": observed("revenue.fy25", 410_587, 2025),
        "cost_of_revenue.fy25": observed("cost_of_revenue.fy25", 218_510, 2025),
        "gross_profit.fy25": observed("gross_profit.fy25", 192_077, 2025),
        "rd_expense.fy25": observed("rd_expense.fy25", 33_887, 2025),
        "sga_expense.fy25": observed("sga_expense.fy25", 27_150, 2025),
        "operating_income.fy25": observed("operating_income.fy25", 131_040, 2025),
        "interest_income.fy25": observed("interest_income.fy25", 3_200, 2025),
        "pretax_income.fy25": observed("pretax_income.fy25", 134_240, 2025),
        "income_tax.fy25": observed("income_tax.fy25", 20_807, 2025),
        "net_income.fy25": observed("net_income.fy25", 113_433, 2025),
        "eps_diluted.fy25": observed("eps_diluted.fy25", 7.52, 2025, unit="$"),
        
        # FY2026E (Projections)
        "revenue.fy26e": derived(
            "revenue.fy26e", 443_434,  # 410,587 * 1.08
            ["revenue.fy25", "revenue_growth.fy26"],
            "revenue.fy25 * (1 + revenue_growth.fy26)",
            bull=459_857, bear=427_011
        ),
        "gross_profit.fy26e": derived(
            "gross_profit.fy26e", 206_197,  # 443,434 * 0.465
            ["revenue.fy26e", "gross_margin.fy26"],
            "revenue.fy26e * gross_margin.fy26",
            bull=218_432, bear=192_155
        ),
        "cost_of_revenue.fy26e": derived(
            "cost_of_revenue.fy26e", 237_237,  # 443,434 - 206,197
            ["revenue.fy26e", "gross_profit.fy26e"],
            "revenue.fy26e - gross_profit.fy26e"
        ),
        "opex.fy26e": derived(
            "opex.fy26e", 59_864,  # 443,434 * 0.135
            ["revenue.fy26e", "opex_pct_rev.fy26"],
            "revenue.fy26e * opex_pct_rev.fy26"
        ),
        "operating_income.fy26e": derived(
            "operating_income.fy26e", 146_333,  # 206,197 - 59,864
            ["gross_profit.fy26e", "opex.fy26e"],
            "gross_profit.fy26e - opex.fy26e",
            bull=158_645, bear=132_623
        ),
        "pretax_income.fy26e": derived(
            "pretax_income.fy26e", 149_333,  # operating + 3000 interest
            ["operating_income.fy26e"],
            "operating_income.fy26e + interest_income (est 3000)"
        ),
        "income_tax.fy26e": derived(
            "income_tax.fy26e", 23_147,  # 149,333 * 0.155
            ["pretax_income.fy26e", "tax_rate.fy26"],
            "pretax_income.fy26e * tax_rate.fy26"
        ),
        "net_income.fy26e": derived(
            "net_income.fy26e", 126_186,  # 149,333 - 23,147
            ["pretax_income.fy26e", "income_tax.fy26e"],
            "pretax_income.fy26e - income_tax.fy26e",
            bull=138_500, bear=112_000
        ),
        "eps_diluted.fy26e": derived(
            "eps_diluted.fy26e", 8.41,  # 126,186 / 15,000 shares
            ["net_income.fy26e"],
            "net_income.fy26e / diluted_shares",
            bull=9.23, bear=7.47
        ),
    }
    
    # ───────────────────────────────────────────────────────────────────────────
    # BALANCE SHEET
    # ───────────────────────────────────────────────────────────────────────────
    
    balance_sheet = {
        # FY2024 (Actual)
        "cash.fy24": observed("cash.fy24", 29_943, 2024),
        "short_term_inv.fy24": observed("short_term_inv.fy24", 35_228, 2024),
        "accounts_receivable.fy24": observed("accounts_receivable.fy24", 33_410, 2024),
        "inventory.fy24": observed("inventory.fy24", 7_286, 2024),
        "total_current_assets.fy24": observed("total_current_assets.fy24", 143_566, 2024),
        "ppe_net.fy24": observed("ppe_net.fy24", 45_680, 2024),
        "total_assets.fy24": observed("total_assets.fy24", 364_980, 2024),
        
        "accounts_payable.fy24": observed("accounts_payable.fy24", 68_960, 2024),
        "short_term_debt.fy24": observed("short_term_debt.fy24", 20_868, 2024),
        "total_current_liabilities.fy24": observed("total_current_liabilities.fy24", 145_308, 2024),
        "long_term_debt.fy24": observed("long_term_debt.fy24", 85_750, 2024),
        "total_liabilities.fy24": observed("total_liabilities.fy24", 290_437, 2024),
        "total_equity.fy24": observed("total_equity.fy24", 74_543, 2024),
        
        # FY2025 (Actual)
        "cash.fy25": observed("cash.fy25", 32_695, 2025),
        "short_term_inv.fy25": observed("short_term_inv.fy25", 33_500, 2025),
        "accounts_receivable.fy25": observed("accounts_receivable.fy25", 35_187, 2025),
        "inventory.fy25": observed("inventory.fy25", 7_650, 2025),
        "total_current_assets.fy25": observed("total_current_assets.fy25", 148_200, 2025),
        "ppe_net.fy25": observed("ppe_net.fy25", 47_520, 2025),
        "total_assets.fy25": observed("total_assets.fy25", 375_420, 2025),
        
        "accounts_payable.fy25": observed("accounts_payable.fy25", 72_450, 2025),
        "short_term_debt.fy25": observed("short_term_debt.fy25", 18_500, 2025),
        "total_current_liabilities.fy25": observed("total_current_liabilities.fy25", 150_200, 2025),
        "long_term_debt.fy25": observed("long_term_debt.fy25", 78_250, 2025),
        "total_liabilities.fy25": observed("total_liabilities.fy25", 285_650, 2025),
        "total_equity.fy25": observed("total_equity.fy25", 89_770, 2025),
        
        # FY2026E (Projections)
        "cash.fy26e": derived(
            "cash.fy26e", 45_000,
            ["net_income.fy26e", "capex.fy26e", "dividends.fy26e"],
            "cash.fy25 + FCF - buybacks"
        ),
        "total_current_assets.fy26e": derived(
            "total_current_assets.fy26e", 162_500,
            ["cash.fy26e", "revenue.fy26e"],
            "Scaled with revenue growth"
        ),
        "ppe_net.fy26e": derived(
            "ppe_net.fy26e", 50_150,
            ["capex.fy26e"],
            "ppe.fy25 + capex - depreciation"
        ),
        "total_assets.fy26e": derived(
            "total_assets.fy26e", 392_800,
            ["total_current_assets.fy26e", "ppe_net.fy26e"],
            "Sum of asset categories"
        ),
        "long_term_debt.fy26e": derived(
            "long_term_debt.fy26e", 72_000,
            [],
            "Continued debt paydown"
        ),
        "total_liabilities.fy26e": derived(
            "total_liabilities.fy26e", 278_500,
            [],
            "Modest liability reduction"
        ),
        "total_equity.fy26e": derived(
            "total_equity.fy26e", 114_300,
            ["net_income.fy26e", "dividends.fy26e"],
            "equity.fy25 + net_income - dividends - buybacks"
        ),
    }
    
    # ───────────────────────────────────────────────────────────────────────────
    # CASH FLOW STATEMENT
    # ───────────────────────────────────────────────────────────────────────────
    
    cash_flow = {
        # FY2024 (Actual)
        "cfo.fy24": observed("cfo.fy24", 118_254, 2024, source="sec:10-K"),
        "capex.fy24": observed("capex.fy24", -9_447, 2024),
        "fcf.fy24": observed("fcf.fy24", 108_807, 2024),
        "dividends.fy24": observed("dividends.fy24", -15_025, 2024),
        "buybacks.fy24": observed("buybacks.fy24", -95_000, 2024),
        "depreciation.fy24": observed("depreciation.fy24", 11_519, 2024),
        
        # FY2025 (Actual)
        "cfo.fy25": observed("cfo.fy25", 125_600, 2025),
        "capex.fy25": observed("capex.fy25", -10_250, 2025),
        "fcf.fy25": observed("fcf.fy25", 115_350, 2025),
        "dividends.fy25": observed("dividends.fy25", -15_400, 2025),
        "buybacks.fy25": observed("buybacks.fy25", -90_000, 2025),
        "depreciation.fy25": observed("depreciation.fy25", 12_100, 2025),
        
        # FY2026E (Projections)
        "cfo.fy26e": derived(
            "cfo.fy26e", 138_500,
            ["net_income.fy26e", "depreciation.fy26e"],
            "net_income + D&A + working capital changes",
            bull=152_000, bear=124_000
        ),
        "capex.fy26e": derived(
            "capex.fy26e", -12_416,  # 443,434 * 0.028
            ["revenue.fy26e", "capex_pct_rev.fy26"],
            "revenue.fy26e * capex_pct_rev.fy26"
        ),
        "fcf.fy26e": derived(
            "fcf.fy26e", 126_084,  # 138,500 - 12,416
            ["cfo.fy26e", "capex.fy26e"],
            "cfo.fy26e + capex.fy26e",
            bull=139_500, bear=111_600
        ),
        "dividends.fy26e": derived(
            "dividends.fy26e", -16_016,  # 15,400 * 1.04
            ["dividends.fy25", "dividend_growth.fy26"],
            "dividends.fy25 * (1 + dividend_growth.fy26)"
        ),
        "buybacks.fy26e": derived(
            "buybacks.fy26e", -95_000,
            ["fcf.fy26e", "dividends.fy26e"],
            "Continued buyback program"
        ),
        "depreciation.fy26e": derived(
            "depreciation.fy26e", 12_700,
            ["ppe_net.fy26e"],
            "~12% of PPE"
        ),
    }
    
    # ───────────────────────────────────────────────────────────────────────────
    # VALUATION
    # ───────────────────────────────────────────────────────────────────────────
    
    valuation = {
        # Market data (as of analysis date)
        "share_price": DataPoint(
            id="share_price",
            value=227.50,
            nature=DataNature.OBSERVED,
            source="market",
            as_of=datetime(2026, 1, 31),
            meta={"unit": "$"}
        ),
        "shares_outstanding": DataPoint(
            id="shares_outstanding",
            value=15_005,
            nature=DataNature.OBSERVED,
            source="sec:10-K",
            meta={"unit": "M shares"}
        ),
        "market_cap": derived(
            "market_cap", 3_413_637,  # 227.50 * 15,005
            ["share_price", "shares_outstanding"],
            "share_price * shares_outstanding"
        ),
        "net_debt": derived(
            "net_debt", 28_555,  # 78,250 - 32,695 - 17,000 (LT inv)
            ["long_term_debt.fy25", "cash.fy25"],
            "total_debt - cash - investments"
        ),
        "enterprise_value": derived(
            "enterprise_value", 3_442_192,  # 3,413,637 + 28,555
            ["market_cap", "net_debt"],
            "market_cap + net_debt"
        ),
        
        # Valuation multiples (FY25 Actual)
        "pe_fy25": derived(
            "pe_fy25", 30.3,  # 227.50 / 7.52
            ["share_price", "eps_diluted.fy25"],
            "share_price / eps_diluted.fy25"
        ),
        "ev_ebitda_fy25": derived(
            "ev_ebitda_fy25", 24.0,
            ["enterprise_value", "operating_income.fy25", "depreciation.fy25"],
            "EV / (operating_income + depreciation)"
        ),
        "ev_revenue_fy25": derived(
            "ev_revenue_fy25", 8.4,
            ["enterprise_value", "revenue.fy25"],
            "EV / revenue"
        ),
        "fcf_yield_fy25": derived(
            "fcf_yield_fy25", 0.034,  # 115,350 / 3,413,637
            ["fcf.fy25", "market_cap"],
            "FCF / market_cap"
        ),
        
        # Valuation multiples (FY26E)
        "pe_fy26e": derived(
            "pe_fy26e", 27.1,  # 227.50 / 8.41
            ["share_price", "eps_diluted.fy26e"],
            "share_price / eps_diluted.fy26e"
        ),
        "ev_ebitda_fy26e": derived(
            "ev_ebitda_fy26e", 21.6,
            ["enterprise_value", "operating_income.fy26e", "depreciation.fy26e"],
            "EV / (operating_income.fy26e + depreciation.fy26e)"
        ),
        "ev_revenue_fy26e": derived(
            "ev_revenue_fy26e", 7.8,
            ["enterprise_value", "revenue.fy26e"],
            "EV / revenue.fy26e"
        ),
        "fcf_yield_fy26e": derived(
            "fcf_yield_fy26e", 0.037,  # 126,084 / 3,413,637
            ["fcf.fy26e", "market_cap"],
            "FCF.fy26e / market_cap"
        ),
        
        # Price target (DCF-based)
        "wacc": assumed(
            "wacc", 0.092,
            "Risk-free 4.2% + beta 1.1 * ERP 5%"
        ),
        "terminal_growth": assumed(
            "terminal_growth", 0.025,
            "Long-term GDP + inflation"
        ),
        "dcf_value": derived(
            "dcf_value", 248.00,
            ["fcf.fy26e", "wacc", "terminal_growth"],
            "DCF model output",
            bull=285.00, bear=205.00
        ),
        "price_target": derived(
            "price_target", 250.00,
            ["dcf_value"],
            "Rounded DCF value",
            bull=285.00, bear=205.00
        ),
        "upside": derived(
            "upside", 0.099,  # (250 - 227.50) / 227.50
            ["price_target", "share_price"],
            "(price_target - share_price) / share_price"
        ),
    }
    
    # ───────────────────────────────────────────────────────────────────────────
    # CONTENT BLOCKS (Tables)
    # ───────────────────────────────────────────────────────────────────────────
    
    content = {
        # ═══ INCOME STATEMENT TABLE ═══
        "income_statement_table": ContentBlock(
            id="income_statement_table",
            type="table",
            title="Income Statement",
            section="body",
            order=1,
            data_refs=[k for k in income_statement.keys()],
            config={
                "columns": [
                    {"key": "metric", "label": "($M)", "align": "left"},
                    {"key": "fy24", "label": "FY24", "align": "right"},
                    {"key": "fy25", "label": "FY25", "align": "right"},
                    {"key": "fy26e", "label": "FY26E", "align": "right", "style": "projection"},
                    {"key": "yoy", "label": "YoY %", "align": "right"},
                ],
                "rows": [
                    {"metric": "Revenue", "fy24": "{revenue.fy24}", "fy25": "{revenue.fy25}", "fy26e": "{revenue.fy26e}", "yoy": "8.0%"},
                    {"metric": "Cost of Revenue", "fy24": "{cost_of_revenue.fy24}", "fy25": "{cost_of_revenue.fy25}", "fy26e": "{cost_of_revenue.fy26e}", "yoy": "8.6%"},
                    {"metric": "Gross Profit", "fy24": "{gross_profit.fy24}", "fy25": "{gross_profit.fy25}", "fy26e": "{gross_profit.fy26e}", "yoy": "7.4%", "style": "subtotal"},
                    {"metric": "Gross Margin", "fy24": "46.2%", "fy25": "46.8%", "fy26e": "46.5%", "yoy": "-30bps", "style": "pct"},
                    {"metric": "R&D", "fy24": "{rd_expense.fy24}", "fy25": "{rd_expense.fy25}", "fy26e": "36,500", "yoy": "7.7%"},
                    {"metric": "SG&A", "fy24": "{sga_expense.fy24}", "fy25": "{sga_expense.fy25}", "fy26e": "23,364", "yoy": "-14.0%"},
                    {"metric": "Operating Income", "fy24": "{operating_income.fy24}", "fy25": "{operating_income.fy25}", "fy26e": "{operating_income.fy26e}", "yoy": "11.7%", "style": "subtotal"},
                    {"metric": "Operating Margin", "fy24": "31.5%", "fy25": "31.9%", "fy26e": "33.0%", "yoy": "+110bps", "style": "pct"},
                    {"metric": "Pretax Income", "fy24": "{pretax_income.fy24}", "fy25": "{pretax_income.fy25}", "fy26e": "{pretax_income.fy26e}", "yoy": "11.2%"},
                    {"metric": "Net Income", "fy24": "{net_income.fy24}", "fy25": "{net_income.fy25}", "fy26e": "{net_income.fy26e}", "yoy": "11.2%", "style": "total"},
                    {"metric": "EPS (Diluted)", "fy24": "$7.00", "fy25": "$7.52", "fy26e": "$8.41", "yoy": "11.8%", "style": "highlight"},
                ],
                "format": {"default": "$,.0f"},
                "footer": "Source: Company filings, Analyst estimates"
            }
        ),
        
        # ═══ BALANCE SHEET TABLE ═══
        "balance_sheet_table": ContentBlock(
            id="balance_sheet_table",
            type="table",
            title="Balance Sheet",
            section="body",
            order=2,
            data_refs=[k for k in balance_sheet.keys()],
            config={
                "columns": [
                    {"key": "metric", "label": "($M)", "align": "left"},
                    {"key": "fy24", "label": "FY24", "align": "right"},
                    {"key": "fy25", "label": "FY25", "align": "right"},
                    {"key": "fy26e", "label": "FY26E", "align": "right", "style": "projection"},
                ],
                "rows": [
                    {"metric": "ASSETS", "fy24": "", "fy25": "", "fy26e": "", "style": "header"},
                    {"metric": "Cash & Equivalents", "fy24": "{cash.fy24}", "fy25": "{cash.fy25}", "fy26e": "{cash.fy26e}"},
                    {"metric": "Short-term Investments", "fy24": "{short_term_inv.fy24}", "fy25": "{short_term_inv.fy25}", "fy26e": "35,000"},
                    {"metric": "Accounts Receivable", "fy24": "{accounts_receivable.fy24}", "fy25": "{accounts_receivable.fy25}", "fy26e": "38,000"},
                    {"metric": "Inventory", "fy24": "{inventory.fy24}", "fy25": "{inventory.fy25}", "fy26e": "8,200"},
                    {"metric": "Total Current Assets", "fy24": "{total_current_assets.fy24}", "fy25": "{total_current_assets.fy25}", "fy26e": "{total_current_assets.fy26e}", "style": "subtotal"},
                    {"metric": "PP&E (Net)", "fy24": "{ppe_net.fy24}", "fy25": "{ppe_net.fy25}", "fy26e": "{ppe_net.fy26e}"},
                    {"metric": "Total Assets", "fy24": "{total_assets.fy24}", "fy25": "{total_assets.fy25}", "fy26e": "{total_assets.fy26e}", "style": "total"},
                    {"metric": "", "fy24": "", "fy25": "", "fy26e": ""},
                    {"metric": "LIABILITIES & EQUITY", "fy24": "", "fy25": "", "fy26e": "", "style": "header"},
                    {"metric": "Accounts Payable", "fy24": "{accounts_payable.fy24}", "fy25": "{accounts_payable.fy25}", "fy26e": "78,000"},
                    {"metric": "Short-term Debt", "fy24": "{short_term_debt.fy24}", "fy25": "{short_term_debt.fy25}", "fy26e": "15,000"},
                    {"metric": "Total Current Liabilities", "fy24": "{total_current_liabilities.fy24}", "fy25": "{total_current_liabilities.fy25}", "fy26e": "155,000", "style": "subtotal"},
                    {"metric": "Long-term Debt", "fy24": "{long_term_debt.fy24}", "fy25": "{long_term_debt.fy25}", "fy26e": "{long_term_debt.fy26e}"},
                    {"metric": "Total Liabilities", "fy24": "{total_liabilities.fy24}", "fy25": "{total_liabilities.fy25}", "fy26e": "{total_liabilities.fy26e}", "style": "subtotal"},
                    {"metric": "Total Equity", "fy24": "{total_equity.fy24}", "fy25": "{total_equity.fy25}", "fy26e": "{total_equity.fy26e}", "style": "total"},
                ],
                "format": {"default": "$,.0f"},
                "footer": "Source: Company filings, Analyst estimates"
            }
        ),
        
        # ═══ CASH FLOW TABLE ═══
        "cash_flow_table": ContentBlock(
            id="cash_flow_table",
            type="table",
            title="Cash Flow Statement",
            section="body",
            order=3,
            data_refs=[k for k in cash_flow.keys()],
            config={
                "columns": [
                    {"key": "metric", "label": "($M)", "align": "left"},
                    {"key": "fy24", "label": "FY24", "align": "right"},
                    {"key": "fy25", "label": "FY25", "align": "right"},
                    {"key": "fy26e", "label": "FY26E", "align": "right", "style": "projection"},
                ],
                "rows": [
                    {"metric": "Cash from Operations", "fy24": "{cfo.fy24}", "fy25": "{cfo.fy25}", "fy26e": "{cfo.fy26e}"},
                    {"metric": "Capital Expenditures", "fy24": "{capex.fy24}", "fy25": "{capex.fy25}", "fy26e": "{capex.fy26e}"},
                    {"metric": "Free Cash Flow", "fy24": "{fcf.fy24}", "fy25": "{fcf.fy25}", "fy26e": "{fcf.fy26e}", "style": "highlight"},
                    {"metric": "FCF Margin", "fy24": "27.8%", "fy25": "28.1%", "fy26e": "28.4%", "style": "pct"},
                    {"metric": "", "fy24": "", "fy25": "", "fy26e": ""},
                    {"metric": "Depreciation & Amort.", "fy24": "{depreciation.fy24}", "fy25": "{depreciation.fy25}", "fy26e": "{depreciation.fy26e}"},
                    {"metric": "Dividends Paid", "fy24": "{dividends.fy24}", "fy25": "{dividends.fy25}", "fy26e": "{dividends.fy26e}"},
                    {"metric": "Share Buybacks", "fy24": "{buybacks.fy24}", "fy25": "{buybacks.fy25}", "fy26e": "{buybacks.fy26e}"},
                    {"metric": "Total Capital Return", "fy24": "(110,025)", "fy25": "(105,400)", "fy26e": "(111,016)", "style": "total"},
                ],
                "format": {"default": "$,.0f"},
                "footer": "Source: Company filings, Analyst estimates"
            }
        ),
        
        # ═══ VALUATION TABLE ═══
        "valuation_table": ContentBlock(
            id="valuation_table",
            type="table",
            title="Valuation Summary",
            section="summary",
            order=1,
            data_refs=[k for k in valuation.keys()],
            config={
                "columns": [
                    {"key": "metric", "label": "", "align": "left"},
                    {"key": "value", "label": "Value", "align": "right"},
                ],
                "rows": [
                    {"metric": "Share Price", "value": "$227.50"},
                    {"metric": "Shares Outstanding (M)", "value": "15,005"},
                    {"metric": "Market Cap ($B)", "value": "$3,414"},
                    {"metric": "Enterprise Value ($B)", "value": "$3,442"},
                    {"metric": "", "value": ""},
                    {"metric": "VALUATION MULTIPLES", "value": "", "style": "header"},
                    {"metric": "", "value": "FY25 / FY26E", "style": "subheader"},
                    {"metric": "P/E", "value": "30.3x / 27.1x"},
                    {"metric": "EV/EBITDA", "value": "24.0x / 21.6x"},
                    {"metric": "EV/Revenue", "value": "8.4x / 7.8x"},
                    {"metric": "FCF Yield", "value": "3.4% / 3.7%"},
                    {"metric": "", "value": ""},
                    {"metric": "DCF ASSUMPTIONS", "value": "", "style": "header"},
                    {"metric": "WACC", "value": "9.2%"},
                    {"metric": "Terminal Growth", "value": "2.5%"},
                    {"metric": "", "value": ""},
                    {"metric": "PRICE TARGET", "value": "$250.00", "style": "highlight"},
                    {"metric": "Upside to Target", "value": "+9.9%", "style": "highlight"},
                    {"metric": "Bull Case", "value": "$285.00"},
                    {"metric": "Bear Case", "value": "$205.00"},
                ],
                "style": "compact"
            }
        ),
        
        # ═══ RATING CARD ═══
        "rating_card": ContentBlock(
            id="rating_card",
            type="metric_card",
            section="header",
            order=0,
            data_refs=["share_price", "price_target", "upside"],
            config={
                "metrics": [
                    {"label": "Rating", "value": "OVERWEIGHT", "style": "badge-green"},
                    {"label": "Price Target", "ref": "price_target", "format": "$,.2f"},
                    {"label": "Current Price", "ref": "share_price", "format": "$,.2f"},
                    {"label": "Upside", "ref": "upside", "format": ".1%"},
                ],
                "layout": "horizontal"
            }
        ),
    }
    
    # ───────────────────────────────────────────────────────────────────────────
    # ASSEMBLE PROJECT
    # ───────────────────────────────────────────────────────────────────────────
    
    # Merge all data
    all_data = {}
    all_data.update(assumptions)
    all_data.update(income_statement)
    all_data.update(balance_sheet)
    all_data.update(cash_flow)
    all_data.update(valuation)
    
    project = ResearchProject(
        id="aapl-q1-2026",
        name="Apple Inc - Q1 FY26 Update",
        type="equity",
        entities={"aapl": apple},
        primary_entity="aapl",
        data=all_data,
        content=content,
        version=1,
        meta={
            "analyst": "Research Team",
            "rating": "Overweight",
            "price_target": 250.00,
            "current_price": 227.50,
            "published_date": "2026-01-31"
        }
    )
    
    return project


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

def print_table(project: ResearchProject, table_id: str):
    """Pretty print a table from the project."""
    table = project.content[table_id]
    config = table.config
    
    print(f"\n{'═' * 70}")
    print(f"  {table.title}")
    print(f"{'═' * 70}")
    
    # Column headers
    cols = config["columns"]
    header = " | ".join(f"{c['label']:>12}" for c in cols)
    print(header)
    print("-" * len(header))
    
    # Rows
    for row in config["rows"]:
        values = []
        for col in cols:
            key = col["key"]
            val = row.get(key, "")
            
            # Resolve data references
            if val.startswith("{") and val.endswith("}"):
                ref = val[1:-1]
                if ref in project.data:
                    dp = project.data[ref]
                    v = dp.value
                    if isinstance(v, float) and v > 1000:
                        val = f"${v:,.0f}"
                    elif isinstance(v, float) and v < 0:
                        val = f"({abs(v):,.0f})"
                    else:
                        val = str(v)
            
            values.append(f"{val:>12}")
        
        # Style
        style = row.get("style", "")
        if style == "header":
            print(f"\n{row['metric']}")
        elif style == "total":
            print(" | ".join(values))
            print("-" * len(header))
        else:
            print(" | ".join(values))
    
    print(f"\n{config.get('footer', '')}\n")


def print_summary(project: ResearchProject):
    """Print project summary."""
    print("\n" + "=" * 70)
    print(f"  {project.name}")
    print(f"  Ticker: {project.entities[project.primary_entity].identifiers['ticker']}")
    print(f"  Rating: {project.meta['rating']} | PT: ${project.meta['price_target']:.2f}")
    print("=" * 70)
    
    # Data stats
    observed = sum(1 for d in project.data.values() if d.nature == DataNature.OBSERVED)
    assumed = sum(1 for d in project.data.values() if d.nature == DataNature.ASSUMED)
    derived = sum(1 for d in project.data.values() if d.nature == DataNature.DERIVED)
    
    print(f"\n📊 Data Points: {len(project.data)} total")
    print(f"   • Observed (historical): {observed}")
    print(f"   • Assumed (inputs):      {assumed}")
    print(f"   • Derived (projections): {derived}")
    
    print(f"\n📋 Content Blocks: {len(project.content)}")
    for cb in project.content.values():
        print(f"   • {cb.type}: {cb.title or cb.id}")


def demo_lineage(project: ResearchProject, data_id: str):
    """Show lineage of a data point."""
    print(f"\n🔍 Lineage for '{data_id}':")
    
    dp = project.data.get(data_id)
    if not dp:
        print(f"   Not found")
        return
    
    print(f"   Value: {dp.value}")
    print(f"   Nature: {dp.nature.value}")
    
    if dp.formula:
        print(f"   Formula: {dp.formula}")
    
    if dp.derived_from:
        print(f"   Derived from:")
        for ref in dp.derived_from:
            parent = project.data.get(ref)
            if parent:
                print(f"     • {ref}: {parent.value} ({parent.nature.value})")
    
    if dp.alternatives:
        print(f"   Scenarios:")
        for scenario, val in dp.alternatives.items():
            print(f"     • {scenario}: {val}")


if __name__ == "__main__":
    # Create the demo project
    project = create_apple_demo()
    
    # Print summary
    print_summary(project)
    
    # Print tables
    print_table(project, "income_statement_table")
    print_table(project, "balance_sheet_table") 
    print_table(project, "cash_flow_table")
    print_table(project, "valuation_table")
    
    # Demo lineage tracing
    demo_lineage(project, "eps_diluted.fy26e")
    demo_lineage(project, "fcf.fy26e")
    demo_lineage(project, "price_target")
    
    print("\n✅ Demo complete!")
