# SEC EDGAR MCP Tools

MCP (Model Context Protocol) server for fetching SEC EDGAR filings and extracting financial data.

## Features

- Fetch 10-K (annual) and 10-Q (quarterly) filings
- Extract the **three main financial statements**:
  - **Income Statement** (Revenue, Expenses, Net Income, EPS)
  - **Balance Sheet** (Assets, Liabilities, Equity)
  - **Cash Flow Statement** (Operating, Investing, Financing)
- Compare metrics across multiple companies
- Automatic CIK lookup from ticker symbols

## Installation

```bash
pip install mcp httpx
```

## Usage

### As MCP Server

Add to your MCP configuration:

```json
{
  "mcpServers": {
    "sec-edgar": {
      "command": "python",
      "args": ["-m", "mcp_sec.server", "--mcp"],
      "cwd": "/path/to/equity"
    }
  }
}
```

### CLI Testing

```bash
# Get company info
python -m mcp_sec.server info AAPL

# List filings
python -m mcp_sec.server filings AAPL 10-K

# Get all financial statements (3 tables)
python -m mcp_sec.server all AAPL 10-K

# Get individual statements
python -m mcp_sec.server income MSFT
python -m mcp_sec.server balance MSFT
python -m mcp_sec.server cashflow MSFT

# Compare companies
python -m mcp_sec.server compare AAPL,MSFT,GOOGL NetIncome
```

## Available Tools

| Tool | Description |
|------|-------------|
| `sec_get_company_info` | Get CIK, company name, fiscal year end |
| `sec_get_filings` | List 10-K or 10-Q filings |
| `sec_get_financial_statements` | Get all 3 financial statements |
| `sec_get_income_statement` | Get income statement only |
| `sec_get_balance_sheet` | Get balance sheet only |
| `sec_get_cash_flow` | Get cash flow statement only |
| `sec_compare_companies` | Compare a metric across companies |

## Data Structure

### Income Statement Fields
- `Revenues`, `CostOfRevenue`, `GrossProfit`
- `OperatingExpenses`, `ResearchAndDevelopment`, `SellingGeneralAdmin`
- `OperatingIncome`, `IncomeBeforeTax`, `IncomeTaxExpense`, `NetIncome`
- `EPS_Basic`, `EPS_Diluted`

### Balance Sheet Fields
- **Assets**: `CashAndEquivalents`, `AccountsReceivable`, `Inventory`, `TotalCurrentAssets`, `PropertyPlantEquipment`, `TotalAssets`
- **Liabilities**: `AccountsPayable`, `LongTermDebt`, `TotalLiabilities`
- **Equity**: `RetainedEarnings`, `TotalEquity`

### Cash Flow Fields
- **Operating**: `NetIncome_CF`, `DepreciationAmortization`, `CashFromOperations`
- **Investing**: `CapitalExpenditures`, `CashFromInvesting`
- **Financing**: `DebtRepayments`, `StockRepurchases`, `DividendsPaid`, `CashFromFinancing`

## Example Output

```
FINANCIAL STATEMENTS: AAPL
Form: 10-K | Periods: 4

INCOME STATEMENTS
============================================================
  Income Statement - 2025
  Period: 2025-09-27 (ANNUAL)
============================================================
  GrossProfit         :        $195.20B
  OperatingIncome     :        $133.05B
  NetIncome           :        $112.01B
  EPS_Diluted         :           $7.46
============================================================
```

## Notes

- Data comes from SEC EDGAR XBRL API (structured data)
- Requires valid User-Agent header (SEC requirement)
- Rate limited by SEC (best practice: 10 requests/second)
- Works with any publicly traded US company
