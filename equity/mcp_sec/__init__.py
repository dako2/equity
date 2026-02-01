"""
SEC EDGAR MCP Tools - Complete Equity Research Data Suite

## Data Sources:
  1. SEC EDGAR - Financial Statements (10-K, 10-Q)
  2. Market Data - Real-time price, 52-week, volume
  3. Analysis - Metrics, valuation, rating
  4. Insider Trading - Form 4 filings
  5. Institutional Holdings - 13F filings
  6. Earnings - Calendar and estimates
  7. Peer Comparison - Industry comparables
  8. Technical Analysis - SMA, RSI, MACD, Bollinger
  9. News & Sentiment - Headlines and sentiment scoring
  10. ESG Scores - Environmental, Social, Governance
  11. Guidance - Management outlook from 8-K filings
  12. Earnings Transcripts - NASDAQ earnings history + beat rate
  13. Analyst Research - NASDAQ ratings, targets, ratios, Zacks coverage
  14. Report Generator - Full equity research report with LLM analyst
  15. Data to Visual - LLM-powered data → tables, charts, diagrams

## CLI Usage:

    # Financial Statements (3 tables)
    python -m mcp_sec.server all AAPL
    
    # Market Data (price, 52W, volume)
    python -m mcp_sec.market_data AAPL
    
    # Full Equity Analysis
    python -m mcp_sec.analyzer AAPL
    
    # Insider Trading
    python -m mcp_sec.insider_holdings insider AAPL
    
    # Institutional Holdings
    python -m mcp_sec.insider_holdings institutions AAPL
    
    # Earnings Calendar
    python -m mcp_sec.earnings_peers earnings AAPL
    
    # Industry Peers
    python -m mcp_sec.earnings_peers peers AAPL
    
    # Peer Comparison Table
    python -m mcp_sec.earnings_peers compare NVDA
    
    # Technical Analysis
    python -m mcp_sec.technicals AAPL
    
    # News & Sentiment
    python -m mcp_sec.news_sentiment AAPL
    
    # ESG Scores
    python -m mcp_sec.esg_scores AAPL
    
    # Management Guidance
    python -m mcp_sec.guidance_transcripts AAPL
    
    # Earnings History (NASDAQ)
    python -m mcp_sec.earnings_transcripts list AAPL
    
    # Analyst Research (NASDAQ + Zacks)
    python -m mcp_sec.analyst_research AAPL
    
    # Full Equity Research Report (LLM-powered)
    python -m mcp_sec.report_generator AAPL examples/AAPL_report.md
    
    # Data to Visual Converter (demo)
    python -m mcp_sec.data_to_visual
"""

from .server import (
    SECClient,
    FinancialStatement,
    FilingInfo,
    handle_get_company_info,
    handle_get_filings,
    handle_get_financial_statements,
    handle_get_income_statement,
    handle_get_balance_sheet,
    handle_get_cash_flow,
    handle_compare_companies,
)

from .analyzer import (
    FinancialAnalyzer,
    FinancialMetrics,
    GrowthMetrics,
    ValuationMetrics,
    EquityAnalysis,
    analyze_ticker,
    format_analysis_markdown,
)

from .market_data import (
    MarketDataProvider,
    MarketData,
    get_market_data,
)

from .insider_holdings import (
    InsiderHoldingsClient,
    InsiderTransaction,
    InstitutionalHolder,
    get_insider_trading,
    get_institutional_holdings,
)

from .earnings_peers import (
    EarningsPeersClient,
    EarningsInfo,
    PeerMetrics,
    get_earnings,
    get_peers,
    get_peer_comparison,
)

from .technicals import (
    TechnicalAnalyzer,
    TechnicalIndicators,
    get_technicals,
)

from .news_sentiment import (
    NewsSentimentAnalyzer,
    NewsSentiment,
    NewsItem,
    get_news_sentiment,
)

from .esg_scores import (
    ESGAnalyzer,
    ESGScore,
    get_esg_score,
)

from .guidance_transcripts import (
    GuidanceParser,
    Guidance,
    get_company_guidance,
)

from .earnings_transcripts import (
    EarningsTranscriptClient,
    EarningsResult,
    get_transcript_list,
    get_transcript_analysis,
)

from .analyst_research import (
    AnalystResearchClient,
    AnalystResearch,
    FinancialRatios,
    get_analyst_research,
)

from .data_to_visual import (
    DataToVisualConverter,
    data_to_table,
    data_to_bar_chart,
    data_to_line_chart,
    data_to_pie_chart,
    data_to_sparkline,
    data_to_visual,
)

__all__ = [
    # SEC Server
    "SECClient",
    "FinancialStatement",
    "FilingInfo",
    "handle_get_company_info",
    "handle_get_filings",
    "handle_get_financial_statements",
    "handle_get_income_statement",
    "handle_get_balance_sheet",
    "handle_get_cash_flow",
    "handle_compare_companies",
    # Analyzer
    "FinancialAnalyzer",
    "FinancialMetrics",
    "GrowthMetrics",
    "ValuationMetrics",
    "EquityAnalysis",
    "analyze_ticker",
    "format_analysis_markdown",
    # Market Data
    "MarketDataProvider",
    "MarketData",
    "get_market_data",
    # Insider & Holdings
    "InsiderHoldingsClient",
    "InsiderTransaction",
    "InstitutionalHolder",
    "get_insider_trading",
    "get_institutional_holdings",
    # Earnings & Peers
    "EarningsPeersClient",
    "EarningsInfo",
    "PeerMetrics",
    "get_earnings",
    "get_peers",
    "get_peer_comparison",
    # Technical Analysis
    "TechnicalAnalyzer",
    "TechnicalIndicators",
    "get_technicals",
    # News & Sentiment
    "NewsSentimentAnalyzer",
    "NewsSentiment",
    "NewsItem",
    "get_news_sentiment",
    # ESG Scores
    "ESGAnalyzer",
    "ESGScore",
    "get_esg_score",
    # Guidance
    "GuidanceParser",
    "Guidance",
    "get_company_guidance",
    # Earnings Transcripts
    "EarningsTranscriptClient",
    "EarningsResult",
    "get_transcript_list",
    "get_transcript_analysis",
    # Analyst Research
    "AnalystResearchClient",
    "AnalystResearch",
    "FinancialRatios",
    "get_analyst_research",
    # Data to Visual
    "DataToVisualConverter",
    "data_to_table",
    "data_to_bar_chart",
    "data_to_line_chart",
    "data_to_pie_chart",
    "data_to_sparkline",
    "data_to_visual",
]
