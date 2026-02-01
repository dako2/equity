"""
News & Sentiment Analysis

Fetches recent news and analyzes sentiment:
  - SEC 8-K filings (material events)
  - Financial news from free APIs
  - Basic sentiment scoring
"""

import asyncio
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import httpx


@dataclass
class NewsItem:
    """A single news item."""
    title: str
    source: str
    date: str
    url: str
    summary: str = ""
    sentiment: str = ""  # "Positive", "Negative", "Neutral"
    sentiment_score: float = 0  # -1 to 1


@dataclass
class NewsSentiment:
    """News sentiment analysis results."""
    ticker: str
    company_name: str
    last_updated: str
    news_items: List[NewsItem] = field(default_factory=list)
    overall_sentiment: str = "Neutral"
    sentiment_score: float = 0
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    key_themes: List[str] = field(default_factory=list)
    material_events: List[Dict[str, Any]] = field(default_factory=list)


class NewsSentimentAnalyzer:
    """Analyzes news and sentiment for a stock."""
    
    # Sentiment keywords (simple rule-based)
    POSITIVE_WORDS = {
        'beat', 'beats', 'exceeds', 'exceeded', 'surpass', 'strong', 'growth',
        'profit', 'gains', 'upgrade', 'upgraded', 'buy', 'outperform', 'bullish',
        'record', 'high', 'soars', 'jumps', 'rally', 'rallies', 'positive',
        'breakthrough', 'innovation', 'partnership', 'acquisition', 'expand',
        'dividend', 'buyback', 'revenue', 'earnings', 'momentum', 'surge',
        'optimistic', 'confident', 'success', 'successful', 'launches'
    }
    
    NEGATIVE_WORDS = {
        'miss', 'misses', 'missed', 'disappoints', 'weak', 'decline', 'loss',
        'losses', 'downgrade', 'downgraded', 'sell', 'underperform', 'bearish',
        'low', 'falls', 'drops', 'crash', 'plunge', 'negative', 'lawsuit',
        'investigation', 'recall', 'layoffs', 'cuts', 'warning', 'risk',
        'concerns', 'troubled', 'fails', 'failed', 'delay', 'delayed',
        'pessimistic', 'uncertainty', 'slump', 'tumbles', 'slides'
    }
    
    def __init__(self):
        self.headers = {
            "User-Agent": "Equity Research Tool research@example.com"
        }
    
    async def get_sec_8k_filings(
        self,
        ticker: str,
        cik: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent 8-K filings from SEC (material events)."""
        url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
        
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(url, headers=self.headers, timeout=15.0)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                return []
        
        filings = []
        recent = data.get("filings", {}).get("recent", {})
        
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        
        for i, form in enumerate(forms):
            if form == "8-K" and len(filings) < limit:
                acc = accessions[i].replace("-", "")
                doc = primary_docs[i] if i < len(primary_docs) else ""
                filings.append({
                    "form": form,
                    "date": dates[i] if i < len(dates) else "",
                    "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}",
                    "accession": accessions[i]
                })
        
        return filings
    
    async def get_cik(self, ticker: str) -> Optional[str]:
        """Get CIK for a ticker."""
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
        """Get company name for a ticker."""
        url = "https://www.sec.gov/files/company_tickers.json"
        
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(url, headers=self.headers, timeout=15.0)
                resp.raise_for_status()
                data = resp.json()
            except:
                return ticker
        
        ticker = ticker.upper()
        for entry in data.values():
            if entry.get("ticker") == ticker:
                return entry.get("title", ticker)
        return ticker
    
    async def get_yahoo_news(self, ticker: str, limit: int = 10) -> List[NewsItem]:
        """Get news from Yahoo Finance."""
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        
        news_items = []
        
        # Yahoo doesn't have a clean news API, use search
        search_url = f"https://query2.finance.yahoo.com/v1/finance/search"
        params = {"q": ticker, "newsCount": limit}
        
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(
                    search_url,
                    params=params,
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=15.0
                )
                resp.raise_for_status()
                data = resp.json()
                
                for item in data.get("news", [])[:limit]:
                    title = item.get("title", "")
                    sentiment, score = self.analyze_sentiment(title)
                    
                    news_items.append(NewsItem(
                        title=title,
                        source=item.get("publisher", "Unknown"),
                        date=datetime.fromtimestamp(
                            item.get("providerPublishTime", 0)
                        ).strftime("%Y-%m-%d"),
                        url=item.get("link", ""),
                        sentiment=sentiment,
                        sentiment_score=score
                    ))
            except Exception as e:
                pass
        
        return news_items
    
    def analyze_sentiment(self, text: str) -> tuple:
        """Simple keyword-based sentiment analysis."""
        text_lower = text.lower()
        words = set(re.findall(r'\w+', text_lower))
        
        positive_count = len(words & self.POSITIVE_WORDS)
        negative_count = len(words & self.NEGATIVE_WORDS)
        
        total = positive_count + negative_count
        if total == 0:
            return "Neutral", 0
        
        score = (positive_count - negative_count) / total
        
        if score > 0.3:
            return "Positive", score
        elif score < -0.3:
            return "Negative", score
        else:
            return "Neutral", score
    
    def extract_themes(self, news_items: List[NewsItem]) -> List[str]:
        """Extract common themes from news."""
        theme_keywords = {
            "Earnings": ["earnings", "revenue", "profit", "eps", "quarter"],
            "Guidance": ["guidance", "outlook", "forecast", "expects"],
            "Product": ["launch", "product", "release", "announces", "new"],
            "Legal": ["lawsuit", "investigation", "sec", "doj", "legal"],
            "M&A": ["acquisition", "merger", "deal", "buyout", "acquire"],
            "Analyst": ["upgrade", "downgrade", "rating", "target", "analyst"],
            "Market": ["market", "stock", "shares", "trading", "price"],
            "Management": ["ceo", "cfo", "executive", "appoint", "resign"],
            "Dividend": ["dividend", "buyback", "repurchase", "capital"],
            "Industry": ["industry", "sector", "competition", "competitor"]
        }
        
        all_text = " ".join([n.title.lower() for n in news_items])
        found_themes = []
        
        for theme, keywords in theme_keywords.items():
            if any(kw in all_text for kw in keywords):
                found_themes.append(theme)
        
        return found_themes[:5]  # Top 5 themes
    
    async def analyze(self, ticker: str) -> NewsSentiment:
        """Full news and sentiment analysis."""
        ticker = ticker.upper()
        
        # Get company info
        cik = await self.get_cik(ticker)
        company_name = await self.get_company_name(ticker)
        
        # Get news
        news_items = await self.get_yahoo_news(ticker, limit=15)
        
        # Get 8-K filings
        material_events = []
        if cik:
            filings = await self.get_sec_8k_filings(ticker, cik, limit=5)
            material_events = filings
        
        # Calculate overall sentiment
        if news_items:
            scores = [n.sentiment_score for n in news_items]
            avg_score = sum(scores) / len(scores)
            
            positive_count = sum(1 for n in news_items if n.sentiment == "Positive")
            negative_count = sum(1 for n in news_items if n.sentiment == "Negative")
            neutral_count = sum(1 for n in news_items if n.sentiment == "Neutral")
            
            if avg_score > 0.2:
                overall = "Positive"
            elif avg_score < -0.2:
                overall = "Negative"
            else:
                overall = "Neutral"
        else:
            avg_score = 0
            overall = "Neutral"
            positive_count = negative_count = neutral_count = 0
        
        themes = self.extract_themes(news_items)
        
        return NewsSentiment(
            ticker=ticker,
            company_name=company_name,
            last_updated=datetime.now().isoformat(),
            news_items=news_items,
            overall_sentiment=overall,
            sentiment_score=avg_score,
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            key_themes=themes,
            material_events=material_events
        )


def format_news_markdown(ns: NewsSentiment) -> str:
    """Format news sentiment as Markdown."""
    
    sentiment_emoji = "🟢" if ns.overall_sentiment == "Positive" else \
                     "🔴" if ns.overall_sentiment == "Negative" else "🟡"
    
    md = f"""# {ns.ticker} News & Sentiment
**{ns.company_name}** | *Updated: {ns.last_updated}*

---

## Sentiment Summary

| Metric | Value |
|:-------|------:|
| **Overall Sentiment** | {sentiment_emoji} **{ns.overall_sentiment}** |
| Sentiment Score | {ns.sentiment_score:.2f} |
| Positive Articles | {ns.positive_count} |
| Negative Articles | {ns.negative_count} |
| Neutral Articles | {ns.neutral_count} |

### Key Themes
"""
    
    if ns.key_themes:
        md += ", ".join([f"**{t}**" for t in ns.key_themes]) + "\n"
    else:
        md += "No major themes identified\n"
    
    md += "\n---\n\n## Recent News\n\n"
    
    if ns.news_items:
        for item in ns.news_items[:10]:
            sent_icon = "🟢" if item.sentiment == "Positive" else \
                       "🔴" if item.sentiment == "Negative" else "⚪"
            md += f"- {sent_icon} **{item.title}**\n"
            md += f"  - *{item.source}* | {item.date}\n\n"
    else:
        md += "*No recent news found*\n"
    
    md += "\n---\n\n## Material Events (SEC 8-K Filings)\n\n"
    
    if ns.material_events:
        md += "| Date | Form | Link |\n"
        md += "|:-----|:----:|:-----|\n"
        for event in ns.material_events:
            md += f"| {event['date']} | {event['form']} | [View]({event['url']}) |\n"
    else:
        md += "*No recent 8-K filings found*\n"
    
    md += """
---

*Sentiment analysis is rule-based and should be used as one input among many.*
*8-K filings contain material events that may affect stock price.*
"""
    
    return md


async def get_news_sentiment(ticker: str) -> str:
    """Get news sentiment as Markdown."""
    analyzer = NewsSentimentAnalyzer()
    sentiment = await analyzer.analyze(ticker)
    return format_news_markdown(sentiment)


# CLI
if __name__ == "__main__":
    import sys
    
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(asyncio.run(get_news_sentiment(ticker)))
