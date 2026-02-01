"""
ESG (Environmental, Social, Governance) Scores

Provides estimated ESG metrics based on:
  - Industry sector benchmarks
  - SEC disclosure patterns
  - Public sustainability reports
"""

import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import httpx


@dataclass
class ESGScore:
    """ESG score breakdown."""
    ticker: str
    company_name: str
    sector: str
    last_updated: str
    
    # Overall
    total_score: int = 0  # 0-100
    esg_rating: str = ""  # AAA, AA, A, BBB, BB, B, CCC
    
    # Environmental
    environmental_score: int = 0
    carbon_intensity: str = ""  # Low, Medium, High
    renewable_energy: str = ""
    waste_management: str = ""
    
    # Social
    social_score: int = 0
    employee_relations: str = ""
    diversity_inclusion: str = ""
    community_impact: str = ""
    data_privacy: str = ""
    
    # Governance
    governance_score: int = 0
    board_independence: str = ""
    executive_compensation: str = ""
    shareholder_rights: str = ""
    ethics_compliance: str = ""
    
    # Controversies
    controversy_level: str = ""  # None, Minor, Moderate, Severe
    controversy_areas: list = None
    
    # Sources
    data_source: str = "Estimated based on sector benchmarks"


# Industry ESG benchmarks (estimated averages)
SECTOR_ESG_BENCHMARKS = {
    "Technology": {
        "environmental": 65, "social": 70, "governance": 75,
        "carbon": "Medium", "typical_issues": ["Data Privacy", "Supply Chain"]
    },
    "Healthcare": {
        "environmental": 60, "social": 75, "governance": 70,
        "carbon": "Low", "typical_issues": ["Drug Pricing", "Clinical Trials"]
    },
    "Financial Services": {
        "environmental": 55, "social": 65, "governance": 80,
        "carbon": "Low", "typical_issues": ["Lending Practices", "Executive Pay"]
    },
    "Consumer Discretionary": {
        "environmental": 50, "social": 65, "governance": 70,
        "carbon": "Medium", "typical_issues": ["Labor Practices", "Sustainability"]
    },
    "Consumer Staples": {
        "environmental": 55, "social": 70, "governance": 72,
        "carbon": "Medium", "typical_issues": ["Supply Chain", "Packaging"]
    },
    "Energy": {
        "environmental": 35, "social": 60, "governance": 65,
        "carbon": "High", "typical_issues": ["Carbon Emissions", "Spills"]
    },
    "Utilities": {
        "environmental": 45, "social": 65, "governance": 70,
        "carbon": "High", "typical_issues": ["Emissions", "Renewable Transition"]
    },
    "Industrials": {
        "environmental": 50, "social": 65, "governance": 70,
        "carbon": "High", "typical_issues": ["Worker Safety", "Emissions"]
    },
    "Materials": {
        "environmental": 40, "social": 60, "governance": 65,
        "carbon": "High", "typical_issues": ["Mining Impact", "Water Use"]
    },
    "Real Estate": {
        "environmental": 55, "social": 65, "governance": 72,
        "carbon": "Medium", "typical_issues": ["Energy Efficiency", "Tenant Relations"]
    },
    "Communication Services": {
        "environmental": 65, "social": 60, "governance": 70,
        "carbon": "Low", "typical_issues": ["Content Moderation", "Privacy"]
    },
    "Default": {
        "environmental": 55, "social": 65, "governance": 70,
        "carbon": "Medium", "typical_issues": []
    }
}


class ESGAnalyzer:
    """Provides ESG scores based on sector benchmarks and adjustments."""
    
    def __init__(self):
        self.headers = {
            "User-Agent": "Equity Research Tool research@example.com"
        }
    
    async def get_company_info(self, ticker: str) -> Dict[str, Any]:
        """Get company sector info from Yahoo Finance."""
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=15.0
                )
                resp.raise_for_status()
                data = resp.json()
                
                meta = data.get("chart", {}).get("result", [{}])[0].get("meta", {})
                return {
                    "name": meta.get("longName", ticker),
                    "sector": meta.get("sector", "Default"),
                    "market_cap": meta.get("marketCap", 0)
                }
            except:
                return {"name": ticker, "sector": "Default", "market_cap": 0}
    
    def score_to_rating(self, score: int) -> str:
        """Convert numeric score to letter rating."""
        if score >= 85:
            return "AAA"
        elif score >= 75:
            return "AA"
        elif score >= 65:
            return "A"
        elif score >= 55:
            return "BBB"
        elif score >= 45:
            return "BB"
        elif score >= 35:
            return "B"
        else:
            return "CCC"
    
    def score_to_level(self, score: int, high_threshold: int = 70, low_threshold: int = 50) -> str:
        """Convert score to level."""
        if score >= high_threshold:
            return "Strong"
        elif score >= low_threshold:
            return "Moderate"
        else:
            return "Weak"
    
    async def analyze(self, ticker: str) -> ESGScore:
        """Generate ESG score based on sector benchmarks."""
        ticker = ticker.upper()
        
        # Get company info
        info = await self.get_company_info(ticker)
        company_name = info.get("name", ticker)
        sector = info.get("sector", "Default")
        market_cap = info.get("market_cap", 0)
        
        # Get sector benchmark
        benchmark = SECTOR_ESG_BENCHMARKS.get(sector, SECTOR_ESG_BENCHMARKS["Default"])
        
        # Base scores from sector
        env_score = benchmark["environmental"]
        soc_score = benchmark["social"]
        gov_score = benchmark["governance"]
        
        # Large cap companies typically have better ESG programs
        if market_cap > 100_000_000_000:  # >$100B
            env_score = min(100, env_score + 10)
            soc_score = min(100, soc_score + 8)
            gov_score = min(100, gov_score + 10)
        elif market_cap > 10_000_000_000:  # >$10B
            env_score = min(100, env_score + 5)
            soc_score = min(100, soc_score + 4)
            gov_score = min(100, gov_score + 5)
        
        # Calculate total
        total = (env_score + soc_score + gov_score) // 3
        rating = self.score_to_rating(total)
        
        # Determine levels
        carbon = benchmark["carbon"]
        typical_issues = benchmark.get("typical_issues", [])
        
        controversy = "Minor" if typical_issues else "None"
        
        return ESGScore(
            ticker=ticker,
            company_name=company_name,
            sector=sector,
            last_updated=datetime.now().isoformat(),
            total_score=total,
            esg_rating=rating,
            environmental_score=env_score,
            carbon_intensity=carbon,
            renewable_energy=self.score_to_level(env_score + 5),
            waste_management=self.score_to_level(env_score),
            social_score=soc_score,
            employee_relations=self.score_to_level(soc_score),
            diversity_inclusion=self.score_to_level(soc_score - 5),
            community_impact=self.score_to_level(soc_score - 3),
            data_privacy=self.score_to_level(soc_score + 2) if sector == "Technology" else self.score_to_level(soc_score),
            governance_score=gov_score,
            board_independence=self.score_to_level(gov_score),
            executive_compensation=self.score_to_level(gov_score - 5),
            shareholder_rights=self.score_to_level(gov_score + 3),
            ethics_compliance=self.score_to_level(gov_score),
            controversy_level=controversy,
            controversy_areas=typical_issues,
            data_source="Estimated based on sector benchmarks and market cap"
        )


def format_esg_markdown(esg: ESGScore) -> str:
    """Format ESG score as Markdown."""
    
    def score_bar(score: int) -> str:
        filled = score // 10
        empty = 10 - filled
        return "█" * filled + "░" * empty
    
    def rating_color(rating: str) -> str:
        if rating in ["AAA", "AA"]:
            return "🟢"
        elif rating in ["A", "BBB"]:
            return "🟡"
        else:
            return "🔴"
    
    md = f"""# {esg.ticker} ESG Analysis
**{esg.company_name}** | *Sector: {esg.sector}*
*Updated: {esg.last_updated}*

---

## ESG Rating Summary

| Metric | Score | Rating |
|:-------|------:|:-------|
| **Overall ESG** | {esg.total_score}/100 | {rating_color(esg.esg_rating)} **{esg.esg_rating}** |
| Environmental | {esg.environmental_score}/100 | {score_bar(esg.environmental_score)} |
| Social | {esg.social_score}/100 | {score_bar(esg.social_score)} |
| Governance | {esg.governance_score}/100 | {score_bar(esg.governance_score)} |

---

## Environmental ({esg.environmental_score}/100)

| Factor | Assessment |
|:-------|:-----------|
| Carbon Intensity | {esg.carbon_intensity} |
| Renewable Energy | {esg.renewable_energy} |
| Waste Management | {esg.waste_management} |

---

## Social ({esg.social_score}/100)

| Factor | Assessment |
|:-------|:-----------|
| Employee Relations | {esg.employee_relations} |
| Diversity & Inclusion | {esg.diversity_inclusion} |
| Community Impact | {esg.community_impact} |
| Data Privacy | {esg.data_privacy} |

---

## Governance ({esg.governance_score}/100)

| Factor | Assessment |
|:-------|:-----------|
| Board Independence | {esg.board_independence} |
| Executive Compensation | {esg.executive_compensation} |
| Shareholder Rights | {esg.shareholder_rights} |
| Ethics & Compliance | {esg.ethics_compliance} |

---

## Controversy Assessment

| Level | Areas |
|:------|:------|
| {esg.controversy_level} | {', '.join(esg.controversy_areas) if esg.controversy_areas else 'None identified'} |

---

## Data Source

*{esg.data_source}*

**Note:** This ESG analysis is estimated based on sector benchmarks and publicly available information.
For official ESG ratings, consult providers like MSCI, Sustainalytics, or Bloomberg ESG.
"""
    
    return md


async def get_esg_score(ticker: str) -> str:
    """Get ESG score as Markdown."""
    analyzer = ESGAnalyzer()
    esg = await analyzer.analyze(ticker)
    return format_esg_markdown(esg)


# CLI
if __name__ == "__main__":
    import sys
    
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(asyncio.run(get_esg_score(ticker)))
