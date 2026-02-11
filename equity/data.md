# Research Data Model

A flexible, schema-agnostic data structure for financial research reports.

## Design Principles

1. **Entity-agnostic** — Works for equities, fixed income, macro, thematic research
2. **Schema-flexible** — Custom fields via `tags` and `meta`, not rigid schemas
3. **Provenance-first** — Every value tracks its source and derivation
4. **Scenario-aware** — Built-in support for base/bull/bear cases
5. **Graph structure** — Values can derive from other values (DAG)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ResearchProject                          │
│  (container for all research on a topic)                    │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   ┌─────────┐          ┌──────────┐         ┌────────────┐
   │ Entity  │          │DataPoint │         │ContentBlock│
   │ (what)  │          │ (values) │         │ (visuals)  │
   └─────────┘          └──────────┘         └────────────┘
        │                     │                     │
   company, bond,        observed,             table, chart,
   indicator, theme      assumed,              text, metric
                         derived
```

---

## Core Types

### DataNature

Classifies how a value was obtained:

| Nature | Description | Example |
|--------|-------------|---------|
| `OBSERVED` | Historical fact from authoritative source | Revenue FY2024: $96.7B |
| `ASSUMED` | Analyst belief or model input | Revenue growth rate: 12% |
| `DERIVED` | Calculated from other values | Revenue FY2026E: $121B |
| `EXTERNAL` | Third-party estimate | Street consensus EPS |

```python
from enum import Enum

class DataNature(Enum):
    OBSERVED = "observed"
    ASSUMED = "assumed"
    DERIVED = "derived"
    EXTERNAL = "external"
```

---

### DataPoint

The atomic unit of data with full provenance.

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

@dataclass
class DataPoint:
    """
    Atomic data unit - intentionally flexible.
    Schema defined by 'tags' and 'meta', not rigid fields.
    """
    id: str                         # Unique key: "revenue.fy2025"
    value: Any                      # The actual value
    nature: DataNature              # How was this obtained?
    
    # ─── Temporal ───
    as_of: datetime | None = None   # When this value was true
    fetched_at: datetime | None = None  # When we retrieved it
    
    # ─── Provenance ───
    source: str | None = None       # "sec:10-K", "bloomberg", "manual"
    source_ref: str | None = None   # URL, document ID, page number
    
    # ─── Derivation (for DERIVED values) ───
    derived_from: list[str] = field(default_factory=list)  # Parent DataPoint IDs
    formula: str | None = None      # Human-readable: "revenue.fy25 * (1 + growth_rate)"
    
    # ─── Flexible Schema ───
    tags: list[str] = field(default_factory=list)
    # Examples: ["revenue", "fy2025", "usd", "gaap"]
    
    meta: dict[str, Any] = field(default_factory=dict)
    # Examples: {"unit": "$M", "period_type": "annual", "segment": "iPhone"}
    
    # ─── Scenarios ───
    scenario: str = "base"
    alternatives: dict[str, Any] = field(default_factory=dict)
    # Example: {"bull": 120000, "bear": 95000}
```

#### Example DataPoints

```python
# Historical fact
revenue_fy24 = DataPoint(
    id="revenue.fy2024",
    value=96_773_000_000,
    nature=DataNature.OBSERVED,
    as_of=datetime(2024, 9, 30),
    source="sec:10-K",
    source_ref="https://sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193",
    tags=["revenue", "fy2024", "usd", "annual"],
    meta={"unit": "$", "period_end": "2024-09-30"}
)

# Assumption
growth_rate = DataPoint(
    id="revenue_growth.fy2026",
    value=0.12,
    nature=DataNature.ASSUMED,
    source="analyst",
    tags=["growth_rate", "revenue", "fy2026"],
    meta={"rationale": "Expect iPhone 17 cycle + Services growth"},
    alternatives={"bull": 0.15, "bear": 0.08}
)

# Derived projection
revenue_fy26e = DataPoint(
    id="revenue.fy2026e",
    value=121_500_000_000,
    nature=DataNature.DERIVED,
    derived_from=["revenue.fy2025", "revenue_growth.fy2026"],
    formula="revenue.fy2025 * (1 + revenue_growth.fy2026)",
    tags=["revenue", "fy2026", "projection"],
    alternatives={"bull": 128_000_000_000, "bear": 112_000_000_000}
)
```

---

### Entity

The subject of research — not limited to equities.

```python
@dataclass
class Entity:
    """
    The subject of research - could be anything.
    """
    id: str                         # Unique within project
    type: str                       # "company", "bond", "indicator", "sector", "theme"
    name: str                       # Display name
    
    # Flexible identifiers (type-dependent)
    identifiers: dict[str, str] = field(default_factory=dict)
    
    # Flexible attributes
    attributes: dict[str, Any] = field(default_factory=dict)
```

#### Entity Examples by Type

```python
# Company (equity)
apple = Entity(
    id="aapl",
    type="company",
    name="Apple Inc",
    identifiers={
        "ticker": "AAPL",
        "cusip": "037833100",
        "isin": "US0378331005",
        "cik": "0000320193"
    },
    attributes={
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "exchange": "NASDAQ",
        "market_cap": 3_000_000_000_000
    }
)

# Bond (fixed income)
apple_bond = Entity(
    id="aapl-2030",
    type="bond",
    name="Apple 3.5% 2030",
    identifiers={
        "cusip": "037833DV4",
        "isin": "US037833DV49"
    },
    attributes={
        "issuer": "Apple Inc",
        "coupon": 3.5,
        "maturity": "2030-05-01",
        "rating_sp": "AA+",
        "rating_moody": "Aaa"
    }
)

# Macro indicator
us_gdp = Entity(
    id="us-gdp",
    type="indicator",
    name="US Real GDP",
    identifiers={
        "fred_series": "GDPC1",
        "region": "US"
    },
    attributes={
        "frequency": "quarterly",
        "unit": "billions_chained_2017_dollars",
        "seasonally_adjusted": True
    }
)

# Thematic
ev_theme = Entity(
    id="ev-adoption",
    type="theme",
    name="Electric Vehicle Adoption",
    identifiers={},
    attributes={
        "related_tickers": ["TSLA", "RIVN", "NIO", "LI"],
        "key_drivers": ["battery_cost", "charging_infra", "regulation"]
    }
)
```

---

### ContentBlock

Generic container for visual elements — type determines rendering.

```python
@dataclass
class ContentBlock:
    """
    Generic content container - type determines rendering.
    """
    id: str
    type: str                       # "table", "chart", "text", "metric_card", ...
    title: str | None = None
    
    # Data bindings
    data_refs: list[str] = field(default_factory=list)  # DataPoint IDs used
    
    # Type-specific configuration
    config: dict[str, Any] = field(default_factory=dict)
    
    # Semantic placement
    section: str = "body"           # "summary", "body", "appendix"
    order: int = 0                  # Render order within section
```

#### ContentBlock Configurations by Type

**Table**
```python
income_table = ContentBlock(
    id="income_statement",
    type="table",
    title="Income Statement",
    data_refs=["revenue.fy2024", "revenue.fy2025", "revenue.fy2026e", ...],
    section="body",
    order=1,
    config={
        "columns": [
            {"key": "metric", "label": ""},
            {"key": "fy2024", "label": "FY2024"},
            {"key": "fy2025", "label": "FY2025"},
            {"key": "fy2026e", "label": "FY2026E", "style": "projection"}
        ],
        "rows": [
            {"metric": "Revenue", "fy2024": "{revenue.fy2024}", "fy2025": "{revenue.fy2025}", "fy2026e": "{revenue.fy2026e}"},
            {"metric": "Gross Profit", "fy2024": "{gross_profit.fy2024}", ...},
            {"metric": "Operating Income", ...}
        ],
        "format": {
            "revenue.*": "$,.0f",
            "margin.*": ".1%"
        },
        "style": "financial"
    }
)
```

**Chart**
```python
revenue_chart = ContentBlock(
    id="revenue_trend",
    type="chart",
    title="Revenue Trend",
    data_refs=["revenue.fy2022", "revenue.fy2023", "revenue.fy2024", "revenue.fy2025", "revenue.fy2026e"],
    section="summary",
    config={
        "chart_type": "bar",
        "x_axis": {"labels": ["FY22", "FY23", "FY24", "FY25", "FY26E"]},
        "y_axis": {"label": "Revenue ($B)", "format": "$,.0f"},
        "series": [
            {
                "name": "Revenue",
                "data_refs": ["revenue.fy2022", "revenue.fy2023", "revenue.fy2024", "revenue.fy2025", "revenue.fy2026e"],
                "color": "#2563eb"
            }
        ],
        "annotations": [
            {"type": "divider", "after_index": 3, "label": "Projected"}
        ]
    }
)
```

**Text Block**
```python
thesis = ContentBlock(
    id="investment_thesis",
    type="text",
    title="Investment Thesis",
    data_refs=["revenue_growth.fy2026", "price_target"],
    section="summary",
    config={
        "template": """
We rate Apple **Overweight** with a price target of {price_target}.

Key drivers:
- Revenue growth of {revenue_growth.fy2026} expected in FY26
- Services margin expansion continues
- iPhone 17 cycle anticipated to drive upgrade momentum
        """,
        "format": {
            "price_target": "$,.2f",
            "revenue_growth.fy2026": ".1%"
        }
    }
)
```

**Metric Card**
```python
rating_card = ContentBlock(
    id="rating_summary",
    type="metric_card",
    data_refs=["price_target", "current_price", "upside"],
    section="header",
    config={
        "metrics": [
            {"label": "Rating", "value": "Overweight", "style": "badge-green"},
            {"label": "Price Target", "ref": "price_target", "format": "$,.2f"},
            {"label": "Current Price", "ref": "current_price", "format": "$,.2f"},
            {"label": "Upside", "ref": "upside", "format": ".1%", "conditional": {"positive": "green", "negative": "red"}}
        ],
        "layout": "horizontal"
    }
)
```

---

### ResearchProject

The top-level container.

```python
@dataclass
class ResearchProject:
    """
    Universal research container - works for any asset class.
    """
    id: str
    name: str
    type: str                       # "equity", "fixed_income", "macro", "sector", "thematic"
    
    # ─── Entities ───
    entities: dict[str, Entity] = field(default_factory=dict)
    primary_entity: str | None = None  # For single-subject reports
    
    # ─── Data Layer ───
    data: dict[str, DataPoint] = field(default_factory=dict)
    
    # ─── Content Layer ───
    content: dict[str, ContentBlock] = field(default_factory=dict)
    
    # ─── Versioning ───
    version: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # ─── Project Metadata ───
    meta: dict[str, Any] = field(default_factory=dict)
    # Examples:
    #   equity:  {"analyst": "J. Smith", "rating": "Overweight", "price_target": 225.0}
    #   macro:   {"forecast_horizon": "12M", "base_date": "2025-01-01"}
    
    # ─── Methods ───
    
    def get_dependents(self, data_id: str) -> list[str]:
        """Find all DataPoints and ContentBlocks that depend on a given DataPoint."""
        dependents = []
        
        # DataPoints that derive from this
        for dp in self.data.values():
            if data_id in dp.derived_from:
                dependents.append(f"data:{dp.id}")
        
        # ContentBlocks that reference this
        for cb in self.content.values():
            if data_id in cb.data_refs:
                dependents.append(f"content:{cb.id}")
        
        return dependents
    
    def get_lineage(self, data_id: str) -> list[str]:
        """Trace a DataPoint back to its root sources (BFS)."""
        visited = set()
        queue = [data_id]
        lineage = []
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            dp = self.data.get(current)
            if dp:
                lineage.append(current)
                queue.extend(dp.derived_from)
        
        return lineage
    
    def validate(self) -> list[str]:
        """Check for broken references and other issues."""
        errors = []
        
        # Check derived_from references exist
        for dp in self.data.values():
            for ref in dp.derived_from:
                if ref not in self.data:
                    errors.append(f"DataPoint '{dp.id}' references missing '{ref}'")
        
        # Check content data_refs exist
        for cb in self.content.values():
            for ref in cb.data_refs:
                if ref not in self.data:
                    errors.append(f"ContentBlock '{cb.id}' references missing '{ref}'")
        
        # Check primary_entity exists
        if self.primary_entity and self.primary_entity not in self.entities:
            errors.append(f"primary_entity '{self.primary_entity}' not in entities")
        
        return errors
```

---

## Full Example: Equity Research Project

```python
from datetime import datetime

project = ResearchProject(
    id="aapl-2025q1",
    name="Apple Inc - Q1 2025 Update",
    type="equity",
    
    # Entity
    entities={
        "aapl": Entity(
            id="aapl",
            type="company", 
            name="Apple Inc",
            identifiers={"ticker": "AAPL", "cik": "0000320193"},
            attributes={"sector": "Technology", "exchange": "NASDAQ"}
        )
    },
    primary_entity="aapl",
    
    # Data
    data={
        "revenue.fy2024": DataPoint(
            id="revenue.fy2024",
            value=96_773_000_000,
            nature=DataNature.OBSERVED,
            source="sec:10-K",
            source_ref="https://sec.gov/...",
            tags=["revenue", "annual", "usd"]
        ),
        "revenue_growth.fy2026": DataPoint(
            id="revenue_growth.fy2026",
            value=0.12,
            nature=DataNature.ASSUMED,
            source="analyst",
            meta={"rationale": "iPhone 17 cycle + Services"},
            alternatives={"bull": 0.15, "bear": 0.08}
        ),
        "price_target": DataPoint(
            id="price_target",
            value=225.0,
            nature=DataNature.DERIVED,
            derived_from=["dcf_value", "comps_value"],
            formula="0.5 * dcf_value + 0.5 * comps_value"
        )
    },
    
    # Content
    content={
        "summary_card": ContentBlock(
            id="summary_card",
            type="metric_card",
            section="header",
            data_refs=["price_target"],
            config={
                "metrics": [
                    {"label": "Rating", "value": "Overweight"},
                    {"label": "Target", "ref": "price_target", "format": "$,.2f"}
                ]
            }
        ),
        "financials_table": ContentBlock(
            id="financials_table",
            type="table",
            title="Financial Summary",
            section="body",
            data_refs=["revenue.fy2024", "revenue.fy2025", "revenue.fy2026e"],
            config={...}
        )
    },
    
    # Metadata
    meta={
        "analyst": "Jane Smith",
        "rating": "Overweight",
        "price_target": 225.0,
        "published": False
    },
    
    version=1,
    created_at=datetime.now()
)
```

---

## Typed Helpers (Optional)

For convenience and type safety, add asset-class-specific wrappers:

```python
class EquityProject(ResearchProject):
    """Typed wrapper for equity research."""
    
    @property
    def ticker(self) -> str:
        entity = self.entities[self.primary_entity]
        return entity.identifiers.get("ticker", "")
    
    @property
    def rating(self) -> str:
        return self.meta.get("rating", "Not Rated")
    
    @property
    def price_target(self) -> float | None:
        return self.meta.get("price_target")
    
    @price_target.setter
    def price_target(self, value: float):
        self.meta["price_target"] = value


class FixedIncomeProject(ResearchProject):
    """Typed wrapper for fixed income research."""
    
    @property
    def spread_target(self) -> float | None:
        return self.meta.get("spread_target")
    
    @property
    def recommendation(self) -> str:
        return self.meta.get("recommendation", "Hold")
```

---

## Serialization

Projects serialize to JSON for storage and version control:

```python
import json
from dataclasses import asdict

def save_project(project: ResearchProject, path: str):
    """Save project to JSON file."""
    data = asdict(project)
    # Convert datetime to ISO strings
    data["created_at"] = project.created_at.isoformat()
    data["updated_at"] = project.updated_at.isoformat()
    
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

def load_project(path: str) -> ResearchProject:
    """Load project from JSON file."""
    with open(path) as f:
        data = json.load(f)
    
    # Reconstruct nested objects
    data["entities"] = {k: Entity(**v) for k, v in data["entities"].items()}
    data["data"] = {k: DataPoint(**v) for k, v in data["data"].items()}
    data["content"] = {k: ContentBlock(**v) for k, v in data["content"].items()}
    data["created_at"] = datetime.fromisoformat(data["created_at"])
    data["updated_at"] = datetime.fromisoformat(data["updated_at"])
    
    return ResearchProject(**data)
```

---

## Directory Structure

```
project/
├── project.json              # ResearchProject serialized
├── data/
│   ├── sec_filings/          # Raw SEC documents
│   ├── market_data/          # Price history, fundamentals
│   ├── transcripts/          # Earnings calls
│   └── external/             # Third-party data
└── output/
    ├── report_v1.pdf
    ├── report_v2.pdf
    └── report_v2.html
```

---

## Key Capabilities

| Feature | Implementation |
|---------|----------------|
| **Hover for source** | `data[ref].source_ref` → display as tooltip |
| **Stale data warning** | Check `data[ref].fetched_at` age |
| **Impact analysis** | `project.get_dependents(data_id)` |
| **Lineage tracing** | `project.get_lineage(data_id)` |
| **Scenario switching** | Swap `DataPoint.alternatives` values |
| **Validation** | `project.validate()` checks references |
| **Version control** | JSON is git-diffable |

---

## Future Extensions

- **Collaboration**: Add `author`, `reviewers`, `comments` fields
- **Permissions**: Track who can edit which DataPoints
- **Audit log**: Record all changes with timestamps
- **Templates**: Pre-built ContentBlock configurations
- **Formulas engine**: Parse and execute `formula` strings
- **Real-time sync**: WebSocket updates when data changes
