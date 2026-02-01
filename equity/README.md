# Equity Research Report Generator

A professional equity research report generator with two output pathways: **PDF** (ReportLab) and **HTML/Markdown** (browser-based).

# System design 
we should design a data structure for equity research report first -- version control, ... , data trustworthy, data source, date, layout position, ...

Phase 1: generate an outline of the report
let's say the user create a project and throw a few queries and questions related to the high level equity research reports setting up the theme of the report, but there should be some must execute like the sec reports three big tables valuations.. and then the ai agent should return some feedbacks based on the high level analysis. it should pull the data from the necessary data sources save all the raw into the project/data folder. something should be tools but something should be hardcoded (pulling data from xx)

Phase 2: the user should be able to provide feedbacks anytime into the agent and the agent should return or responses with the information to do more research. 

Features: 
source and citation should appear when the user hover the mouse on top of a column / number / table 


## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Convert your notes to a professional report
python equity_report.py convert my_notes.md --format both

# Generate a placeholder report
python equity_report.py generate --company "Tesla" --ticker TSLA
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     equity_report.py (CLI)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │   convert   │   │  generate   │   │    tune     │
    │  (MD→Report)│   │ (Template)  │   │ (AI Layout) │
    └─────────────┘   └─────────────┘   └─────────────┘
              │               │               │
              └───────────────┼───────────────┘
                              ▼
              ┌───────────────────────────────┐
              │       ReportBuilder           │
              │   (Unified Build Interface)   │
              └───────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │    PDF      │   │    HTML     │   │  Markdown   │
    │ (ReportLab) │   │ (Standalone)│   │ (md-viewer) │
    └─────────────┘   └─────────────┘   └─────────────┘
```

## Output Pathways

### 1. PDF Pathway (ReportLab)
Professional PDF output with precise layout control.

```bash
# Generate PDF
python equity_report.py convert notes.md --format pdf

# AI-powered layout tuning
python equity_report.py tune output/report.pdf --provider xai
```

**Features:**
- Pixel-perfect layout matching JPM style
- Two-column summary page with sidebar
- Automatic content flowing and pagination
- Tables, charts, and financial data formatting

### 2. HTML/Markdown Pathway
Browser-based viewing and editing with live preview.

```bash
# Generate Markdown for viewer
python equity_report.py convert notes.md --format html

# Open the viewer
open frontend/md-viewer.html
```

**Features:**
- Side-by-side editor and preview
- Two-column (summary) and single-column (body) layouts
- PDF export via html2pdf.js
- Real-time rendering with Marked.js

## Commands

### `convert` - Markdown to Report
Convert raw markdown notes into a structured equity research report.

```bash
python equity_report.py convert <input.md> [options]

Options:
  --format     pdf | html | both (default: both)
  --output-dir Output directory
  --provider   anthropic | openai | xai (default: xai)
  --company    Company name hint for better parsing
```

### `generate` - Template Report
Generate a report with placeholder or custom content.

```bash
python equity_report.py generate [options]

Options:
  --company    Company name (default: [Company])
  --ticker     Stock ticker (default: [TICKER])
  --rating     Rating (default: Overweight)
  --target     Price target (default: $100.00)
  --format     pdf | html | markdown | both
  --output-dir Output directory (default: ./output)
```

### `serve` - Web Editor
Start the FastAPI server for interactive PDF editing.

```bash
python equity_report.py serve [options]

Options:
  --port    Port number (default: 8080)
  --reload  Enable auto-reload for development
```

### `tune` - AI Layout Tuning
Use AI vision to analyze and improve PDF layout.

```bash
python equity_report.py tune <input.pdf> [options]

Options:
  --provider      anthropic | openai | xai (default: xai)
  --priority      density | readability | balanced
  --iterations    Max iterations (default: 5)
  --target-score  Target score 0-100 (default: 85)
  --output-dir    Output directory
```

## Project Structure

```
equity/
├── equity_report.py          # Main CLI entry point
├── requirements.txt          # Python dependencies
│
├── backend/
│   ├── report_builder.py     # Unified report builder
│   ├── layout_params.py      # Layout parameters (shared)
│   ├── content_placeholders.py # Content generation
│   │
│   ├── # PDF Pathway
│   ├── dynamic_report_builder.py  # ReportLab PDF builder
│   ├── pdf_renderer.py       # PDF to PNG rendering
│   ├── pdf_analyzer.py       # PDF layout analysis
│   │
│   ├── # Markdown Pathway
│   ├── md_to_report.py       # MD to structured report
│   │
│   ├── # AI Agents
│   ├── format_agent.py       # AI-powered PDF layout tuning
│   ├── layout_critic.py      # Layout critique engine
│   ├── agent.py              # Deep research agent (market data)
│   │
│   └── goal/                 # Reference PDFs
│       └── JPM-Equity-Research-Report-Hulu.pdf
│
├── frontend/
│   ├── index.html            # PDF editor UI
│   └── md-viewer.html        # Markdown viewer UI
│
└── output/                   # Generated reports
```

## Layout Parameters

Both PDF and HTML use consistent layout parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `page_width` | 612pt | US Letter width |
| `page_height` | 792pt | US Letter height |
| `margin_left` | 54pt | Left margin |
| `margin_right` | 43.2pt | Right margin |
| `margin_top` | 54pt | Top margin |
| `margin_bottom` | 54pt | Bottom margin |
| `sidebar_width` | 162pt | Sidebar width on summary page |
| `gutter_width` | 14.4pt | Gap between main and sidebar |
| `h1_font_size` | 17pt | Main headline |
| `body_font_size` | 9.5pt | Body text |

## Markdown Viewer Format

The `md-viewer.html` supports special markers for two-column layout:

```markdown
---MAIN---
<h1>Report Headline</h1>
<p>Main content...</p>

---SIDEBAR---
<h2>Company Name</h2>
<p>Sidebar content...</p>
---ENDSIDEBAR---
```

## Environment Variables

```bash
# AI Providers (for convert and tune commands)
export XAI_API_KEY=your-xai-key
export ANTHROPIC_API_KEY=your-anthropic-key
export OPENAI_API_KEY=your-openai-key
```

## Examples

### Convert Research Notes to PDF

```bash
# Your research notes
cat > notes.md << 'EOF'
# Tesla Analysis

## Overview
Tesla is the leading EV manufacturer...

## Bull Case
- Market leadership
- Growing energy business
- FSD potential

## Valuation
Current price: $248.50
Target: $300.00
EOF

# Convert to professional report
python equity_report.py convert notes.md --format pdf --company "Tesla"
```

### View in Browser

```bash
# Generate markdown
python equity_report.py convert notes.md --format html

# Open viewer and paste content
open frontend/md-viewer.html
```

## License

MIT
