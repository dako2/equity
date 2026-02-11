"""
Interactive HTML Renderer for Research Data Model
Generates tables with hover-to-source functionality
"""

from data_demo import create_apple_demo, DataNature, ResearchProject, DataPoint
from datetime import datetime
import json
import html as html_lib

def generate_html_report(project: ResearchProject, output_path: str = "report_interactive.html"):
    """Generate an interactive HTML report with hover-to-source."""
    
    # Prepare data for JavaScript
    data_json = {}
    for id, dp in project.data.items():
        data_json[id] = {
            "id": dp.id,
            "value": dp.value,
            "nature": dp.nature.value,
            "source": dp.source,
            "source_ref": dp.source_ref,
            "as_of": dp.as_of.isoformat() if dp.as_of else None,
            "derived_from": dp.derived_from,
            "formula": dp.formula,
            "tags": dp.tags,
            "meta": dp.meta,
            "alternatives": dp.alternatives,
        }
    
    entity = project.entities[project.primary_entity]
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html_lib.escape(project.name)}</title>
    <style>
        :root {{
            --bg-dark: #0a0a0f;
            --bg-card: #12121a;
            --bg-hover: #1a1a2e;
            --border: #2a2a3e;
            --text: #e4e4eb;
            --text-dim: #8888a0;
            --accent: #6366f1;
            --accent-glow: rgba(99, 102, 241, 0.3);
            --green: #10b981;
            --red: #ef4444;
            --yellow: #f59e0b;
            --blue: #3b82f6;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
            background: var(--bg-dark);
            color: var(--text);
            line-height: 1.6;
            padding: 2rem;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        /* Header */
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 2rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid var(--border);
        }}
        
        .header-left h1 {{
            font-size: 1.8rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }}
        
        .ticker {{
            display: inline-block;
            background: var(--accent);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 4px;
            font-size: 0.9rem;
            margin-right: 0.5rem;
        }}
        
        .rating {{
            display: inline-block;
            background: var(--green);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 4px;
            font-size: 0.9rem;
        }}
        
        .header-right {{
            text-align: right;
        }}
        
        .price-target {{
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--green);
        }}
        
        .upside {{
            color: var(--green);
            font-size: 1.1rem;
        }}
        
        /* Stats bar */
        .stats-bar {{
            display: flex;
            gap: 2rem;
            margin-bottom: 2rem;
            padding: 1rem;
            background: var(--bg-card);
            border-radius: 8px;
            border: 1px solid var(--border);
        }}
        
        .stat {{
            display: flex;
            flex-direction: column;
        }}
        
        .stat-label {{
            font-size: 0.75rem;
            color: var(--text-dim);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        .stat-value {{
            font-size: 1.1rem;
            font-weight: 600;
        }}
        
        .stat-value.observed {{ color: var(--blue); }}
        .stat-value.assumed {{ color: var(--yellow); }}
        .stat-value.derived {{ color: var(--accent); }}
        
        /* Tables */
        .table-section {{
            margin-bottom: 2.5rem;
        }}
        
        .table-title {{
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: var(--text);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .table-title::before {{
            content: '';
            display: inline-block;
            width: 4px;
            height: 1.2rem;
            background: var(--accent);
            border-radius: 2px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--bg-card);
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border);
        }}
        
        th {{
            text-align: left;
            padding: 0.75rem 1rem;
            background: var(--bg-hover);
            font-weight: 600;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-dim);
            border-bottom: 1px solid var(--border);
        }}
        
        th:not(:first-child) {{
            text-align: right;
        }}
        
        td {{
            padding: 0.6rem 1rem;
            border-bottom: 1px solid var(--border);
            font-size: 0.9rem;
        }}
        
        td:not(:first-child) {{
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}
        
        tr:last-child td {{
            border-bottom: none;
        }}
        
        tr:hover {{
            background: var(--bg-hover);
        }}
        
        /* Data cells with source */
        .data-cell {{
            position: relative;
            cursor: help;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            transition: all 0.15s ease;
        }}
        
        .data-cell:hover {{
            background: var(--accent-glow);
        }}
        
        .data-cell.observed {{
            border-left: 2px solid var(--blue);
        }}
        
        .data-cell.assumed {{
            border-left: 2px solid var(--yellow);
        }}
        
        .data-cell.derived {{
            border-left: 2px solid var(--accent);
            font-style: italic;
        }}
        
        .data-cell.projection {{
            background: rgba(99, 102, 241, 0.1);
        }}
        
        /* Tooltip */
        .tooltip {{
            position: fixed;
            z-index: 1000;
            background: #1e1e2e;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
            min-width: 320px;
            max-width: 420px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
            pointer-events: none;
            opacity: 0;
            transform: translateY(8px);
            transition: opacity 0.15s ease, transform 0.15s ease;
        }}
        
        .tooltip.visible {{
            opacity: 1;
            transform: translateY(0);
        }}
        
        .tooltip-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 0.75rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--border);
        }}
        
        .tooltip-id {{
            font-weight: 600;
            font-size: 0.95rem;
            color: var(--accent);
        }}
        
        .tooltip-nature {{
            font-size: 0.75rem;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            text-transform: uppercase;
            font-weight: 600;
        }}
        
        .tooltip-nature.observed {{
            background: rgba(59, 130, 246, 0.2);
            color: var(--blue);
        }}
        
        .tooltip-nature.assumed {{
            background: rgba(245, 158, 11, 0.2);
            color: var(--yellow);
        }}
        
        .tooltip-nature.derived {{
            background: rgba(99, 102, 241, 0.2);
            color: var(--accent);
        }}
        
        .tooltip-row {{
            display: flex;
            margin-bottom: 0.5rem;
            font-size: 0.85rem;
        }}
        
        .tooltip-label {{
            color: var(--text-dim);
            min-width: 80px;
        }}
        
        .tooltip-value {{
            color: var(--text);
            flex: 1;
        }}
        
        .tooltip-value a {{
            color: var(--blue);
            text-decoration: none;
            word-break: break-all;
        }}
        
        .tooltip-value a:hover {{
            text-decoration: underline;
        }}
        
        .tooltip-formula {{
            background: var(--bg-dark);
            padding: 0.5rem 0.75rem;
            border-radius: 4px;
            font-family: 'SF Mono', monospace;
            font-size: 0.8rem;
            color: var(--accent);
            margin-top: 0.5rem;
        }}
        
        .tooltip-scenarios {{
            display: flex;
            gap: 1rem;
            margin-top: 0.75rem;
            padding-top: 0.75rem;
            border-top: 1px solid var(--border);
        }}
        
        .scenario {{
            flex: 1;
            text-align: center;
            padding: 0.5rem;
            border-radius: 4px;
        }}
        
        .scenario.bull {{
            background: rgba(16, 185, 129, 0.1);
        }}
        
        .scenario.bear {{
            background: rgba(239, 68, 68, 0.1);
        }}
        
        .scenario-label {{
            font-size: 0.7rem;
            text-transform: uppercase;
            color: var(--text-dim);
        }}
        
        .scenario-value {{
            font-weight: 600;
            font-size: 0.95rem;
        }}
        
        .scenario.bull .scenario-value {{ color: var(--green); }}
        .scenario.bear .scenario-value {{ color: var(--red); }}
        
        .tooltip-deps {{
            margin-top: 0.5rem;
        }}
        
        .dep-item {{
            font-size: 0.8rem;
            color: var(--text-dim);
            padding: 0.2rem 0;
        }}
        
        .dep-item::before {{
            content: '← ';
            color: var(--accent);
        }}
        
        /* Legend */
        .legend {{
            display: flex;
            gap: 1.5rem;
            margin-bottom: 1.5rem;
            font-size: 0.8rem;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .legend-color {{
            width: 12px;
            height: 12px;
            border-radius: 2px;
        }}
        
        .legend-color.observed {{ background: var(--blue); }}
        .legend-color.assumed {{ background: var(--yellow); }}
        .legend-color.derived {{ background: var(--accent); }}
        
        /* Row styles */
        .row-header td {{
            font-weight: 600;
            background: var(--bg-hover);
            color: var(--text-dim);
            text-transform: uppercase;
            font-size: 0.8rem;
            letter-spacing: 0.05em;
        }}
        
        .row-subtotal td {{
            font-weight: 600;
            border-top: 1px solid var(--border);
        }}
        
        .row-total td {{
            font-weight: 700;
            background: var(--bg-hover);
            border-top: 2px solid var(--border);
        }}
        
        .row-highlight td {{
            background: rgba(99, 102, 241, 0.1);
        }}
        
        /* Footer */
        .table-footer {{
            font-size: 0.75rem;
            color: var(--text-dim);
            margin-top: 0.5rem;
            font-style: italic;
        }}
        
        /* Grid layout for tables */
        .tables-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
        }}
        
        @media (max-width: 1200px) {{
            .tables-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        
        .full-width {{
            grid-column: 1 / -1;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div class="header-left">
                <h1>{html_lib.escape(entity.name)}</h1>
                <span class="ticker">{html_lib.escape(entity.identifiers.get('ticker', ''))}</span>
                <span class="rating">{html_lib.escape(project.meta.get('rating', 'Not Rated'))}</span>
            </div>
            <div class="header-right">
                <div class="price-target">${project.meta.get('price_target', 0):.2f}</div>
                <div class="upside">▲ {((project.meta.get('price_target', 0) / project.meta.get('current_price', 1) - 1) * 100):.1f}% upside</div>
            </div>
        </div>
        
        <!-- Stats -->
        <div class="stats-bar">
            <div class="stat">
                <span class="stat-label">Data Points</span>
                <span class="stat-value">{len(project.data)}</span>
            </div>
            <div class="stat">
                <span class="stat-label">Observed (Historical)</span>
                <span class="stat-value observed">{sum(1 for d in project.data.values() if d.nature == DataNature.OBSERVED)}</span>
            </div>
            <div class="stat">
                <span class="stat-label">Assumed (Inputs)</span>
                <span class="stat-value assumed">{sum(1 for d in project.data.values() if d.nature == DataNature.ASSUMED)}</span>
            </div>
            <div class="stat">
                <span class="stat-label">Derived (Projections)</span>
                <span class="stat-value derived">{sum(1 for d in project.data.values() if d.nature == DataNature.DERIVED)}</span>
            </div>
        </div>
        
        <!-- Legend -->
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color observed"></div>
                <span>Observed (SEC filings, market data)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color assumed"></div>
                <span>Assumed (analyst inputs)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color derived"></div>
                <span>Derived (calculated projections)</span>
            </div>
            <span style="margin-left: auto; color: var(--text-dim);">Hover any value to see source →</span>
        </div>
        
        <!-- Tables -->
        <div class="tables-grid">
            {generate_income_statement_html(project)}
            {generate_valuation_html(project)}
            {generate_balance_sheet_html(project)}
            {generate_cash_flow_html(project)}
        </div>
    </div>
    
    <!-- Tooltip -->
    <div class="tooltip" id="tooltip"></div>
    
    <script>
        // Data store
        const DATA = {json.dumps(data_json, default=str)};
        
        // Tooltip element
        const tooltip = document.getElementById('tooltip');
        
        // Format value for display
        function formatValue(value, dp) {{
            if (typeof value === 'number') {{
                if (Math.abs(value) >= 1000) {{
                    return '$' + value.toLocaleString('en-US', {{maximumFractionDigits: 0}});
                }} else if (Math.abs(value) < 1) {{
                    return (value * 100).toFixed(1) + '%';
                }} else {{
                    return value.toFixed(2);
                }}
            }}
            return value;
        }}
        
        // Generate source URL display
        function getSourceDisplay(dp) {{
            if (!dp.source) return 'Unknown';
            
            const sourceMap = {{
                'sec:10-K': 'SEC 10-K Annual Report',
                'sec:10-Q': 'SEC 10-Q Quarterly Report',
                'market': 'Market Data',
                'analyst': 'Analyst Estimate',
            }};
            
            let display = sourceMap[dp.source] || dp.source;
            
            if (dp.source_ref) {{
                display = `<a href="${{dp.source_ref}}" target="_blank">${{display}} ↗</a>`;
            }}
            
            return display;
        }}
        
        // Show tooltip
        function showTooltip(event, dataId) {{
            const dp = DATA[dataId];
            if (!dp) return;
            
            let html = `
                <div class="tooltip-header">
                    <span class="tooltip-id">${{dp.id}}</span>
                    <span class="tooltip-nature ${{dp.nature}}">${{dp.nature}}</span>
                </div>
                <div class="tooltip-row">
                    <span class="tooltip-label">Value</span>
                    <span class="tooltip-value">${{formatValue(dp.value, dp)}}</span>
                </div>
            `;
            
            if (dp.as_of) {{
                html += `
                    <div class="tooltip-row">
                        <span class="tooltip-label">As of</span>
                        <span class="tooltip-value">${{new Date(dp.as_of).toLocaleDateString()}}</span>
                    </div>
                `;
            }}
            
            html += `
                <div class="tooltip-row">
                    <span class="tooltip-label">Source</span>
                    <span class="tooltip-value">${{getSourceDisplay(dp)}}</span>
                </div>
            `;
            
            if (dp.meta && dp.meta.rationale) {{
                html += `
                    <div class="tooltip-row">
                        <span class="tooltip-label">Rationale</span>
                        <span class="tooltip-value">${{dp.meta.rationale}}</span>
                    </div>
                `;
            }}
            
            if (dp.formula) {{
                html += `<div class="tooltip-formula">ƒ ${{dp.formula}}</div>`;
            }}
            
            if (dp.derived_from && dp.derived_from.length > 0) {{
                html += `<div class="tooltip-deps">`;
                dp.derived_from.forEach(ref => {{
                    const parent = DATA[ref];
                    if (parent) {{
                        html += `<div class="dep-item">${{ref}}: ${{formatValue(parent.value)}} (${{parent.nature}})</div>`;
                    }}
                }});
                html += `</div>`;
            }}
            
            if (dp.alternatives && (dp.alternatives.bull || dp.alternatives.bear)) {{
                html += `
                    <div class="tooltip-scenarios">
                        <div class="scenario bull">
                            <div class="scenario-label">Bull Case</div>
                            <div class="scenario-value">${{dp.alternatives.bull ? formatValue(dp.alternatives.bull) : '—'}}</div>
                        </div>
                        <div class="scenario bear">
                            <div class="scenario-label">Bear Case</div>
                            <div class="scenario-value">${{dp.alternatives.bear ? formatValue(dp.alternatives.bear) : '—'}}</div>
                        </div>
                    </div>
                `;
            }}
            
            tooltip.innerHTML = html;
            
            // Position tooltip
            const rect = event.target.getBoundingClientRect();
            let left = rect.right + 10;
            let top = rect.top;
            
            // Adjust if would go off screen
            if (left + 400 > window.innerWidth) {{
                left = rect.left - 340;
            }}
            if (top + 300 > window.innerHeight) {{
                top = window.innerHeight - 320;
            }}
            
            tooltip.style.left = left + 'px';
            tooltip.style.top = top + 'px';
            tooltip.classList.add('visible');
        }}
        
        // Hide tooltip
        function hideTooltip() {{
            tooltip.classList.remove('visible');
        }}
        
        // Attach event listeners
        document.querySelectorAll('.data-cell').forEach(cell => {{
            cell.addEventListener('mouseenter', (e) => showTooltip(e, cell.dataset.ref));
            cell.addEventListener('mouseleave', hideTooltip);
        }});
    </script>
</body>
</html>
'''
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"✅ Generated interactive report: {output_path}")
    return output_path


def data_cell(project: ResearchProject, ref: str, display: str = None) -> str:
    """Generate a data cell with hover capability."""
    dp = project.data.get(ref)
    if not dp:
        return display or ref
    
    nature = dp.nature.value
    value = dp.value
    
    if display is None:
        if isinstance(value, float):
            if abs(value) >= 1000:
                display = f"${value:,.0f}"
            elif abs(value) < 0:
                display = f"({abs(value):,.0f})"
            elif abs(value) < 1:
                display = f"{value * 100:.1f}%"
            else:
                display = f"{value:.2f}"
        else:
            display = str(value)
    
    projection = "projection" if "fy26" in ref.lower() or "26e" in ref.lower() else ""
    
    return f'<span class="data-cell {nature} {projection}" data-ref="{ref}">{display}</span>'


def generate_income_statement_html(project: ResearchProject) -> str:
    """Generate Income Statement table HTML."""
    return f'''
        <div class="table-section full-width">
            <div class="table-title">Income Statement</div>
            <table>
                <thead>
                    <tr>
                        <th style="width: 200px;">($M)</th>
                        <th>FY24</th>
                        <th>FY25</th>
                        <th>FY26E</th>
                        <th>YoY %</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Revenue</td>
                        <td>{data_cell(project, "revenue.fy24")}</td>
                        <td>{data_cell(project, "revenue.fy25")}</td>
                        <td>{data_cell(project, "revenue.fy26e")}</td>
                        <td>8.0%</td>
                    </tr>
                    <tr>
                        <td>Cost of Revenue</td>
                        <td>{data_cell(project, "cost_of_revenue.fy24")}</td>
                        <td>{data_cell(project, "cost_of_revenue.fy25")}</td>
                        <td>{data_cell(project, "cost_of_revenue.fy26e")}</td>
                        <td>8.6%</td>
                    </tr>
                    <tr class="row-subtotal">
                        <td>Gross Profit</td>
                        <td>{data_cell(project, "gross_profit.fy24")}</td>
                        <td>{data_cell(project, "gross_profit.fy25")}</td>
                        <td>{data_cell(project, "gross_profit.fy26e")}</td>
                        <td>7.4%</td>
                    </tr>
                    <tr>
                        <td style="color: var(--text-dim);">Gross Margin</td>
                        <td style="color: var(--text-dim);">46.2%</td>
                        <td style="color: var(--text-dim);">46.8%</td>
                        <td style="color: var(--text-dim);">46.5%</td>
                        <td style="color: var(--text-dim);">-30bps</td>
                    </tr>
                    <tr>
                        <td>R&D Expense</td>
                        <td>{data_cell(project, "rd_expense.fy24")}</td>
                        <td>{data_cell(project, "rd_expense.fy25")}</td>
                        <td>$36,500</td>
                        <td>7.7%</td>
                    </tr>
                    <tr>
                        <td>SG&A Expense</td>
                        <td>{data_cell(project, "sga_expense.fy24")}</td>
                        <td>{data_cell(project, "sga_expense.fy25")}</td>
                        <td>$23,364</td>
                        <td>-14.0%</td>
                    </tr>
                    <tr class="row-subtotal">
                        <td>Operating Income</td>
                        <td>{data_cell(project, "operating_income.fy24")}</td>
                        <td>{data_cell(project, "operating_income.fy25")}</td>
                        <td>{data_cell(project, "operating_income.fy26e")}</td>
                        <td>11.7%</td>
                    </tr>
                    <tr>
                        <td style="color: var(--text-dim);">Operating Margin</td>
                        <td style="color: var(--text-dim);">31.5%</td>
                        <td style="color: var(--text-dim);">31.9%</td>
                        <td style="color: var(--text-dim);">33.0%</td>
                        <td style="color: var(--text-dim);">+110bps</td>
                    </tr>
                    <tr>
                        <td>Pretax Income</td>
                        <td>{data_cell(project, "pretax_income.fy24")}</td>
                        <td>{data_cell(project, "pretax_income.fy25")}</td>
                        <td>{data_cell(project, "pretax_income.fy26e")}</td>
                        <td>11.2%</td>
                    </tr>
                    <tr class="row-total">
                        <td>Net Income</td>
                        <td>{data_cell(project, "net_income.fy24")}</td>
                        <td>{data_cell(project, "net_income.fy25")}</td>
                        <td>{data_cell(project, "net_income.fy26e")}</td>
                        <td>11.2%</td>
                    </tr>
                    <tr class="row-highlight">
                        <td>EPS (Diluted)</td>
                        <td>{data_cell(project, "eps_diluted.fy24", "$7.00")}</td>
                        <td>{data_cell(project, "eps_diluted.fy25", "$7.52")}</td>
                        <td>{data_cell(project, "eps_diluted.fy26e", "$8.41")}</td>
                        <td>11.8%</td>
                    </tr>
                </tbody>
            </table>
            <div class="table-footer">Source: Company SEC filings (10-K), Analyst estimates for FY26E</div>
        </div>
    '''


def generate_balance_sheet_html(project: ResearchProject) -> str:
    """Generate Balance Sheet table HTML."""
    return f'''
        <div class="table-section">
            <div class="table-title">Balance Sheet</div>
            <table>
                <thead>
                    <tr>
                        <th style="width: 180px;">($M)</th>
                        <th>FY24</th>
                        <th>FY25</th>
                        <th>FY26E</th>
                    </tr>
                </thead>
                <tbody>
                    <tr class="row-header">
                        <td colspan="4">ASSETS</td>
                    </tr>
                    <tr>
                        <td>Cash & Equivalents</td>
                        <td>{data_cell(project, "cash.fy24")}</td>
                        <td>{data_cell(project, "cash.fy25")}</td>
                        <td>{data_cell(project, "cash.fy26e")}</td>
                    </tr>
                    <tr>
                        <td>Short-term Inv.</td>
                        <td>{data_cell(project, "short_term_inv.fy24")}</td>
                        <td>{data_cell(project, "short_term_inv.fy25")}</td>
                        <td>$35,000</td>
                    </tr>
                    <tr>
                        <td>Accounts Receivable</td>
                        <td>{data_cell(project, "accounts_receivable.fy24")}</td>
                        <td>{data_cell(project, "accounts_receivable.fy25")}</td>
                        <td>$38,000</td>
                    </tr>
                    <tr class="row-subtotal">
                        <td>Total Current Assets</td>
                        <td>{data_cell(project, "total_current_assets.fy24")}</td>
                        <td>{data_cell(project, "total_current_assets.fy25")}</td>
                        <td>{data_cell(project, "total_current_assets.fy26e")}</td>
                    </tr>
                    <tr>
                        <td>PP&E (Net)</td>
                        <td>{data_cell(project, "ppe_net.fy24")}</td>
                        <td>{data_cell(project, "ppe_net.fy25")}</td>
                        <td>{data_cell(project, "ppe_net.fy26e")}</td>
                    </tr>
                    <tr class="row-total">
                        <td>Total Assets</td>
                        <td>{data_cell(project, "total_assets.fy24")}</td>
                        <td>{data_cell(project, "total_assets.fy25")}</td>
                        <td>{data_cell(project, "total_assets.fy26e")}</td>
                    </tr>
                    <tr class="row-header">
                        <td colspan="4">LIABILITIES & EQUITY</td>
                    </tr>
                    <tr>
                        <td>Long-term Debt</td>
                        <td>{data_cell(project, "long_term_debt.fy24")}</td>
                        <td>{data_cell(project, "long_term_debt.fy25")}</td>
                        <td>{data_cell(project, "long_term_debt.fy26e")}</td>
                    </tr>
                    <tr class="row-subtotal">
                        <td>Total Liabilities</td>
                        <td>{data_cell(project, "total_liabilities.fy24")}</td>
                        <td>{data_cell(project, "total_liabilities.fy25")}</td>
                        <td>{data_cell(project, "total_liabilities.fy26e")}</td>
                    </tr>
                    <tr class="row-total">
                        <td>Total Equity</td>
                        <td>{data_cell(project, "total_equity.fy24")}</td>
                        <td>{data_cell(project, "total_equity.fy25")}</td>
                        <td>{data_cell(project, "total_equity.fy26e")}</td>
                    </tr>
                </tbody>
            </table>
            <div class="table-footer">Source: SEC 10-K filings</div>
        </div>
    '''


def generate_cash_flow_html(project: ResearchProject) -> str:
    """Generate Cash Flow table HTML."""
    return f'''
        <div class="table-section">
            <div class="table-title">Cash Flow Statement</div>
            <table>
                <thead>
                    <tr>
                        <th style="width: 180px;">($M)</th>
                        <th>FY24</th>
                        <th>FY25</th>
                        <th>FY26E</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Cash from Operations</td>
                        <td>{data_cell(project, "cfo.fy24")}</td>
                        <td>{data_cell(project, "cfo.fy25")}</td>
                        <td>{data_cell(project, "cfo.fy26e")}</td>
                    </tr>
                    <tr>
                        <td>Capital Expenditures</td>
                        <td>{data_cell(project, "capex.fy24")}</td>
                        <td>{data_cell(project, "capex.fy25")}</td>
                        <td>{data_cell(project, "capex.fy26e")}</td>
                    </tr>
                    <tr class="row-highlight">
                        <td>Free Cash Flow</td>
                        <td>{data_cell(project, "fcf.fy24")}</td>
                        <td>{data_cell(project, "fcf.fy25")}</td>
                        <td>{data_cell(project, "fcf.fy26e")}</td>
                    </tr>
                    <tr>
                        <td style="color: var(--text-dim);">FCF Margin</td>
                        <td style="color: var(--text-dim);">27.8%</td>
                        <td style="color: var(--text-dim);">28.1%</td>
                        <td style="color: var(--text-dim);">28.4%</td>
                    </tr>
                    <tr>
                        <td>Depreciation</td>
                        <td>{data_cell(project, "depreciation.fy24")}</td>
                        <td>{data_cell(project, "depreciation.fy25")}</td>
                        <td>{data_cell(project, "depreciation.fy26e")}</td>
                    </tr>
                    <tr>
                        <td>Dividends Paid</td>
                        <td>{data_cell(project, "dividends.fy24")}</td>
                        <td>{data_cell(project, "dividends.fy25")}</td>
                        <td>{data_cell(project, "dividends.fy26e")}</td>
                    </tr>
                    <tr>
                        <td>Share Buybacks</td>
                        <td>{data_cell(project, "buybacks.fy24")}</td>
                        <td>{data_cell(project, "buybacks.fy25")}</td>
                        <td>{data_cell(project, "buybacks.fy26e")}</td>
                    </tr>
                    <tr class="row-total">
                        <td>Total Capital Return</td>
                        <td>($110,025)</td>
                        <td>($105,400)</td>
                        <td>($111,016)</td>
                    </tr>
                </tbody>
            </table>
            <div class="table-footer">Source: SEC 10-K filings</div>
        </div>
    '''


def generate_valuation_html(project: ResearchProject) -> str:
    """Generate Valuation table HTML."""
    return f'''
        <div class="table-section">
            <div class="table-title">Valuation Summary</div>
            <table>
                <thead>
                    <tr>
                        <th style="width: 180px;"></th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Share Price</td>
                        <td>{data_cell(project, "share_price", "$227.50")}</td>
                    </tr>
                    <tr>
                        <td>Shares Outstanding</td>
                        <td>{data_cell(project, "shares_outstanding", "15,005M")}</td>
                    </tr>
                    <tr>
                        <td>Market Cap</td>
                        <td>{data_cell(project, "market_cap", "$3.41T")}</td>
                    </tr>
                    <tr>
                        <td>Enterprise Value</td>
                        <td>{data_cell(project, "enterprise_value", "$3.44T")}</td>
                    </tr>
                    <tr class="row-header">
                        <td colspan="2">MULTIPLES (FY25 / FY26E)</td>
                    </tr>
                    <tr>
                        <td>P/E</td>
                        <td>{data_cell(project, "pe_fy25", "30.3x")} / {data_cell(project, "pe_fy26e", "27.1x")}</td>
                    </tr>
                    <tr>
                        <td>EV/EBITDA</td>
                        <td>{data_cell(project, "ev_ebitda_fy25", "24.0x")} / {data_cell(project, "ev_ebitda_fy26e", "21.6x")}</td>
                    </tr>
                    <tr>
                        <td>EV/Revenue</td>
                        <td>{data_cell(project, "ev_revenue_fy25", "8.4x")} / {data_cell(project, "ev_revenue_fy26e", "7.8x")}</td>
                    </tr>
                    <tr>
                        <td>FCF Yield</td>
                        <td>{data_cell(project, "fcf_yield_fy25", "3.4%")} / {data_cell(project, "fcf_yield_fy26e", "3.7%")}</td>
                    </tr>
                    <tr class="row-header">
                        <td colspan="2">DCF ASSUMPTIONS</td>
                    </tr>
                    <tr>
                        <td>WACC</td>
                        <td>{data_cell(project, "wacc", "9.2%")}</td>
                    </tr>
                    <tr>
                        <td>Terminal Growth</td>
                        <td>{data_cell(project, "terminal_growth", "2.5%")}</td>
                    </tr>
                    <tr class="row-highlight">
                        <td><strong>Price Target</strong></td>
                        <td><strong>{data_cell(project, "price_target", "$250.00")}</strong></td>
                    </tr>
                    <tr>
                        <td>Upside</td>
                        <td style="color: var(--green);">{data_cell(project, "upside", "+9.9%")}</td>
                    </tr>
                    <tr>
                        <td>Bull Case</td>
                        <td style="color: var(--green);">{data_cell(project, "dcf_value", "$285.00")}</td>
                    </tr>
                    <tr>
                        <td>Bear Case</td>
                        <td style="color: var(--red);">$205.00</td>
                    </tr>
                </tbody>
            </table>
        </div>
    '''


if __name__ == "__main__":
    # Create demo project
    project = create_apple_demo()
    
    # Add real SEC source URLs to the data points
    sec_base = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193&type=10-K"
    
    for key, dp in project.data.items():
        if dp.source == "sec:10-K":
            dp.source_ref = sec_base
        elif dp.source == "market":
            dp.source_ref = "https://finance.yahoo.com/quote/AAPL"
    
    # Generate the HTML report
    output_path = "report_interactive.html"
    generate_html_report(project, output_path)
    
    print(f"\n🌐 Open in browser: file://{__import__('os').path.abspath(output_path)}")
