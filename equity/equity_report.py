#!/usr/bin/env python3
"""
Equity Research Report Generator

Unified CLI for generating equity research reports in PDF or HTML format.

Usage:
    # From Markdown (AI-powered restructuring)
    python equity_report.py convert my_notes.md --format pdf
    
    # Generate placeholder report
    python equity_report.py generate --company "Tesla" --ticker TSLA --format both
    
    # View in browser (HTML)
    python equity_report.py serve --port 8080
    
    # Run format agent (AI-powered layout tuning)
    python equity_report.py tune my_report.pdf --provider xai
"""

import argparse
import sys
import os
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.report_builder import ReportBuilder, ReportContent, OutputFormat
from backend.layout_params import LayoutParams


def cmd_convert(args):
    """Convert a Markdown file to equity research report."""
    from backend.md_to_report import MarkdownToReportConverter
    
    if not os.path.exists(args.input):
        print(f"❌ File not found: {args.input}")
        return 1
    
    converter = MarkdownToReportConverter(provider=args.provider)
    
    result = converter.convert(
        md_path=args.input,
        output_format=args.format,
        output_dir=args.output_dir,
        company_hint=args.company
    )
    
    print("\n📂 Generated Files:")
    for fmt, path in result["outputs"].items():
        print(f"   {fmt.upper()}: {path}")
    
    return 0


def cmd_generate(args):
    """Generate a report with placeholder or custom content."""
    
    # Build content
    content = ReportContent(
        company_name=args.company,
        ticker=args.ticker,
        rating=args.rating,
        price_target=args.target,
        headline=f"{args.company} Investment Report",
        key_points=[
            "Strong market position with competitive advantages",
            "Robust financial performance and cash generation",
            "Multiple growth drivers in core and adjacent markets",
            "Attractive risk/reward at current valuation levels"
        ],
        sections=[
            {
                "title": "Executive Summary",
                "content": f"We initiate coverage of {args.company} ({args.ticker}) with an {args.rating} rating. "
                          f"The company demonstrates strong fundamentals and is well-positioned for growth.",
                "bullets": []
            },
            {
                "title": "Investment Thesis",
                "content": "Our constructive view is supported by the company's market leadership, "
                          "proven business model, and multiple growth catalysts.",
                "bullets": [
                    "Market leading position with strong brand",
                    "Demonstrated operating leverage",
                    "Expansion into high-growth adjacencies"
                ]
            },
            {
                "title": "Valuation",
                "content": f"Our {args.target} price target is based on a blended DCF and comparable analysis.",
                "bullets": []
            },
            {
                "title": "Investment Risks",
                "content": "Key risks to our thesis include:",
                "bullets": [
                    "Increased competition and margin pressure",
                    "Macroeconomic headwinds",
                    "Execution risk on growth initiatives"
                ]
            }
        ],
        sidebar_data={
            "key_metrics": {
                "Market Cap": "$XXB",
                "P/E (FY25E)": "XX.Xx",
                "EV/EBITDA": "XX.Xx"
            }
        }
    )
    
    # Determine output paths
    output_dir = args.output_dir or "."
    os.makedirs(output_dir, exist_ok=True)
    base_name = f"{args.ticker.lower()}_report"
    
    outputs = []
    
    if args.format in ["pdf", "both"]:
        pdf_path = os.path.join(output_dir, f"{base_name}.pdf")
        builder = ReportBuilder(output_format=OutputFormat.PDF)
        builder.build(content, pdf_path)
        outputs.append(("PDF", pdf_path))
        print(f"✅ Generated PDF: {pdf_path}")
    
    if args.format in ["html", "both"]:
        html_path = os.path.join(output_dir, f"{base_name}.html")
        builder = ReportBuilder(output_format=OutputFormat.HTML)
        builder.build(content, html_path)
        outputs.append(("HTML", html_path))
        print(f"✅ Generated HTML: {html_path}")
    
    if args.format in ["markdown", "both"]:
        md_path = os.path.join(output_dir, f"{base_name}.md")
        builder = ReportBuilder(output_format=OutputFormat.MARKDOWN)
        builder.build(content, md_path)
        outputs.append(("Markdown", md_path))
        print(f"✅ Generated Markdown: {md_path}")
    
    # Save JSON data
    json_path = os.path.join(output_dir, f"{base_name}_data.json")
    with open(json_path, "w") as f:
        json.dump(content.to_dict(), f, indent=2)
    outputs.append(("JSON", json_path))
    
    print(f"\n📂 Output Files:")
    for fmt, path in outputs:
        print(f"   {fmt}: {path}")
    
    return 0


def cmd_serve(args):
    """Start the web server for interactive editing."""
    import uvicorn
    
    print("\n" + "="*60)
    print("  Equity Research Report Editor")
    print("="*60)
    print(f"\n  Open http://localhost:{args.port} in your browser")
    print("  PDF Editor: http://localhost:{args.port}/")
    print(f"  MD Viewer: Open frontend/md-viewer.html directly")
    print("\n" + "="*60 + "\n")
    
    uvicorn.run("main:app", host="0.0.0.0", port=args.port, reload=args.reload)
    return 0


def cmd_tune(args):
    """Run the AI format agent to tune layout."""
    from backend.format_agent import FormatAgent, DesignSessionConfig, DesignPriority
    
    if not os.path.exists(args.input):
        print(f"❌ File not found: {args.input}")
        return 1
    
    priority = DesignPriority(args.priority)
    
    config = DesignSessionConfig(
        max_iterations=args.iterations,
        target_score=args.target_score,
        llm_provider=args.provider,
        design_priority=priority
    )
    
    agent = FormatAgent(config)
    result = agent.run_session(pdf_path=args.input, output_dir=args.output_dir)
    
    print(f"\n📊 Final Score: {result.final_score}/100")
    print(f"📄 Final PDF: {result.final_pdf_path}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Equity Research Report Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert markdown notes to PDF
    python equity_report.py convert notes.md --format pdf
    
    # Generate a placeholder report for Tesla
    python equity_report.py generate --company "Tesla" --ticker TSLA
    
    # Start the web editor
    python equity_report.py serve --port 8080
    
    # Tune PDF layout with AI
    python equity_report.py tune report.pdf --provider xai
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Convert command
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert Markdown to equity research report"
    )
    convert_parser.add_argument("input", help="Input Markdown file")
    convert_parser.add_argument("--format", choices=["pdf", "html", "both"], default="both")
    convert_parser.add_argument("--output-dir", type=str, default=None)
    convert_parser.add_argument("--provider", choices=["anthropic", "openai", "xai"], default="xai")
    convert_parser.add_argument("--company", type=str, help="Company name hint")
    convert_parser.set_defaults(func=cmd_convert)
    
    # Generate command
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate a report with placeholder content"
    )
    gen_parser.add_argument("--company", type=str, default="[Company]")
    gen_parser.add_argument("--ticker", type=str, default="[TICKER]")
    gen_parser.add_argument("--rating", type=str, default="Overweight")
    gen_parser.add_argument("--target", type=str, default="$100.00")
    gen_parser.add_argument("--format", choices=["pdf", "html", "markdown", "both"], default="both")
    gen_parser.add_argument("--output-dir", type=str, default="./output")
    gen_parser.set_defaults(func=cmd_generate)
    
    # Serve command
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the web editor"
    )
    serve_parser.add_argument("--port", type=int, default=8080)
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    serve_parser.set_defaults(func=cmd_serve)
    
    # Tune command
    tune_parser = subparsers.add_parser(
        "tune",
        help="AI-powered layout tuning"
    )
    tune_parser.add_argument("input", help="Input PDF file")
    tune_parser.add_argument("--provider", choices=["anthropic", "openai", "xai"], default="xai")
    tune_parser.add_argument("--priority", choices=["density", "readability", "balanced"], default="balanced")
    tune_parser.add_argument("--iterations", type=int, default=5)
    tune_parser.add_argument("--target-score", type=float, default=85.0)
    tune_parser.add_argument("--output-dir", type=str, default="./design_sessions")
    tune_parser.set_defaults(func=cmd_tune)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
