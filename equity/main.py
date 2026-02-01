"""
Equity Research Report Editor - Interactive PDF Generator
Run with: python main.py
"""

import io
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.dynamic_report_builder import build_dynamic_report, create_sample_dynamic_data
from backend.layout_params import LayoutParams
from backend.content_placeholders import create_jpm_style_placeholders

app = FastAPI(
    title="Equity Research Report Editor",
    description="Interactive PDF report generator with live preview",
    version="2.0.0"
)

# Serve static files
frontend_path = Path(__file__).parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")


# ============================================
# REQUEST MODELS
# ============================================

class LayoutConfig(BaseModel):
    margin_left: float = 54
    margin_right: float = 43
    margin_top: float = 63
    margin_bottom: float = 54
    sidebar_width: float = 162
    gutter_width: float = 14
    h1_font_size: float = 17
    body_font_size: float = 9.5


class PlaceholderConfig(BaseModel):
    key_points: int = 4
    summary_body: int = 3
    exec_summary: int = 5
    industry_text: int = 4
    risk_factors: int = 6


class ReportRequest(BaseModel):
    company_name: str = "[PLACEHOLDER]"
    ticker: str = "[PLACEHOLDER]"
    rating: str = "Overweight"
    price_target: float = 0.0
    layout: Optional[LayoutConfig] = None
    placeholders: Optional[PlaceholderConfig] = None
    content: Optional[Dict[str, Any]] = None


# ============================================
# ENDPOINTS
# ============================================

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the interactive editor UI"""
    html_path = frontend_path / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="""
    <html>
    <head><title>Equity Report Editor</title></head>
    <body style="background:#0d1117;color:#e6edf3;font-family:sans-serif;padding:40px;">
        <h1>Equity Report Editor</h1>
        <p>Frontend not found. Make sure frontend/index.html exists.</p>
        <p>API endpoints available:</p>
        <ul>
            <li>POST /api/preview-pdf - Generate PDF preview</li>
            <li>POST /api/generate-pdf - Download PDF</li>
            <li>GET /api/demo-pdf - Get demo template PDF</li>
        </ul>
    </body>
    </html>
    """)


@app.post("/api/preview-pdf")
async def preview_pdf(request: ReportRequest):
    """Generate PDF and return as binary for preview"""
    try:
        pdf_bytes = generate_pdf_from_request(request)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": "inline; filename=preview.pdf"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-pdf")
async def generate_pdf(request: ReportRequest):
    """Generate PDF and return for download"""
    try:
        pdf_bytes = generate_pdf_from_request(request)
        filename = f"equity_report_{request.ticker}_{datetime.now().strftime('%Y%m%d')}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/demo-pdf")
async def demo_pdf():
    """Generate a demo placeholder PDF"""
    try:
        request = ReportRequest()
        pdf_bytes = generate_pdf_from_request(request)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/layout-params")
async def get_layout_params():
    """Get default layout parameters"""
    params = LayoutParams()
    return {
        "margin_left": params.margin_left,
        "margin_right": params.margin_right,
        "margin_top": params.margin_top,
        "margin_bottom": params.margin_bottom,
        "sidebar_width": params.sidebar_width,
        "gutter_width": params.gutter_width,
        "h1_font_size": params.h1_font_size,
        "h2_font_size": params.h2_font_size,
        "body_font_size": params.body_font_size,
        "small_font_size": params.small_font_size,
        "tiny_font_size": params.tiny_font_size,
    }


@app.get("/api/placeholder-registry")
async def get_placeholder_registry():
    """Get placeholder registry with all available placeholders"""
    registry = create_jpm_style_placeholders()
    return registry.to_manifest()


# ============================================
# PDF GENERATION HELPER
# ============================================

def generate_pdf_from_request(request: ReportRequest) -> bytes:
    """Generate PDF bytes from request data"""
    
    # Build layout params
    params = LayoutParams()
    if request.layout:
        params.margin_left = request.layout.margin_left
        params.margin_right = request.layout.margin_right
        params.margin_top = request.layout.margin_top
        params.margin_bottom = request.layout.margin_bottom
        params.sidebar_width = request.layout.sidebar_width
        params.gutter_width = request.layout.gutter_width
        params.h1_font_size = request.layout.h1_font_size
        params.body_font_size = request.layout.body_font_size
    
    # Build placeholder registry
    registry = create_jpm_style_placeholders()
    if request.placeholders:
        # Update placeholder counts
        if "key_points" in registry.placeholders:
            registry.placeholders["key_points"].current_items = request.placeholders.key_points
        if "summary_body" in registry.placeholders:
            registry.placeholders["summary_body"].current_items = request.placeholders.summary_body
        if "exec_summary" in registry.placeholders:
            registry.placeholders["exec_summary"].current_items = request.placeholders.exec_summary
        if "industry_text" in registry.placeholders:
            registry.placeholders["industry_text"].current_items = request.placeholders.industry_text
        if "risk_factors" in registry.placeholders:
            registry.placeholders["risk_factors"].current_items = request.placeholders.risk_factors
    
    # Build report data
    data = create_sample_dynamic_data()
    data["company"] = request.company_name
    data["ticker"] = request.ticker
    data["rating"] = request.rating
    data["target"] = request.price_target
    
    # Apply any custom content
    if request.content:
        data.update(request.content)
    
    # Generate PDF to temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        build_dynamic_report(data, params, registry, tmp_path)
        with open(tmp_path, "rb") as f:
            pdf_bytes = f.read()
        return pdf_bytes
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("  Equity Research Report Editor")
    print("="*60)
    print("\n  Open http://localhost:8080 in your browser")
    print("  Interactive PDF editor with live preview\n")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8080)
