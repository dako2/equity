"""
PDF to PNG Renderer for Visual Feedback Loop

Renders PDF pages to high-quality PNG images for LLM visual analysis.
"""

import os
from typing import List, Tuple, Optional
from pathlib import Path

# Try to import PyMuPDF (fitz)
try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False
    print("⚠️  PyMuPDF not available. Install with: pip install PyMuPDF")


def render_pdf_to_png(
    pdf_path: str,
    output_dir: str,
    pages: Optional[List[int]] = None,
    zoom: float = 2.0,
    prefix: str = "page"
) -> List[str]:
    """
    Render PDF pages to PNG images.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save PNG files
        pages: List of page indices (0-based). None = all pages.
        zoom: Zoom factor for resolution (2.0 = 144 DPI)
        prefix: Filename prefix
    
    Returns:
        List of paths to generated PNG files
    """
    if not FITZ_AVAILABLE:
        raise ImportError("PyMuPDF (fitz) is required. Install with: pip install PyMuPDF")
    
    os.makedirs(output_dir, exist_ok=True)
    
    doc = fitz.open(pdf_path)
    
    if pages is None:
        pages = list(range(len(doc)))
    
    png_paths = []
    matrix = fitz.Matrix(zoom, zoom)
    
    for page_idx in pages:
        if page_idx >= len(doc):
            continue
        
        page = doc.load_page(page_idx)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        
        output_path = os.path.join(output_dir, f"{prefix}_{page_idx + 1}.png")
        pixmap.save(output_path)
        png_paths.append(output_path)
    
    doc.close()
    return png_paths


def get_page_dimensions(pdf_path: str, page_idx: int = 0) -> Tuple[float, float]:
    """Get the dimensions of a PDF page in points."""
    if not FITZ_AVAILABLE:
        return (612.0, 792.0)  # Default to letter size
    
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_idx)
    rect = page.rect
    doc.close()
    return (rect.width, rect.height)


def compare_pages_ssim(
    img_a_path: str,
    img_b_path: str
) -> float:
    """
    Compare two images using Structural Similarity Index (SSIM).
    Returns a score between 0 (completely different) and 1 (identical).
    """
    try:
        from skimage.metrics import structural_similarity as ssim
        import cv2
        
        img_a = cv2.imread(img_a_path, cv2.IMREAD_GRAYSCALE)
        img_b = cv2.imread(img_b_path, cv2.IMREAD_GRAYSCALE)
        
        if img_a is None or img_b is None:
            return 0.0
        
        # Resize to match dimensions
        if img_a.shape != img_b.shape:
            img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))
        
        score, _ = ssim(img_a, img_b, full=True)
        return float(score)
    except ImportError:
        print("⚠️  scikit-image or opencv not available for SSIM comparison")
        return 0.5  # Neutral score


def detect_text_overflow(
    pdf_path: str,
    page_idx: int = 0
) -> List[dict]:
    """
    Detect potential text overflow or clipping issues.
    Returns list of detected issues.
    """
    if not FITZ_AVAILABLE:
        return []
    
    issues = []
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_idx)
    
    # Get page dimensions
    rect = page.rect
    margin = 28  # ~0.4 inch safety margin (professional docs use tight margins)
    
    # Get all text blocks
    blocks = page.get_text("dict")["blocks"]
    
    for block in blocks:
        if block.get("type") == 0:  # Text block
            bbox = block.get("bbox", [0, 0, 0, 0])
            
            # Check if text is too close to edges
            if bbox[0] < margin:
                issues.append({
                    "type": "left_overflow",
                    "bbox": bbox,
                    "message": f"Text too close to left edge: {bbox[0]:.1f}pt"
                })
            if bbox[2] > rect.width - margin:
                issues.append({
                    "type": "right_overflow",
                    "bbox": bbox,
                    "message": f"Text too close to right edge: {rect.width - bbox[2]:.1f}pt"
                })
            if bbox[1] < margin:
                issues.append({
                    "type": "top_overflow",
                    "bbox": bbox,
                    "message": f"Text too close to top edge: {bbox[1]:.1f}pt"
                })
            if bbox[3] > rect.height - margin:
                issues.append({
                    "type": "bottom_overflow",
                    "bbox": bbox,
                    "message": f"Text too close to bottom edge: {rect.height - bbox[3]:.1f}pt"
                })
    
    doc.close()
    return issues


def analyze_layout_metrics(pdf_path: str, page_idx: int = 0) -> dict:
    """
    Analyze layout metrics of a PDF page.
    Returns metrics that can be used for automated quality checks.
    """
    if not FITZ_AVAILABLE:
        return {"error": "PyMuPDF not available"}
    
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_idx)
    
    rect = page.rect
    page_area = rect.width * rect.height
    
    # Get text blocks
    blocks = page.get_text("dict")["blocks"]
    
    text_blocks = [b for b in blocks if b.get("type") == 0]
    image_blocks = [b for b in blocks if b.get("type") == 1]
    
    # Calculate coverage
    text_area = sum(
        (b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1])
        for b in text_blocks
    )
    
    image_area = sum(
        (b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1])
        for b in image_blocks
    )
    
    # Detect overlaps (with tolerance for minor intersections)
    overlaps = []
    OVERLAP_TOLERANCE = 2.0  # Ignore overlaps smaller than 2pt in any dimension
    
    for i, block_a in enumerate(text_blocks):
        for block_b in text_blocks[i+1:]:
            bbox_a = block_a["bbox"]
            bbox_b = block_b["bbox"]
            
            # Calculate overlap dimensions
            x_overlap = min(bbox_a[2], bbox_b[2]) - max(bbox_a[0], bbox_b[0])
            y_overlap = min(bbox_a[3], bbox_b[3]) - max(bbox_a[1], bbox_b[1])
            
            # Only count as overlap if significant in BOTH dimensions
            if x_overlap > OVERLAP_TOLERANCE and y_overlap > OVERLAP_TOLERANCE:
                overlaps.append({
                    "blocks": [bbox_a, bbox_b],
                    "type": "text_overlap",
                    "overlap_size": (x_overlap, y_overlap)
                })
    
    doc.close()
    
    return {
        "page_width": rect.width,
        "page_height": rect.height,
        "text_block_count": len(text_blocks),
        "image_block_count": len(image_blocks),
        "text_coverage_pct": (text_area / page_area) * 100 if page_area > 0 else 0,
        "image_coverage_pct": (image_area / page_area) * 100 if page_area > 0 else 0,
        "whitespace_pct": max(0, 100 - (text_area + image_area) / page_area * 100),
        "overlap_count": len(overlaps),
        "overlaps": overlaps[:5]  # First 5 overlaps only
    }


if __name__ == "__main__":
    # Test rendering
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "./renders"
        
        print(f"Rendering {pdf_path} to {output_dir}...")
        paths = render_pdf_to_png(pdf_path, output_dir, pages=[0, 1])
        print(f"Generated: {paths}")
        
        print("\nLayout metrics for page 1:")
        metrics = analyze_layout_metrics(pdf_path, 0)
        for k, v in metrics.items():
            if k != "overlaps":
                print(f"  {k}: {v}")
    else:
        print("Usage: python pdf_renderer.py <pdf_path> [output_dir]")

