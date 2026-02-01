"""
Parameterized Layout Configuration for Equity Research Reports

All geometric values are in points (1 inch = 72 points).
This allows the feedback loop to adjust any parameter.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any
import json


@dataclass
class LayoutParams:
    """
    Complete parameterization of the PDF layout.
    Every geometric/typographic value lives here so the LLM can adjust it.
    """
    
    # Page margins (points)
    margin_left: float = 54.0       # 0.75 inch
    margin_right: float = 43.2      # 0.60 inch
    margin_top: float = 63.36       # 0.88 inch
    margin_bottom: float = 54.0     # 0.75 inch
    
    # Summary page (Page 1) layout
    sidebar_width: float = 162.0    # 2.25 inch
    gutter_width: float = 14.4      # 0.20 inch
    bottom_band_height: float = 122.4  # 1.70 inch
    header_clearance: float = 39.6  # 0.55 inch (space below header line)
    
    # Body pages (Page 2+) layout
    body_header_extra: float = 32.4  # 0.45 inch (extra space for analyst info)
    
    # Typography - Headers
    h1_font_size: float = 17.0
    h1_leading: float = 19.0
    h2_font_size: float = 11.0
    h2_leading: float = 13.0
    h2_space_before: float = 10.0
    h2_space_after: float = 4.0
    
    # Typography - Body text
    body_font_size: float = 9.5
    body_leading: float = 12.0
    
    # Typography - Small/Tiny text
    small_font_size: float = 7.6
    small_leading: float = 9.0
    tiny_font_size: float = 6.8
    tiny_leading: float = 8.0
    
    # Table styling
    table_font_size: float = 7.8
    table_header_font_size: float = 7.8
    table_cell_padding_top: float = 3.0
    table_cell_padding_bottom: float = 3.0
    table_cell_padding_left: float = 4.0
    table_cell_padding_right: float = 4.0
    
    # EPS table specific
    eps_table_font_size: float = 7.4
    company_data_table_font_size: float = 7.4
    
    # Sidebar card styling
    sidebar_card_padding: float = 6.0
    sidebar_internal_spacing: float = 4.0
    
    # Bottom band layout (ratio of left vs right tables)
    bottom_band_left_ratio: float = 0.66
    
    # Spacing
    paragraph_spacing: float = 6.0
    section_spacing: float = 10.0
    bullet_indent: float = 12.0
    
    # Header/Footer
    header_line_y_offset: float = 50.4   # 0.70 inch from top
    footer_line_y_offset: float = 44.64  # 0.62 inch from bottom
    footer_text_y_offset: float = 30.24  # 0.42 inch from bottom
    header_font_size: float = 9.2
    footer_font_size: float = 7.5
    
    # Chart/Figure sizing
    figure_width: float = 489.6   # 6.8 inch
    figure_height: float = 216.0  # 3.0 inch
    sidebar_chart_height: float = 90.0  # 1.25 inch
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LayoutParams":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_json(cls, s: str) -> "LayoutParams":
        return cls.from_dict(json.loads(s))
    
    def apply_patch(self, patch: Dict[str, float]) -> "LayoutParams":
        """
        Apply a patch of adjustments (deltas) to create new params.
        Patch values are DELTAS, not absolute values.
        
        Example patch:
        {"sidebar_width": +6.0, "h1_font_size": -1.0}
        """
        current = self.to_dict()
        for key, delta in patch.items():
            if key in current:
                current[key] = current[key] + delta
        return LayoutParams.from_dict(current)
    
    def apply_absolute(self, updates: Dict[str, float]) -> "LayoutParams":
        """Apply absolute value updates (not deltas)."""
        current = self.to_dict()
        current.update(updates)
        return LayoutParams.from_dict(current)


# Presets for different report styles
PRESET_JPMORGAN = LayoutParams(
    margin_left=54.0,
    margin_right=43.2,
    sidebar_width=162.0,
    h1_font_size=17.0,
    body_font_size=9.5,
)

PRESET_GOLDMAN = LayoutParams(
    margin_left=50.4,
    margin_right=50.4,
    sidebar_width=180.0,
    h1_font_size=18.0,
    body_font_size=9.0,
)

PRESET_MORGAN_STANLEY = LayoutParams(
    margin_left=57.6,
    margin_right=43.2,
    sidebar_width=158.4,
    h1_font_size=16.0,
    body_font_size=10.0,
)

