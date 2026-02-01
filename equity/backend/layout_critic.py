"""
Layout Critic - LLM Vision-based Layout Analysis

Uses a vision LLM (xAI Grok, OpenAI GPT-4V, or Anthropic Claude) to analyze
PDF screenshots and suggest layout adjustments.
"""

import json
import base64
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

# Import the layout params
from .layout_params import LayoutParams


@dataclass
class LayoutCritique:
    """Result of a layout critique."""
    score: float  # 0-100 quality score
    issues: List[str]  # List of identified issues
    suggestions: List[str]  # Human-readable suggestions
    patch: Dict[str, float]  # Geometry parameter adjustments (deltas)
    content_patch: Dict[str, Any]  # Content placeholder adjustments
    confidence: float  # 0-1 confidence in the critique
    iteration_complete: bool  # True if no more adjustments needed


def encode_image_base64(image_path: str) -> str:
    """Encode an image file to base64."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def get_image_media_type(image_path: str) -> str:
    """Get the media type for an image."""
    ext = os.path.splitext(image_path)[1].lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp"
    }.get(ext, "image/png")


# System prompt for the layout critic
CRITIC_SYSTEM_PROMPT = """You are an expert layout QA system for institutional equity research PDFs.

Your job is to analyze PDF page screenshots and identify layout issues, then suggest BOTH:
1. Geometry adjustments (margins, spacing, fonts)
2. Content adjustments (add/remove paragraphs, bullets, tables to fill space)

=== GEOMETRY PARAMETERS (values in points, 72pt = 1 inch) ===
- margin_left, margin_right, margin_top, margin_bottom: Page margins
- sidebar_width: Width of summary page sidebar
- gutter_width: Space between main content and sidebar
- bottom_band_height: Height of bottom data tables section
- header_clearance: Space below header line
- h1_font_size, h1_leading: Headline typography
- h2_font_size, h2_leading: Section header typography
- body_font_size, body_leading: Body text typography
- table_font_size: Table typography
- paragraph_spacing, section_spacing: Vertical spacing

=== CONTENT PLACEHOLDERS ===
You can also adjust content density via placeholder tokens:

{{TEXT:id:min:max:current}} - Text paragraphs
{{BULLETS:id:min:max:current}} - Bullet point lists
{{TABLE:id:min:max:w:h}} - Data tables (row count)
{{CHART:id:min:max:w:h}} - Charts/figures
{{SPACER:id:min:max:current}} - Flexible whitespace (pt)
{{ANALYTICS:id:min:max:current}} - Analytics metrics

QUALITY CRITERIA:
1. NO TEXT OVERFLOW - all text within margins
2. NO OVERLAPPING ELEMENTS
3. BALANCED WHITESPACE - 70-85% is ideal for professional documents
4. CONSISTENT ALIGNMENT
5. READABLE TYPOGRAPHY
6. PROPER CONTENT DENSITY - page should feel complete, not empty

OUTPUT FORMAT (JSON only, no prose):
{
  "score": <0-100>,
  "issues": ["issue 1", "issue 2", ...],
  "suggestions": ["suggestion 1", ...],
  "geometry_patch": {
    "<param_name>": <delta_value>,
    ...
  },
  "content_patch": {
    "<placeholder_id>": {"action": "expand|shrink|hide|show", "value": <int>},
    ...
  },
  "confidence": <0.0-1.0>,
  "iteration_complete": <true if acceptable>
}

CONTENT PATCH ACTIONS:
- "expand": Add items ({"action": "expand", "value": 2} = add 2 paragraphs)
- "shrink": Remove items
- "hide": Hide placeholder entirely
- "show": Show hidden placeholder
- "set": Set exact count ({"action": "set", "current_items": 5})

RULES:
- If page is too sparse (>90% white), expand TEXT or BULLETS content
- If page is too dense (<70% white), shrink content or increase margins
- Use geometry for precise fit, content for density balance
- Maximum 5 geometry + 3 content changes per iteration"""


CRITIC_USER_PROMPT_TEMPLATE = """Analyze this equity research PDF page screenshot.

Current layout parameters:
{params_json}

Current content placeholders:
{placeholders_summary}

Previous iteration feedback (if any):
{previous_feedback}

Page being analyzed: {page_description}

Provide your analysis as JSON only. Identify layout issues and suggest BOTH geometry and content adjustments."""


class LayoutCritic:
    """
    Vision LLM-based layout critic.
    
    Supports multiple backends:
    - xAI Grok (grok-3)
    - OpenAI GPT-4V (gpt-4o)
    - Anthropic Claude (claude-3-5-sonnet)
    """
    
    def __init__(
        self,
        provider: str = "xai",
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.provider = provider.lower()
        self.api_key = api_key or self._get_api_key()
        self.model = model or self._get_default_model()
        self.client = self._create_client()
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment."""
        if self.provider == "xai":
            return os.getenv("XAI_API_KEY")
        elif self.provider == "openai":
            return os.getenv("OPENAI_API_KEY")
        elif self.provider == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY")
        return None
    
    def _get_default_model(self) -> str:
        """Get default model for provider."""
        return {
            "xai": "grok-3",
            "openai": "gpt-4o",
            "anthropic": "claude-3-5-sonnet-20241022"
        }.get(self.provider, "gpt-4o")
    
    def _create_client(self):
        """Create the appropriate API client."""
        if not self.api_key:
            return None
        
        if self.provider == "xai":
            from openai import OpenAI
            return OpenAI(
                api_key=self.api_key,
                base_url="https://api.x.ai/v1"
            )
        elif self.provider == "openai":
            from openai import OpenAI
            return OpenAI(api_key=self.api_key)
        elif self.provider == "anthropic":
            try:
                import anthropic
                return anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                print("⚠️  anthropic package not installed")
                return None
        
        return None
    
    def critique(
        self,
        image_paths: List[str],
        current_params: LayoutParams,
        previous_feedback: str = "",
        page_descriptions: Optional[List[str]] = None,
        placeholders_summary: str = ""
    ) -> LayoutCritique:
        """
        Critique the PDF layout based on page screenshots.
        
        Args:
            image_paths: List of PNG screenshot paths
            current_params: Current layout parameters
            previous_feedback: Feedback from previous iteration (if any)
            page_descriptions: Descriptions for each page (e.g., "Summary Page 1")
            placeholders_summary: Summary of content placeholder states
        
        Returns:
            LayoutCritique with score, issues, and suggested patches
        """
        if not self.client:
            return self._fallback_critique(current_params)
        
        if page_descriptions is None:
            page_descriptions = [f"Page {i+1}" for i in range(len(image_paths))]
        
        # Build the critique request
        params_json = current_params.to_json()
        
        try:
            if self.provider in ["xai", "openai"]:
                return self._critique_openai_style(
                    image_paths, params_json, previous_feedback, page_descriptions, placeholders_summary
                )
            elif self.provider == "anthropic":
                return self._critique_anthropic(
                    image_paths, params_json, previous_feedback, page_descriptions, placeholders_summary
                )
        except Exception as e:
            print(f"⚠️  LLM critique failed: {e}")
            return self._fallback_critique(current_params)
        
        return self._fallback_critique(current_params)
    
    def _critique_openai_style(
        self,
        image_paths: List[str],
        params_json: str,
        previous_feedback: str,
        page_descriptions: List[str],
        placeholders_summary: str = ""
    ) -> LayoutCritique:
        """Critique using OpenAI-style API (works for xAI and OpenAI)."""
        
        # Build content with images
        content = []
        
        for i, (img_path, desc) in enumerate(zip(image_paths, page_descriptions)):
            # Add image
            base64_image = encode_image_base64(img_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{get_image_media_type(img_path)};base64,{base64_image}"
                }
            })
            content.append({
                "type": "text",
                "text": f"[{desc}]"
            })
        
        # Add the analysis request
        user_prompt = CRITIC_USER_PROMPT_TEMPLATE.format(
            params_json=params_json,
            placeholders_summary=placeholders_summary or "No placeholders configured",
            previous_feedback=previous_feedback or "None (first iteration)",
            page_description=", ".join(page_descriptions)
        )
        content.append({"type": "text", "text": user_prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                {"role": "user", "content": content}
            ],
            max_tokens=1500,
            temperature=0.3
        )
        
        return self._parse_response(response.choices[0].message.content)
    
    def _critique_anthropic(
        self,
        image_paths: List[str],
        params_json: str,
        previous_feedback: str,
        page_descriptions: List[str],
        placeholders_summary: str = ""
    ) -> LayoutCritique:
        """Critique using Anthropic Claude API."""
        
        content = []
        
        for i, (img_path, desc) in enumerate(zip(image_paths, page_descriptions)):
            base64_image = encode_image_base64(img_path)
            media_type = get_image_media_type(img_path)
            
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_image
                }
            })
            content.append({
                "type": "text",
                "text": f"[{desc}]"
            })
        
        user_prompt = CRITIC_USER_PROMPT_TEMPLATE.format(
            params_json=params_json,
            placeholders_summary=placeholders_summary or "No placeholders configured",
            previous_feedback=previous_feedback or "None (first iteration)",
            page_description=", ".join(page_descriptions)
        )
        content.append({"type": "text", "text": user_prompt})
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            system=CRITIC_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}]
        )
        
        return self._parse_response(response.content[0].text)
    
    def _parse_response(self, response_text: str) -> LayoutCritique:
        """Parse LLM response into LayoutCritique."""
        try:
            # Extract JSON from response (handle markdown code blocks)
            text = response_text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            data = json.loads(text)
            
            return LayoutCritique(
                score=float(data.get("score", 50)),
                issues=data.get("issues", []),
                suggestions=data.get("suggestions", []),
                patch=data.get("geometry_patch", data.get("patch", {})),  # Support both keys
                content_patch=data.get("content_patch", {}),
                confidence=float(data.get("confidence", 0.5)),
                iteration_complete=data.get("iteration_complete", False)
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"⚠️  Failed to parse LLM response: {e}")
            print(f"Response was: {response_text[:500]}...")
            return self._fallback_critique(LayoutParams())
    
    def _fallback_critique(self, params: LayoutParams) -> LayoutCritique:
        """Fallback critique when LLM is not available."""
        return LayoutCritique(
            score=70.0,
            issues=["Unable to perform visual analysis (LLM not configured)"],
            suggestions=["Configure XAI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY for visual feedback"],
            patch={},
            content_patch={},
            confidence=0.0,
            iteration_complete=True
        )


class RuleBasedCritic:
    """
    Rule-based layout critic using PDF analysis metrics.
    Faster and more deterministic than LLM-based critique.
    """
    
    # Parameter bounds to prevent runaway values
    PARAM_BOUNDS = {
        "paragraph_spacing": (4.0, 16.0),
        "body_leading": (10.0, 18.0),
        "margin_left": (36.0, 72.0),
        "margin_right": (36.0, 72.0),
        "margin_top": (36.0, 90.0),
        "margin_bottom": (36.0, 72.0),
        "h1_font_size": (14.0, 24.0),
        "h2_font_size": (10.0, 14.0),
        "body_font_size": (8.0, 11.0),
    }
    
    def __init__(self):
        from .pdf_renderer import analyze_layout_metrics, detect_text_overflow
        self.analyze_metrics = analyze_layout_metrics
        self.detect_overflow = detect_text_overflow
        self._previous_issues: Dict[str, int] = {}  # Track persistent issues
    
    def _would_exceed_bounds(self, param: str, current_params: LayoutParams, delta: float) -> bool:
        """Check if applying delta would exceed parameter bounds."""
        if param not in self.PARAM_BOUNDS:
            return False
        
        current_val = getattr(current_params, param, 0)
        new_val = current_val + delta
        min_val, max_val = self.PARAM_BOUNDS[param]
        return new_val < min_val or new_val > max_val
    
    def _safe_adjust(
        self, 
        patch: Dict[str, float], 
        param: str, 
        delta: float, 
        current_params: LayoutParams
    ) -> bool:
        """Apply adjustment only if it stays within bounds. Returns True if applied."""
        if self._would_exceed_bounds(param, current_params, delta):
            return False
        patch[param] = patch.get(param, 0) + delta
        return True
    
    def critique(
        self,
        pdf_path: str,
        current_params: LayoutParams,
        pages: List[int] = [0, 1]
    ) -> LayoutCritique:
        """
        Rule-based critique using PDF metrics.
        Includes bounds checking and convergence logic.
        """
        issues = []
        suggestions = []
        patch = {}
        score = 100.0
        
        # Track issue fingerprints for convergence detection
        current_issue_keys = set()
        
        for page_idx in pages:
            # Analyze metrics
            metrics = self.analyze_metrics(pdf_path, page_idx)
            overflow_issues = self.detect_overflow(pdf_path, page_idx)
            
            overlap_count = metrics.get("overlap_count", 0)
            
            # Check for overlaps
            if overlap_count > 0:
                issue_key = f"overlap_p{page_idx}_{overlap_count}"
                current_issue_keys.add(issue_key)
                
                issues.append(f"Page {page_idx + 1}: {overlap_count} text overlaps detected")
                suggestions.append("Increase spacing or reduce font sizes")
                score -= 10 * overlap_count
                
                # Only apply fix if we haven't already maxed out these params
                # Try different fixes based on how many times we've seen this issue
                repeat_count = self._previous_issues.get(issue_key, 0)
                
                if repeat_count == 0:
                    # First attempt: increase paragraph spacing
                    self._safe_adjust(patch, "paragraph_spacing", 2.0, current_params)
                elif repeat_count == 1:
                    # Second attempt: increase body leading
                    self._safe_adjust(patch, "body_leading", 1.5, current_params)
                elif repeat_count == 2:
                    # Third attempt: reduce font size
                    self._safe_adjust(patch, "body_font_size", -0.5, current_params)
                else:
                    # Give up on this fix path, mark as unresolvable
                    suggestions.append(f"Overlap on page {page_idx + 1} may require content reduction")
            
            # Check whitespace (professional docs typically have 70-90% whitespace)
            whitespace = metrics.get("whitespace_pct", 75)
            if whitespace < 50:
                issue_key = f"dense_p{page_idx}"
                current_issue_keys.add(issue_key)
                
                issues.append(f"Page {page_idx + 1}: Too dense ({whitespace:.1f}% whitespace)")
                suggestions.append("Increase margins or reduce content")
                score -= 10
                
                # Only adjust if within bounds
                self._safe_adjust(patch, "margin_left", 3.0, current_params)
                self._safe_adjust(patch, "margin_right", 3.0, current_params)
                
            elif whitespace > 95:
                issue_key = f"sparse_p{page_idx}"
                current_issue_keys.add(issue_key)
                
                issues.append(f"Page {page_idx + 1}: Too sparse ({whitespace:.1f}% whitespace)")
                suggestions.append("Reduce margins or increase content")
                score -= 5
                
                self._safe_adjust(patch, "margin_left", -2.0, current_params)
                self._safe_adjust(patch, "margin_right", -2.0, current_params)
            
            # Check for overflow
            if overflow_issues:
                for issue in overflow_issues[:3]:  # First 3 issues
                    issues.append(f"Page {page_idx + 1}: {issue['message']}")
                score -= 5 * len(overflow_issues)
                
                # Only fix if we haven't already pushed margins too far
                if any(i["type"] == "right_overflow" for i in overflow_issues):
                    if not self._safe_adjust(patch, "margin_right", 4.0, current_params):
                        # Margins maxed, try reducing font
                        self._safe_adjust(patch, "body_font_size", -0.3, current_params)
                        
                if any(i["type"] == "bottom_overflow" for i in overflow_issues):
                    if not self._safe_adjust(patch, "margin_bottom", 4.0, current_params):
                        self._safe_adjust(patch, "paragraph_spacing", -1.0, current_params)
        
        score = max(0, min(100, score))
        
        # Update issue tracking for convergence detection
        for key in current_issue_keys:
            self._previous_issues[key] = self._previous_issues.get(key, 0) + 1
        
        # Clear issues that are no longer present
        resolved = [k for k in self._previous_issues if k not in current_issue_keys]
        for k in resolved:
            del self._previous_issues[k]
        
        # Check for convergence: if all remaining issues have been seen 3+ times, stop
        stuck_issues = sum(1 for v in self._previous_issues.values() if v >= 3)
        all_stuck = stuck_issues == len(self._previous_issues) and len(self._previous_issues) > 0
        
        # Also stop if we have no useful adjustments to make
        no_adjustments = len(patch) == 0
        
        iteration_complete = len(issues) == 0 or score >= 85 or all_stuck or no_adjustments
        
        if all_stuck and issues:
            suggestions.append("Layout issues persist despite adjustments - may need content changes")
        
        # Content adjustments based on whitespace analysis
        content_patch = {}
        for page_idx in pages:
            metrics = self.analyze_metrics(pdf_path, page_idx)
            whitespace = metrics.get("whitespace_pct", 75)
            
            if whitespace > 90:
                # Too sparse - expand content
                content_patch["summary_intro"] = {"action": "expand", "value": 1}
                content_patch["key_points"] = {"action": "expand", "value": 1}
            elif whitespace < 65:
                # Too dense - shrink content
                content_patch["summary_intro"] = {"action": "shrink", "value": 1}
                content_patch["sidebar_metrics"] = {"action": "shrink", "value": 2}
        
        return LayoutCritique(
            score=score,
            issues=issues,
            suggestions=suggestions,
            patch=patch,
            content_patch=content_patch,
            confidence=0.9 if issues else 1.0,
            iteration_complete=iteration_complete
        )


if __name__ == "__main__":
    # Test the critic
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        params = LayoutParams()
        
        # Try LLM critic
        critic = LayoutCritic(provider="xai")
        if critic.client:
            print("Testing LLM critic...")
            result = critic.critique(
                [image_path],
                params,
                page_descriptions=["Summary Page 1"]
            )
            print(f"Score: {result.score}")
            print(f"Issues: {result.issues}")
            print(f"Patch: {result.patch}")
        else:
            print("No API key configured, testing rule-based critic...")
    else:
        print("Usage: python layout_critic.py <image_path>")
        print("\nSet XAI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY for LLM critique")

