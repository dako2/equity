"""
Format Agent - AI-powered PDF Layout Designer

An intelligent agent that monitors, tunes, and designs PDF layouts by:
1. Rendering PDFs to images
2. Analyzing layouts using LLM vision
3. Providing structured feedback
4. Applying iterative adjustments
5. Tracking design decisions and history
"""

import os
import json
import base64
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

from .layout_params import LayoutParams
from .pdf_renderer import render_pdf_to_png, analyze_layout_metrics, FITZ_AVAILABLE


class DesignPriority(Enum):
    """Design priorities the agent can focus on."""
    READABILITY = "readability"
    DENSITY = "density"
    WHITESPACE = "whitespace"
    ALIGNMENT = "alignment"
    TYPOGRAPHY = "typography"
    PROFESSIONAL = "professional"
    BALANCED = "balanced"


@dataclass
class DesignFeedback:
    """Structured feedback from the format agent."""
    overall_score: float  # 0-100
    category_scores: Dict[str, float]  # Category -> score
    issues: List[str]
    suggestions: List[str]
    param_adjustments: Dict[str, float]  # Parameter deltas
    content_adjustments: Dict[str, Any]
    reasoning: str
    confidence: float  # 0-1
    iteration_complete: bool


@dataclass
class DesignIteration:
    """Record of a single design iteration."""
    iteration: int
    timestamp: str
    params: Dict[str, float]
    feedback: DesignFeedback
    image_paths: List[str]
    metrics: Dict[str, Any]
    duration_seconds: float


@dataclass 
class DesignSession:
    """A complete design session with history."""
    session_id: str
    started_at: str
    target_style: str
    priority: DesignPriority
    iterations: List[DesignIteration] = field(default_factory=list)
    final_params: Optional[Dict[str, float]] = None
    final_score: float = 0.0
    status: str = "in_progress"


class FormatAgent:
    """
    AI-powered format agent for PDF layout design.
    
    The agent iteratively improves PDF layouts by:
    1. Rendering the current PDF to images
    2. Analyzing the layout visually using LLM
    3. Generating structured feedback
    4. Applying parameter adjustments
    5. Repeating until satisfied or max iterations reached
    """
    
    # System prompt for the format agent
    AGENT_SYSTEM_PROMPT = """You are an expert PDF layout designer specializing in institutional equity research reports.

Your job is to analyze PDF page screenshots and provide detailed feedback on the layout design, then suggest specific parameter adjustments.

## DESIGN PRINCIPLES FOR EQUITY RESEARCH REPORTS:

1. **Professional Appearance**
   - Clean, organized layout with clear visual hierarchy
   - Consistent spacing and alignment
   - Proper margins (not too tight, not too loose)
   - JPM/Goldman/Morgan Stanley style formatting

2. **Readability**
   - Body text 9-10pt for optimal reading
   - Adequate line spacing (leading 1.2-1.4x font size)
   - Clear section headers with proper spacing
   - Tables with readable cell padding

3. **Content Density**
   - 65-80% whitespace is ideal for professional docs
   - Avoid overcrowding or sparse layouts
   - Balance between text, tables, and charts

4. **Two-Column Layout (Summary Pages)**
   - Main column ~65% width for key points
   - Sidebar ~30% width for data/metrics
   - Clear gutter separation

## ADJUSTABLE PARAMETERS (values in points, 72pt = 1 inch):

### Margins
- margin_left, margin_right, margin_top, margin_bottom (typical: 36-72pt)

### Typography
- h1_font_size (typical: 14-20pt)
- h2_font_size (typical: 10-14pt)
- body_font_size (typical: 8-11pt)
- body_leading (line height, typical: 10-14pt)

### Spacing
- paragraph_spacing (typical: 4-12pt)
- section_spacing (typical: 8-16pt)
- gutter_width (typical: 10-20pt)

### Layout
- sidebar_width (typical: 140-180pt)
- bottom_band_height (typical: 100-150pt)

## YOUR RESPONSE FORMAT (JSON only):

```json
{
  "overall_score": <0-100>,
  "category_scores": {
    "margins": <0-100>,
    "typography": <0-100>,
    "spacing": <0-100>,
    "alignment": <0-100>,
    "density": <0-100>,
    "professionalism": <0-100>
  },
  "issues": [
    "Specific issue 1",
    "Specific issue 2"
  ],
  "suggestions": [
    "Actionable suggestion 1",
    "Actionable suggestion 2"
  ],
  "param_adjustments": {
    "parameter_name": <delta_value>,
    ...
  },
  "reasoning": "Brief explanation of your analysis and recommendations",
  "confidence": <0.0-1.0>,
  "iteration_complete": <true if layout is acceptable>
}
```

## RULES:
- Provide SPECIFIC, ACTIONABLE feedback
- Parameter adjustments are DELTAS (positive = increase, negative = decrease)
- Maximum 5 parameter adjustments per iteration
- Focus on the MOST IMPACTFUL issues first
- Be conservative with adjustments (small incremental changes)
- Mark iteration_complete=true only when layout is professional quality (score >= 85)
"""

    AGENT_USER_PROMPT = """Analyze this equity research PDF layout.

Current Parameters:
{params_json}

Previous Iteration Feedback (if any):
{previous_feedback}

Design Priority: {priority}
Target Style: {target_style}

Pages Being Analyzed:
{page_descriptions}

Provide your analysis as JSON. Focus on {priority} while maintaining professional appearance."""

    def __init__(
        self,
        provider: str = "anthropic",
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize the format agent.
        
        Args:
            provider: LLM provider ("anthropic", "openai", "xai")
            api_key: API key (or from environment)
            model: Model name (or use default)
        """
        self.provider = provider.lower()
        self.api_key = api_key or self._get_api_key()
        self.model = model or self._get_default_model()
        self.client = self._create_client()
        self.sessions: Dict[str, DesignSession] = {}
        
    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment."""
        env_vars = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "xai": "XAI_API_KEY"
        }
        return os.getenv(env_vars.get(self.provider, ""))
    
    def _get_default_model(self) -> str:
        """Get default model for provider."""
        return {
            "anthropic": "claude-sonnet-4-20250514",
            "openai": "gpt-4o",
            "xai": "grok-3"
        }.get(self.provider, "gpt-4o")
    
    def _create_client(self):
        """Create the API client."""
        if not self.api_key:
            print(f"⚠️  No API key for {self.provider}")
            return None
            
        if self.provider == "anthropic":
            try:
                import anthropic
                return anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                print("⚠️  anthropic package not installed")
                return None
        elif self.provider in ["openai", "xai"]:
            from openai import OpenAI
            base_url = "https://api.x.ai/v1" if self.provider == "xai" else None
            return OpenAI(api_key=self.api_key, base_url=base_url)
        
        return None
    
    def _encode_image(self, image_path: str) -> Tuple[str, str]:
        """Encode image to base64 and get media type."""
        with open(image_path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode("utf-8")
        
        ext = os.path.splitext(image_path)[1].lower()
        media_type = {"png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(ext, "image/png")
        
        return data, media_type
    
    def analyze_layout(
        self,
        image_paths: List[str],
        current_params: LayoutParams,
        priority: DesignPriority = DesignPriority.BALANCED,
        target_style: str = "JPM Equity Research",
        previous_feedback: Optional[DesignFeedback] = None,
        page_descriptions: Optional[List[str]] = None
    ) -> DesignFeedback:
        """
        Analyze PDF layout images and provide feedback.
        
        Args:
            image_paths: Paths to rendered PDF page images
            current_params: Current layout parameters
            priority: Design priority to focus on
            target_style: Target style description
            previous_feedback: Feedback from previous iteration
            page_descriptions: Descriptions for each page
        
        Returns:
            DesignFeedback with scores, issues, and adjustments
        """
        if not self.client:
            return self._fallback_feedback()
        
        if page_descriptions is None:
            page_descriptions = [f"Page {i+1}" for i in range(len(image_paths))]
        
        # Build previous feedback string
        prev_feedback_str = "None (first iteration)"
        if previous_feedback:
            prev_feedback_str = json.dumps({
                "score": previous_feedback.overall_score,
                "issues": previous_feedback.issues[:3],
                "adjustments_applied": previous_feedback.param_adjustments
            }, indent=2)
        
        # Build user prompt
        user_prompt = self.AGENT_USER_PROMPT.format(
            params_json=current_params.to_json(),
            previous_feedback=prev_feedback_str,
            priority=priority.value,
            target_style=target_style,
            page_descriptions=", ".join(page_descriptions)
        )
        
        try:
            if self.provider == "anthropic":
                return self._analyze_anthropic(image_paths, user_prompt, page_descriptions)
            else:
                return self._analyze_openai(image_paths, user_prompt, page_descriptions)
        except Exception as e:
            print(f"⚠️  Analysis failed: {e}")
            return self._fallback_feedback()
    
    def _analyze_anthropic(
        self, 
        image_paths: List[str], 
        user_prompt: str,
        page_descriptions: List[str]
    ) -> DesignFeedback:
        """Analyze using Anthropic Claude."""
        content = []
        
        for img_path, desc in zip(image_paths, page_descriptions):
            data, media_type = self._encode_image(img_path)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": data
                }
            })
            content.append({"type": "text", "text": f"[{desc}]"})
        
        content.append({"type": "text", "text": user_prompt})
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=self.AGENT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}]
        )
        
        return self._parse_response(response.content[0].text)
    
    def _analyze_openai(
        self,
        image_paths: List[str],
        user_prompt: str,
        page_descriptions: List[str]
    ) -> DesignFeedback:
        """Analyze using OpenAI/xAI."""
        content = []
        
        for img_path, desc in zip(image_paths, page_descriptions):
            data, media_type = self._encode_image(img_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{data}"}
            })
            content.append({"type": "text", "text": f"[{desc}]"})
        
        content.append({"type": "text", "text": user_prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=2000,
            messages=[
                {"role": "system", "content": self.AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": content}
            ]
        )
        
        return self._parse_response(response.choices[0].message.content)
    
    def _parse_response(self, response_text: str) -> DesignFeedback:
        """Parse LLM response into DesignFeedback."""
        try:
            # Extract JSON from response
            text = response_text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            data = json.loads(text)
            
            return DesignFeedback(
                overall_score=float(data.get("overall_score", 50)),
                category_scores=data.get("category_scores", {}),
                issues=data.get("issues", []),
                suggestions=data.get("suggestions", []),
                param_adjustments=data.get("param_adjustments", {}),
                content_adjustments=data.get("content_adjustments", {}),
                reasoning=data.get("reasoning", ""),
                confidence=float(data.get("confidence", 0.5)),
                iteration_complete=data.get("iteration_complete", False)
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"⚠️  Failed to parse response: {e}")
            return self._fallback_feedback()
    
    def _fallback_feedback(self) -> DesignFeedback:
        """Fallback when LLM is not available."""
        return DesignFeedback(
            overall_score=70.0,
            category_scores={},
            issues=["LLM analysis not available"],
            suggestions=["Configure API key for visual feedback"],
            param_adjustments={},
            content_adjustments={},
            reasoning="Fallback - no LLM configured",
            confidence=0.0,
            iteration_complete=True
        )
    
    def run_design_session(
        self,
        pdf_builder: Callable[[LayoutParams, str], None],
        initial_params: Optional[LayoutParams] = None,
        priority: DesignPriority = DesignPriority.BALANCED,
        target_style: str = "JPM Equity Research",
        max_iterations: int = 5,
        target_score: float = 85.0,
        output_dir: str = "./design_sessions",
        pages_to_analyze: List[int] = [0, 1]
    ) -> DesignSession:
        """
        Run a complete design session with iterative improvements.
        
        Args:
            pdf_builder: Function that generates PDF from (params, output_path)
            initial_params: Starting layout parameters
            priority: Design priority focus
            target_style: Target visual style
            max_iterations: Maximum iterations
            target_score: Score threshold to stop
            output_dir: Directory for session outputs
            pages_to_analyze: Page indices to analyze
        
        Returns:
            DesignSession with complete history
        """
        import time
        
        # Initialize session
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session = DesignSession(
            session_id=session_id,
            started_at=datetime.now().isoformat(),
            target_style=target_style,
            priority=priority
        )
        
        os.makedirs(output_dir, exist_ok=True)
        session_dir = os.path.join(output_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        params = initial_params or LayoutParams()
        previous_feedback = None
        
        print("\n" + "="*70)
        print("🎨 FORMAT AGENT - Design Session Started")
        print("="*70)
        print(f"  Session ID: {session_id}")
        print(f"  Priority: {priority.value}")
        print(f"  Target Style: {target_style}")
        print(f"  Max Iterations: {max_iterations}")
        print(f"  Target Score: {target_score}")
        print("="*70 + "\n")
        
        for i in range(max_iterations):
            iter_start = time.time()
            print(f"\n📐 Iteration {i + 1}/{max_iterations}")
            print("-" * 50)
            
            # 1. Generate PDF
            pdf_path = os.path.join(session_dir, f"iteration_{i+1}.pdf")
            print("  → Generating PDF...")
            try:
                pdf_builder(params, pdf_path)
            except Exception as e:
                print(f"  ❌ PDF generation failed: {e}")
                break
            
            # 2. Render to images
            print("  → Rendering to images...")
            png_dir = os.path.join(session_dir, f"iteration_{i+1}_images")
            
            if FITZ_AVAILABLE:
                image_paths = render_pdf_to_png(pdf_path, png_dir, pages=pages_to_analyze, zoom=2.0)
            else:
                print("  ⚠️  PyMuPDF not available")
                image_paths = []
            
            # 3. Analyze metrics
            metrics = {}
            if FITZ_AVAILABLE:
                for page_idx in pages_to_analyze:
                    metrics[f"page_{page_idx+1}"] = analyze_layout_metrics(pdf_path, page_idx)
            
            # 4. Get LLM feedback
            print("  → Analyzing layout with AI...")
            page_descs = [
                "Summary Page" if p == 0 else f"Body Page {p}"
                for p in pages_to_analyze
            ]
            
            feedback = self.analyze_layout(
                image_paths=image_paths,
                current_params=params,
                priority=priority,
                target_style=target_style,
                previous_feedback=previous_feedback,
                page_descriptions=page_descs
            )
            
            iter_duration = time.time() - iter_start
            
            # Record iteration
            iteration = DesignIteration(
                iteration=i + 1,
                timestamp=datetime.now().isoformat(),
                params=params.to_dict(),
                feedback=feedback,
                image_paths=image_paths,
                metrics=metrics,
                duration_seconds=iter_duration
            )
            session.iterations.append(iteration)
            
            # Print feedback
            print(f"\n  📊 Score: {feedback.overall_score:.1f}/100")
            
            if feedback.category_scores:
                print("  📈 Category Scores:")
                for cat, score in feedback.category_scores.items():
                    bar = "█" * int(score / 10) + "░" * (10 - int(score / 10))
                    print(f"     {cat:15s} [{bar}] {score:.0f}")
            
            if feedback.issues:
                print(f"  ⚠️  Issues ({len(feedback.issues)}):")
                for issue in feedback.issues[:5]:
                    print(f"     • {issue}")
            
            if feedback.suggestions:
                print(f"  💡 Suggestions:")
                for sug in feedback.suggestions[:3]:
                    print(f"     • {sug}")
            
            if feedback.param_adjustments:
                print(f"  🔧 Adjustments ({len(feedback.param_adjustments)}):")
                for param, delta in list(feedback.param_adjustments.items())[:5]:
                    print(f"     • {param}: {delta:+.1f}pt")
            
            if feedback.reasoning:
                print(f"  📝 Reasoning: {feedback.reasoning[:200]}...")
            
            previous_feedback = feedback
            
            # 5. Check termination
            if feedback.overall_score >= target_score:
                print(f"\n  ✅ Target score reached! ({feedback.overall_score:.1f} >= {target_score})")
                session.status = "completed"
                break
            
            if feedback.iteration_complete:
                print("\n  ✅ Agent marked design as complete")
                session.status = "completed"
                break
            
            if not feedback.param_adjustments:
                print("\n  ⚠️  No adjustments suggested, stopping")
                session.status = "no_adjustments"
                break
            
            # 6. Apply adjustments
            print("\n  → Applying adjustments...")
            params = params.apply_patch(feedback.param_adjustments)
        
        # Finalize session
        session.final_params = params.to_dict()
        session.final_score = session.iterations[-1].feedback.overall_score if session.iterations else 0
        
        if session.status == "in_progress":
            session.status = "max_iterations"
        
        # Save final PDF
        final_pdf = os.path.join(session_dir, "final_design.pdf")
        pdf_builder(params, final_pdf)
        
        # Save session data
        self._save_session(session, session_dir)
        
        print("\n" + "="*70)
        print("🎨 DESIGN SESSION COMPLETE")
        print("="*70)
        print(f"  Status: {session.status}")
        print(f"  Final Score: {session.final_score:.1f}/100")
        print(f"  Iterations: {len(session.iterations)}")
        print(f"  Output: {session_dir}")
        print("="*70 + "\n")
        
        self.sessions[session_id] = session
        return session
    
    def _save_session(self, session: DesignSession, output_dir: str):
        """Save session data to JSON."""
        session_data = {
            "session_id": session.session_id,
            "started_at": session.started_at,
            "target_style": session.target_style,
            "priority": session.priority.value,
            "status": session.status,
            "final_score": session.final_score,
            "final_params": session.final_params,
            "iterations": [
                {
                    "iteration": it.iteration,
                    "timestamp": it.timestamp,
                    "params": it.params,
                    "feedback": {
                        "overall_score": it.feedback.overall_score,
                        "category_scores": it.feedback.category_scores,
                        "issues": it.feedback.issues,
                        "suggestions": it.feedback.suggestions,
                        "param_adjustments": it.feedback.param_adjustments,
                        "reasoning": it.feedback.reasoning,
                        "iteration_complete": it.feedback.iteration_complete
                    },
                    "metrics": it.metrics,
                    "duration_seconds": it.duration_seconds
                }
                for it in session.iterations
            ]
        }
        
        with open(os.path.join(output_dir, "session.json"), "w") as f:
            json.dump(session_data, f, indent=2)
    
    def get_design_report(self, session: DesignSession) -> str:
        """Generate a human-readable design report."""
        lines = [
            "=" * 70,
            "📊 DESIGN SESSION REPORT",
            "=" * 70,
            f"Session ID: {session.session_id}",
            f"Target Style: {session.target_style}",
            f"Priority: {session.priority.value}",
            f"Status: {session.status}",
            f"Final Score: {session.final_score:.1f}/100",
            "",
            "ITERATION HISTORY:",
            "-" * 50,
        ]
        
        for it in session.iterations:
            lines.append(f"\nIteration {it.iteration}:")
            lines.append(f"  Score: {it.feedback.overall_score:.1f}")
            lines.append(f"  Issues: {len(it.feedback.issues)}")
            lines.append(f"  Adjustments: {len(it.feedback.param_adjustments)}")
            if it.feedback.param_adjustments:
                for param, delta in it.feedback.param_adjustments.items():
                    lines.append(f"    • {param}: {delta:+.1f}pt")
        
        if session.final_params:
            lines.append("\nFINAL PARAMETERS:")
            lines.append("-" * 50)
            
            initial = LayoutParams()
            for param, value in session.final_params.items():
                initial_val = getattr(initial, param, value)
                if value != initial_val:
                    lines.append(f"  {param}: {initial_val:.1f} → {value:.1f}")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)


# Convenience function
def run_format_agent(
    pdf_builder: Callable[[LayoutParams, str], None],
    initial_params: Optional[LayoutParams] = None,
    provider: str = "anthropic",
    priority: str = "balanced",
    max_iterations: int = 5,
    target_score: float = 85.0,
    output_dir: str = "./design_sessions"
) -> DesignSession:
    """
    Convenience function to run the format agent.
    
    Args:
        pdf_builder: Function that generates PDF from (params, output_path)
        initial_params: Starting layout parameters
        provider: LLM provider ("anthropic", "openai", "xai")
        priority: Design priority ("balanced", "readability", "density", etc.)
        max_iterations: Maximum iterations
        target_score: Target score to stop at
        output_dir: Output directory
    
    Returns:
        DesignSession with results
    """
    agent = FormatAgent(provider=provider)
    
    priority_enum = DesignPriority(priority) if priority in [p.value for p in DesignPriority] else DesignPriority.BALANCED
    
    return agent.run_design_session(
        pdf_builder=pdf_builder,
        initial_params=initial_params,
        priority=priority_enum,
        max_iterations=max_iterations,
        target_score=target_score,
        output_dir=output_dir
    )


if __name__ == "__main__":
    print("Format Agent - AI-powered PDF Layout Designer")
    print("-" * 50)
    print("\nUsage:")
    print("  from backend.format_agent import FormatAgent, run_format_agent")
    print("  ")
    print("  def my_pdf_builder(params, output_path):")
    print("      # Your PDF generation code")
    print("      pass")
    print("  ")
    print("  session = run_format_agent(my_pdf_builder, provider='anthropic')")
    print("  print(f'Final score: {session.final_score}')")
