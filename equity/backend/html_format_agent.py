"""
HTML Format Agent - AI-powered Markdown/HTML Layout Designer

Analyzes the rendered HTML preview using screenshots and provides
feedback similar to the PDF Format Agent.

Requires: playwright (pip install playwright && playwright install)
"""

import os
import json
import asyncio
import tempfile
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from .format_agent import FormatAgent, DesignFeedback, DesignPriority


class HTMLFormatAgent:
    """
    Format agent for HTML/Markdown rendered previews.
    
    Takes screenshots of rendered HTML pages and uses LLM vision
    to analyze and provide layout feedback.
    """
    
    # Default CSS variables that can be tuned
    CSS_VARIABLES = {
        "page-width": 612,
        "page-height": 792,
        "margin-left": 54,
        "margin-right": 43,
        "margin-top": 54,
        "margin-bottom": 54,
        "sidebar-width": 162,
        "gutter-width": 14,
        "h1-size": 17,
        "h2-size": 11,
        "body-size": 9.5,
        "small-size": 7.6,
        "tiny-size": 6.8,
    }
    
    def __init__(
        self,
        provider: str = "anthropic",
        api_key: Optional[str] = None
    ):
        """Initialize the HTML format agent."""
        self.agent = FormatAgent(provider=provider, api_key=api_key)
        self.playwright = None
        self.browser = None
        
    async def _init_browser(self):
        """Initialize Playwright browser."""
        if self.browser:
            return
            
        try:
            from playwright.async_api import async_playwright
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(headless=True)
        except ImportError:
            print("⚠️  Playwright not installed. Run: pip install playwright && playwright install")
            raise
    
    async def _close_browser(self):
        """Close the browser."""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        self.browser = None
        self.playwright = None
    
    async def capture_screenshot(
        self,
        html_path: str,
        output_path: str,
        css_overrides: Optional[Dict[str, float]] = None,
        viewport_width: int = 1200,
        viewport_height: int = 900
    ) -> str:
        """
        Capture screenshot of an HTML file.
        
        Args:
            html_path: Path to HTML file
            output_path: Path to save screenshot
            css_overrides: CSS variable overrides to apply
            viewport_width: Browser viewport width
            viewport_height: Browser viewport height
        
        Returns:
            Path to saved screenshot
        """
        await self._init_browser()
        
        page = await self.browser.new_page()
        await page.set_viewport_size({"width": viewport_width, "height": viewport_height})
        
        # Load the HTML file
        file_url = f"file://{os.path.abspath(html_path)}"
        await page.goto(file_url)
        
        # Apply CSS overrides if provided
        if css_overrides:
            css_script = "document.documentElement.style.cssText = `"
            for var, value in css_overrides.items():
                css_script += f"--{var}: {value}px; "
            css_script += "`;"
            await page.evaluate(css_script)
            await page.wait_for_timeout(100)  # Wait for reflow
        
        # Find the report page element
        report_page = await page.query_selector('.report-page')
        
        if report_page:
            # Screenshot just the report page
            await report_page.screenshot(path=output_path)
        else:
            # Screenshot the full page
            await page.screenshot(path=output_path, full_page=True)
        
        await page.close()
        return output_path
    
    async def analyze_html(
        self,
        html_path: str,
        css_variables: Optional[Dict[str, float]] = None,
        priority: DesignPriority = DesignPriority.BALANCED,
        output_dir: Optional[str] = None
    ) -> DesignFeedback:
        """
        Analyze an HTML file's layout.
        
        Args:
            html_path: Path to HTML file
            css_variables: Current CSS variable values
            priority: Design priority
            output_dir: Directory for screenshots
        
        Returns:
            DesignFeedback with analysis
        """
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="html_analysis_")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Capture screenshot
        screenshot_path = os.path.join(output_dir, "preview.png")
        await self.capture_screenshot(html_path, screenshot_path, css_variables)
        
        # Create a mock LayoutParams-like object for the agent
        from .layout_params import LayoutParams
        params = LayoutParams()
        
        # Map CSS variables to layout params
        if css_variables:
            var_to_param = {
                "margin-left": "margin_left",
                "margin-right": "margin_right",
                "margin-top": "margin_top",
                "margin-bottom": "margin_bottom",
                "sidebar-width": "sidebar_width",
                "gutter-width": "gutter_width",
                "h1-size": "h1_font_size",
                "h2-size": "h2_font_size",
                "body-size": "body_font_size",
                "small-size": "small_font_size",
                "tiny-size": "tiny_font_size",
            }
            for css_var, param_name in var_to_param.items():
                if css_var in css_variables:
                    setattr(params, param_name, css_variables[css_var])
        
        # Analyze with the format agent
        feedback = self.agent.analyze_layout(
            image_paths=[screenshot_path],
            current_params=params,
            priority=priority,
            target_style="JPM Equity Research (HTML Preview)",
            page_descriptions=["HTML Preview"]
        )
        
        return feedback
    
    async def run_design_session(
        self,
        html_path: str,
        initial_css: Optional[Dict[str, float]] = None,
        priority: DesignPriority = DesignPriority.BALANCED,
        max_iterations: int = 5,
        target_score: float = 85.0,
        output_dir: str = "./html_design_sessions"
    ) -> Dict[str, Any]:
        """
        Run a design session for HTML layout optimization.
        
        Args:
            html_path: Path to HTML file
            initial_css: Initial CSS variable values
            priority: Design priority
            max_iterations: Max iterations
            target_score: Target score
            output_dir: Output directory
        
        Returns:
            Session results dictionary
        """
        import time
        
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(output_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        css_vars = initial_css or dict(self.CSS_VARIABLES)
        iterations = []
        
        print("\n" + "="*70)
        print("🎨 HTML FORMAT AGENT - Design Session Started")
        print("="*70)
        print(f"  Session ID: {session_id}")
        print(f"  HTML File: {html_path}")
        print(f"  Priority: {priority.value}")
        print("="*70 + "\n")
        
        try:
            await self._init_browser()
            
            for i in range(max_iterations):
                iter_start = time.time()
                print(f"\n📐 Iteration {i + 1}/{max_iterations}")
                print("-" * 50)
                
                # Capture and analyze
                print("  → Capturing screenshot...")
                screenshot_path = os.path.join(session_dir, f"iteration_{i+1}.png")
                await self.capture_screenshot(html_path, screenshot_path, css_vars)
                
                print("  → Analyzing with AI...")
                from .layout_params import LayoutParams
                params = LayoutParams()
                
                feedback = self.agent.analyze_layout(
                    image_paths=[screenshot_path],
                    current_params=params,
                    priority=priority,
                    page_descriptions=["HTML Preview"]
                )
                
                duration = time.time() - iter_start
                
                # Record iteration
                iterations.append({
                    "iteration": i + 1,
                    "css_variables": dict(css_vars),
                    "score": feedback.overall_score,
                    "issues": feedback.issues,
                    "adjustments": feedback.param_adjustments,
                    "duration": duration
                })
                
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
                
                if feedback.param_adjustments:
                    print(f"  🔧 CSS Adjustments ({len(feedback.param_adjustments)}):")
                    for param, delta in list(feedback.param_adjustments.items())[:5]:
                        css_name = param.replace("_", "-").replace("font-", "")
                        print(f"     • --{css_name}: {delta:+.1f}px")
                
                # Check termination
                if feedback.overall_score >= target_score:
                    print(f"\n  ✅ Target score reached!")
                    break
                
                if feedback.iteration_complete or not feedback.param_adjustments:
                    print("\n  ✅ Design complete")
                    break
                
                # Apply adjustments to CSS
                print("\n  → Applying CSS adjustments...")
                for param, delta in feedback.param_adjustments.items():
                    css_name = param.replace("_", "-").replace("font-", "")
                    if css_name in css_vars:
                        css_vars[css_name] += delta
                    elif param.replace("_", "-") in css_vars:
                        css_vars[param.replace("_", "-")] += delta
            
        finally:
            await self._close_browser()
        
        # Save final results
        final_score = iterations[-1]["score"] if iterations else 0
        
        result = {
            "session_id": session_id,
            "html_path": html_path,
            "priority": priority.value,
            "final_score": final_score,
            "final_css": css_vars,
            "iterations": iterations
        }
        
        with open(os.path.join(session_dir, "session.json"), "w") as f:
            json.dump(result, f, indent=2)
        
        # Generate CSS file
        css_output = ":root {\n"
        for var, value in css_vars.items():
            css_output += f"    --{var}: {value}px;\n"
        css_output += "}\n"
        
        with open(os.path.join(session_dir, "optimized.css"), "w") as f:
            f.write(css_output)
        
        print("\n" + "="*70)
        print("🎨 HTML DESIGN SESSION COMPLETE")
        print("="*70)
        print(f"  Final Score: {final_score:.1f}/100")
        print(f"  Iterations: {len(iterations)}")
        print(f"  Output: {session_dir}")
        print(f"\n  📄 Optimized CSS saved to: {session_dir}/optimized.css")
        print("="*70 + "\n")
        
        return result


def run_html_format_agent(
    html_path: str,
    provider: str = "anthropic",
    priority: str = "balanced",
    max_iterations: int = 5,
    output_dir: str = "./html_design_sessions"
) -> Dict[str, Any]:
    """
    Convenience function to run HTML format agent.
    
    Args:
        html_path: Path to HTML file
        provider: LLM provider
        priority: Design priority
        max_iterations: Max iterations
        output_dir: Output directory
    
    Returns:
        Session results
    """
    agent = HTMLFormatAgent(provider=provider)
    priority_enum = DesignPriority(priority)
    
    return asyncio.run(agent.run_design_session(
        html_path=html_path,
        priority=priority_enum,
        max_iterations=max_iterations,
        output_dir=output_dir
    ))


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python html_format_agent.py <html_file> [--provider xai]")
        sys.exit(1)
    
    html_path = sys.argv[1]
    provider = "xai"
    
    if "--provider" in sys.argv:
        idx = sys.argv.index("--provider")
        if idx + 1 < len(sys.argv):
            provider = sys.argv[idx + 1]
    
    result = run_html_format_agent(html_path, provider=provider)
    print(f"\nFinal Score: {result['final_score']}")
