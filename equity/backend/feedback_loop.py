"""
Visual Feedback Loop Controller

Orchestrates the render → screenshot → critique → adjust cycle
for automated PDF layout optimization.
"""

import os
import json
import shutil
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field

from .layout_params import LayoutParams
from .layout_critic import LayoutCritic, RuleBasedCritic, LayoutCritique


@dataclass
class FeedbackLoopConfig:
    """Configuration for the feedback loop."""
    max_iterations: int = 5
    target_score: float = 85.0
    use_llm_critic: bool = True
    use_rule_critic: bool = True  # Run rule-based checks first
    llm_provider: str = "xai"  # "xai", "openai", "anthropic"
    pages_to_analyze: List[int] = field(default_factory=lambda: [0, 1])
    render_zoom: float = 2.0
    save_iterations: bool = True
    output_dir: str = "./feedback_iterations"
    reference_pdf: Optional[str] = None  # Optional reference PDF for comparison


@dataclass
class IterationResult:
    """Result of a single iteration."""
    iteration: int
    params: LayoutParams
    pdf_path: str
    png_paths: List[str]
    critique: LayoutCritique
    duration_seconds: float


@dataclass
class FeedbackLoopResult:
    """Final result of the feedback loop."""
    success: bool
    final_params: LayoutParams
    final_pdf_path: str
    final_score: float
    iterations: List[IterationResult]
    total_duration_seconds: float
    summary: str


class FeedbackLoop:
    """
    Visual feedback loop for PDF layout optimization.
    
    The loop:
    1. Generates a PDF with current parameters
    2. Renders pages to PNG
    3. Critiques the layout (rule-based + LLM vision)
    4. Applies suggested adjustments
    5. Repeats until target score is reached or max iterations
    """
    
    def __init__(
        self,
        pdf_builder: Callable[[LayoutParams, str], None],
        config: Optional[FeedbackLoopConfig] = None
    ):
        """
        Initialize the feedback loop.
        
        Args:
            pdf_builder: Function that takes (LayoutParams, output_path) and generates PDF
            config: Loop configuration
        """
        self.pdf_builder = pdf_builder
        self.config = config or FeedbackLoopConfig()
        
        # Initialize critics
        self.rule_critic = RuleBasedCritic() if self.config.use_rule_critic else None
        self.llm_critic = None
        if self.config.use_llm_critic:
            try:
                self.llm_critic = LayoutCritic(provider=self.config.llm_provider)
                if not self.llm_critic.client:
                    print("⚠️  LLM critic not available (no API key)")
                    self.llm_critic = None
            except Exception as e:
                print(f"⚠️  Failed to initialize LLM critic: {e}")
        
        # Import renderer
        from .pdf_renderer import render_pdf_to_png, compare_pages_ssim, FITZ_AVAILABLE
        self.render_pdf = render_pdf_to_png
        self.compare_ssim = compare_pages_ssim
        self.fitz_available = FITZ_AVAILABLE
    
    def run(
        self,
        initial_params: Optional[LayoutParams] = None,
        report_data: Any = None
    ) -> FeedbackLoopResult:
        """
        Run the feedback loop.
        
        Args:
            initial_params: Starting layout parameters
            report_data: Data to pass to the PDF builder (if needed)
        
        Returns:
            FeedbackLoopResult with final optimized layout
        """
        import time
        
        start_time = time.time()
        
        # Setup
        params = initial_params or LayoutParams()
        iterations: List[IterationResult] = []
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("🔄 Starting Visual Feedback Loop")
        print("="*60)
        print(f"  Max iterations: {self.config.max_iterations}")
        print(f"  Target score: {self.config.target_score}")
        print(f"  LLM critic: {'enabled' if self.llm_critic else 'disabled'}")
        print(f"  Rule critic: {'enabled' if self.rule_critic else 'disabled'}")
        print("="*60 + "\n")
        
        previous_feedback = ""
        
        for i in range(self.config.max_iterations):
            iter_start = time.time()
            print(f"\n📄 Iteration {i + 1}/{self.config.max_iterations}")
            print("-" * 40)
            
            # 1. Generate PDF
            pdf_path = os.path.join(
                self.config.output_dir,
                f"iteration_{i + 1}.pdf"
            )
            
            print("  → Generating PDF...")
            try:
                # Call builder - it may take params + path, or have data bound
                self.pdf_builder(params, pdf_path)
            except Exception as e:
                print(f"  ❌ PDF generation failed: {e}")
                break
            
            # 2. Render to PNG
            print("  → Rendering pages to PNG...")
            png_dir = os.path.join(self.config.output_dir, f"iteration_{i + 1}_pages")
            
            if self.fitz_available:
                png_paths = self.render_pdf(
                    pdf_path,
                    png_dir,
                    pages=self.config.pages_to_analyze,
                    zoom=self.config.render_zoom
                )
            else:
                print("  ⚠️  PyMuPDF not available, skipping visual analysis")
                png_paths = []
            
            # 3. Run critiques
            critique = LayoutCritique(
                score=50.0,
                issues=[],
                suggestions=[],
                patch={},
                content_patch={},
                confidence=0.0,
                iteration_complete=False
            )
            
            # 3a. Rule-based critique
            if self.rule_critic and self.fitz_available:
                print("  → Running rule-based analysis...")
                rule_critique = self.rule_critic.critique(
                    pdf_path, params, self.config.pages_to_analyze
                )
                
                # Merge into main critique
                critique.issues.extend(rule_critique.issues)
                critique.suggestions.extend(rule_critique.suggestions)
                critique.patch.update(rule_critique.patch)
                critique.content_patch.update(rule_critique.content_patch)
                critique.score = rule_critique.score
            
            # 3b. LLM vision critique
            if self.llm_critic and png_paths:
                print("  → Running LLM visual analysis...")
                page_descs = [
                    "Summary Page" if p == 0 else f"Body Page {p}"
                    for p in self.config.pages_to_analyze
                ]
                
                llm_critique = self.llm_critic.critique(
                    png_paths,
                    params,
                    previous_feedback=previous_feedback,
                    page_descriptions=page_descs
                )
                
                # Merge critiques (LLM takes precedence for score)
                critique.issues.extend(llm_critique.issues)
                critique.suggestions.extend(llm_critique.suggestions)
                # Merge patches (LLM patches override rule-based)
                critique.patch.update(llm_critique.patch)
                critique.content_patch.update(llm_critique.content_patch)
                critique.score = (critique.score + llm_critique.score) / 2
                critique.confidence = max(critique.confidence, llm_critique.confidence)
                critique.iteration_complete = llm_critique.iteration_complete
            
            # 3c. SSIM comparison with reference (if provided)
            if self.config.reference_pdf and png_paths:
                print("  → Comparing with reference...")
                ref_png_dir = os.path.join(self.config.output_dir, "reference_pages")
                if not os.path.exists(ref_png_dir):
                    self.render_pdf(
                        self.config.reference_pdf,
                        ref_png_dir,
                        pages=self.config.pages_to_analyze,
                        zoom=self.config.render_zoom
                    )
                
                # Compare first page
                ref_page = os.path.join(ref_png_dir, "page_1.png")
                if os.path.exists(ref_page) and png_paths:
                    ssim_score = self.compare_ssim(png_paths[0], ref_page)
                    print(f"     SSIM similarity to reference: {ssim_score:.3f}")
            
            # Record iteration
            iter_duration = time.time() - iter_start
            iteration_result = IterationResult(
                iteration=i + 1,
                params=params,
                pdf_path=pdf_path,
                png_paths=png_paths,
                critique=critique,
                duration_seconds=iter_duration
            )
            iterations.append(iteration_result)
            
            # Print critique summary
            print(f"\n  📊 Score: {critique.score:.1f}/100")
            if critique.issues:
                print(f"  ⚠️  Issues ({len(critique.issues)}):")
                for issue in critique.issues[:5]:
                    print(f"     • {issue}")
            if critique.patch:
                print(f"  🔧 Adjustments ({len(critique.patch)}):")
                for param, delta in list(critique.patch.items())[:5]:
                    print(f"     • {param}: {delta:+.1f} pt")
            
            # Update previous feedback for next iteration
            previous_feedback = json.dumps({
                "score": critique.score,
                "issues": critique.issues[:3],
                "applied_patch": critique.patch
            })
            
            # 4. Check termination conditions
            if critique.score >= self.config.target_score:
                print(f"\n  ✅ Target score reached ({critique.score:.1f} >= {self.config.target_score})")
                break
            
            if critique.iteration_complete and not critique.patch:
                print("\n  ✅ Layout optimization complete (no more adjustments needed)")
                break
            
            if not critique.patch:
                print("\n  ⚠️  No adjustments suggested, stopping loop")
                break
            
            # 5. Apply adjustments
            print("\n  → Applying adjustments for next iteration...")
            params = params.apply_patch(critique.patch)
        
        # Finalize
        total_duration = time.time() - start_time
        final_iteration = iterations[-1] if iterations else None
        
        # Copy best result to final output
        final_pdf = os.path.join(self.config.output_dir, "final_optimized.pdf")
        if final_iteration:
            shutil.copy(final_iteration.pdf_path, final_pdf)
        
        # Generate summary
        summary_lines = [
            f"Feedback loop completed in {len(iterations)} iteration(s)",
            f"Total time: {total_duration:.1f}s",
            f"Final score: {final_iteration.critique.score:.1f}/100" if final_iteration else "N/A",
        ]
        if final_iteration and final_iteration.critique.issues:
            summary_lines.append(f"Remaining issues: {len(final_iteration.critique.issues)}")
        
        result = FeedbackLoopResult(
            success=final_iteration.critique.score >= self.config.target_score if final_iteration else False,
            final_params=params,
            final_pdf_path=final_pdf,
            final_score=final_iteration.critique.score if final_iteration else 0,
            iterations=iterations,
            total_duration_seconds=total_duration,
            summary="\n".join(summary_lines)
        )
        
        # Save results
        self._save_results(result)
        
        print("\n" + "="*60)
        print("🏁 Feedback Loop Complete")
        print("="*60)
        print(result.summary)
        print(f"\n📁 Results saved to: {self.config.output_dir}")
        print(f"📄 Final PDF: {final_pdf}")
        print("="*60 + "\n")
        
        return result
    
    def _save_results(self, result: FeedbackLoopResult):
        """Save loop results to JSON."""
        results_path = os.path.join(self.config.output_dir, "results.json")
        
        data = {
            "success": result.success,
            "final_score": result.final_score,
            "total_duration_seconds": result.total_duration_seconds,
            "iteration_count": len(result.iterations),
            "final_params": result.final_params.to_dict(),
            "iterations": [
                {
                    "iteration": ir.iteration,
                    "score": ir.critique.score,
                    "issues": ir.critique.issues,
                    "patch": ir.critique.patch,
                    "duration_seconds": ir.duration_seconds
                }
                for ir in result.iterations
            ],
            "summary": result.summary,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(results_path, "w") as f:
            json.dump(data, f, indent=2)


# Convenience function
def optimize_pdf_layout(
    pdf_builder: Callable[[LayoutParams, str], None],
    initial_params: Optional[LayoutParams] = None,
    max_iterations: int = 5,
    target_score: float = 85.0,
    output_dir: str = "./feedback_iterations",
    reference_pdf: Optional[str] = None
) -> FeedbackLoopResult:
    """
    Convenience function to run the feedback loop.
    
    Args:
        pdf_builder: Function that generates PDF from (params, output_path)
        initial_params: Starting layout parameters
        max_iterations: Maximum optimization iterations
        target_score: Target quality score (0-100)
        output_dir: Directory for iteration outputs
        reference_pdf: Optional reference PDF to match style
    
    Returns:
        FeedbackLoopResult with optimized layout
    """
    config = FeedbackLoopConfig(
        max_iterations=max_iterations,
        target_score=target_score,
        output_dir=output_dir,
        reference_pdf=reference_pdf
    )
    
    loop = FeedbackLoop(pdf_builder, config)
    return loop.run(initial_params)


if __name__ == "__main__":
    print("Visual Feedback Loop for PDF Layout Optimization")
    print("-" * 50)
    print("\nUsage:")
    print("  from backend.feedback_loop import optimize_pdf_layout")
    print("  ")
    print("  def my_pdf_builder(params, output_path):")
    print("      # Your PDF generation code here")
    print("      pass")
    print("  ")
    print("  result = optimize_pdf_layout(my_pdf_builder)")
    print("  print(f'Final score: {result.final_score}')")

