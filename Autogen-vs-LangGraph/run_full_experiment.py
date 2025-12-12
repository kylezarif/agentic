"""
Master Orchestration Script for Agentic AI Framework Comparison
================================================================
Runs comprehensive evaluation of both AutoGen and LangGraph frameworks
on all benchmarks with statistical analysis and comparison.
"""

import os
import sys
import json
import datetime
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List
import statistics


class ExperimentOrchestrator:
    """
    Orchestrates the complete experimental study comparing
    AutoGen and LangGraph frameworks.
    """

    def __init__(
        self,
        n_problems: int = 10,
        repetitions: int = 3,
        temperature: float = 0.3,
        output_dir: str = "results/comparative_study"
    ):
        self.n_problems = n_problems
        self.repetitions = repetitions
        self.temperature = temperature
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_autogen_experiments(self) -> Dict[str, Any]:
        """Run all AutoGen experiments."""
        print("\n" + "=" * 80)
        print("PHASE 1: AUTOGEN FRAMEWORK EVALUATION")
        print("=" * 80)

        cmd = [
            "python",
            "autogen/run_deepeval_benchmarks.py",
            "--n-problems", str(self.n_problems),
            "--repetitions", str(self.repetitions),
            "--temperature", str(self.temperature),
            "--benchmark", "all"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(result.stdout)

            # Load results
            results_dir = Path("autogen/results/single_agent")
            latest_results = self._get_latest_results(results_dir, "autogen_all_benchmarks")

            return latest_results if latest_results else {"status": "completed", "note": "Check autogen/results/"}

        except subprocess.CalledProcessError as e:
            print(f"✗ AutoGen experiments failed: {e}")
            print(e.stderr)
            return {"status": "failed", "error": str(e)}

    def run_langgraph_experiments(self) -> Dict[str, Any]:
        """Run all LangGraph experiments."""
        print("\n" + "=" * 80)
        print("PHASE 2: LANGGRAPH FRAMEWORK EVALUATION")
        print("=" * 80)

        cmd = [
            "python",
            "langgraph/run_deepeval_benchmarks.py",
            "--n-problems", str(self.n_problems),
            "--repetitions", str(self.repetitions),
            "--temperature", str(self.temperature),
            "--benchmark", "all"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(result.stdout)

            # Load results
            results_dir = Path("langgraph/results/single_agent")
            latest_results = self._get_latest_results(results_dir, "langgraph_all_benchmarks")

            return latest_results if latest_results else {"status": "completed", "note": "Check langgraph/results/"}

        except subprocess.CalledProcessError as e:
            print(f"✗ LangGraph experiments failed: {e}")
            print(e.stderr)
            return {"status": "failed", "error": str(e)}

    def _get_latest_results(self, results_dir: Path, prefix: str) -> Dict[str, Any]:
        """Get the latest results file matching the prefix."""
        matching_files = list(results_dir.glob(f"{prefix}_*.json"))

        if not matching_files:
            return None

        # Sort by modification time and get latest
        latest_file = max(matching_files, key=lambda p: p.stat().st_mtime)

        with open(latest_file, "r") as f:
            return json.load(f)

    def compare_results(
        self,
        autogen_results: Dict[str, Any],
        langgraph_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare results between AutoGen and LangGraph."""
        print("\n" + "=" * 80)
        print("PHASE 3: COMPARATIVE ANALYSIS")
        print("=" * 80)

        comparison = {
            "study_metadata": {
                "date": datetime.date.today().isoformat(),
                "n_problems_per_benchmark": self.n_problems,
                "repetitions_per_task": self.repetitions,
                "temperature": self.temperature,
            },
            "benchmarks": {}
        }

        # Compare each benchmark
        for benchmark in ["gsm8k", "humaneval"]:
            if benchmark in autogen_results and benchmark in langgraph_results:
                ag_data = autogen_results[benchmark]
                lg_data = langgraph_results[benchmark]

                if "results" in ag_data and "results" in lg_data:
                    ag_scores = [r["overall_score"] for r in ag_data["results"].get("all_repetitions", [])]
                    lg_scores = [r["overall_score"] for r in lg_data["results"].get("all_repetitions", [])]

                    if ag_scores and lg_scores:
                        comparison["benchmarks"][benchmark] = {
                            "autogen": {
                                "mean": statistics.mean(ag_scores),
                                "stdev": statistics.stdev(ag_scores) if len(ag_scores) > 1 else 0,
                                "min": min(ag_scores),
                                "max": max(ag_scores),
                                "scores": ag_scores
                            },
                            "langgraph": {
                                "mean": statistics.mean(lg_scores),
                                "stdev": statistics.stdev(lg_scores) if len(lg_scores) > 1 else 0,
                                "min": min(lg_scores),
                                "max": max(lg_scores),
                                "scores": lg_scores
                            },
                            "comparison": {
                                "mean_difference": statistics.mean(lg_scores) - statistics.mean(ag_scores),
                                "winner": "LangGraph" if statistics.mean(lg_scores) > statistics.mean(ag_scores) else "AutoGen"
                            }
                        }

        # Print summary
        print("\nComparative Summary:")
        print("-" * 80)

        for benchmark, data in comparison["benchmarks"].items():
            print(f"\n{benchmark.upper()}:")
            print(f"  AutoGen:   {data['autogen']['mean']:.3f} (±{data['autogen']['stdev']:.3f})")
            print(f"  LangGraph: {data['langgraph']['mean']:.3f} (±{data['langgraph']['stdev']:.3f})")
            print(f"  Winner:    {data['comparison']['winner']}")
            print(f"  Difference: {abs(data['comparison']['mean_difference']):.3f}")

        # Save comparison
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        comparison_file = self.output_dir / f"framework_comparison_{timestamp}.json"

        with open(comparison_file, "w") as f:
            json.dump(comparison, f, indent=2)

        print(f"\n✓ Comparison saved to: {comparison_file}")

        return comparison

    def run_full_study(self):
        """Execute the complete experimental study."""
        print("\n" + "=" * 80)
        print("AGENTIC AI FRAMEWORKS COMPARATIVE STUDY")
        print("AutoGen vs LangGraph")
        print("=" * 80)
        print(f"\nExperimental Configuration:")
        print(f"  - Benchmarks: GSM8K, HumanEval")
        print(f"  - Problems per benchmark: {self.n_problems}")
        print(f"  - Repetitions: {self.repetitions}")
        print(f"  - Temperature: {self.temperature}")
        print(f"  - Model: gpt-4o-mini")
        print("=" * 80)

        # Phase 1: AutoGen
        autogen_results = self.run_autogen_experiments()

        # Phase 2: LangGraph
        langgraph_results = self.run_langgraph_experiments()

        # Phase 3: Comparison
        if autogen_results and langgraph_results:
            comparison = self.compare_results(autogen_results, langgraph_results)

            # Save final study report
            study_report = {
                "metadata": {
                    "study_title": "Comparative Evaluation of Agentic AI Frameworks",
                    "frameworks": ["AutoGen", "LangGraph"],
                    "date": datetime.date.today().isoformat(),
                    "configuration": {
                        "n_problems": self.n_problems,
                        "repetitions": self.repetitions,
                        "temperature": self.temperature
                    }
                },
                "autogen_results": autogen_results,
                "langgraph_results": langgraph_results,
                "comparison": comparison
            }

            timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
            report_file = self.output_dir / f"full_study_report_{timestamp}.json"

            with open(report_file, "w") as f:
                json.dump(study_report, f, indent=2)

            print("\n" + "=" * 80)
            print("STUDY COMPLETE")
            print("=" * 80)
            print(f"Full study report saved to: {report_file}")

        else:
            print("\n✗ Study incomplete - check individual framework results")


def main():
    parser = argparse.ArgumentParser(
        description="Run complete comparative study of AutoGen and LangGraph frameworks"
    )
    parser.add_argument(
        "--n-problems",
        type=int,
        default=10,
        help="Number of problems per benchmark (default: 10, recommended: 50-100 for full study)"
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Number of repetitions per task (default: 3, recommended: 5-10 for full study)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Model temperature (default: 0.3)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/comparative_study",
        help="Output directory for comparison results"
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run pilot study with minimal problems (n=5, reps=2)"
    )

    args = parser.parse_args()

    # Adjust for pilot study
    if args.pilot:
        print("\n🚀 Running PILOT STUDY (reduced scale)")
        args.n_problems = 5
        args.repetitions = 2

    # Create orchestrator
    orchestrator = ExperimentOrchestrator(
        n_problems=args.n_problems,
        repetitions=args.repetitions,
        temperature=args.temperature,
        output_dir=args.output_dir
    )

    # Run full study
    orchestrator.run_full_study()


if __name__ == "__main__":
    main()
