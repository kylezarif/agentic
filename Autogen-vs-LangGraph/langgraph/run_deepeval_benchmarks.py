"""
LangGraph Framework Evaluation using DeepEval Benchmarks
========================================================
Runs comprehensive evaluation on GSM8K, HumanEval, ARC, and MATH benchmarks
using DeepEval's framework with custom GEval metrics.
"""

import os
import sys
import json
import datetime
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.benchmarks import GSM8K, HumanEval
from deepeval.dataset import Golden

# Import custom components
from deepeval_langgraph_model import create_langgraph_model
from deepeval_metrics import (
    get_metrics_for_benchmark,
    correctness_metric,
    reasoning_quality_metric,
    tool_usage_metric
)

# Load environment
load_dotenv(dotenv_path=Path("../.env").expanduser(), override=True)


class LangGraphBenchmarkEvaluator:
    """Orchestrates evaluation of LangGraph on multiple benchmarks."""

    def __init__(
        self,
        n_problems: int = 10,
        n_shots: int = 3,
        temperature: float = 0.3,
        repetitions: int = 1,
        output_dir: str = "results/single_agent"
    ):
        self.n_problems = n_problems
        self.n_shots = n_shots
        self.temperature = temperature
        self.repetitions = repetitions
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create model wrapper
        self.model = create_langgraph_model(temperature=temperature)

    def evaluate_gsm8k(self) -> Dict[str, Any]:
        """Evaluate LangGraph on GSM8K benchmark."""
        print("\n" + "=" * 70)
        print("EVALUATING LANGGRAPH ON GSM8K (Math Reasoning)")
        print("=" * 70)

        results_all_reps = []

        for rep in range(self.repetitions):
            print(f"\nRepetition {rep + 1}/{self.repetitions}")

            # Create benchmark
            benchmark = GSM8K(
                n_problems=self.n_problems,
                n_shots=self.n_shots,
                enable_cot=True  # Enable chain-of-thought
            )

            # Evaluate
            benchmark.evaluate(model=self.model)

            # Get results
            result = {
                "repetition": rep + 1,
                "overall_score": benchmark.overall_score,
                "n_problems": self.n_problems,
                "benchmark": "GSM8K"
            }
            results_all_reps.append(result)

            print(f"Score: {benchmark.overall_score:.3f}")

        # Aggregate results
        avg_score = sum(r["overall_score"] for r in results_all_reps) / len(results_all_reps)

        summary = {
            "benchmark": "GSM8K",
            "framework": "LangGraph",
            "agent_type": "single_agent",
            "date": datetime.date.today().isoformat(),
            "config": {
                "n_problems": self.n_problems,
                "n_shots": self.n_shots,
                "temperature": self.temperature,
                "repetitions": self.repetitions
            },
            "results": {
                "average_score": avg_score,
                "all_repetitions": results_all_reps
            }
        }

        # Save results
        self._save_results(summary, "gsm8k")

        print(f"\n✓ GSM8K Evaluation Complete: Average Score = {avg_score:.3f}")
        return summary

    def evaluate_humaneval(self, k: int = 10) -> Dict[str, Any]:
        """Evaluate LangGraph on HumanEval benchmark."""
        print("\n" + "=" * 70)
        print("EVALUATING LANGGRAPH ON HUMANEVAL (Code Generation)")
        print("=" * 70)

        results_all_reps = []

        for rep in range(self.repetitions):
            print(f"\nRepetition {rep + 1}/{self.repetitions}")

            # Create benchmark - evaluate all tasks with n generations
            benchmark = HumanEval(
                n=self.n_problems  # Number of code generations per problem
            )

            # Evaluate with pass@k metric
            benchmark.evaluate(model=self.model, k=k)

            result = {
                "repetition": rep + 1,
                "overall_score": benchmark.overall_score,
                "n_generations": self.n_problems,
                "pass_at_k": k,
                "benchmark": "HumanEval"
            }
            results_all_reps.append(result)

            print(f"Pass@{k} Score: {benchmark.overall_score:.3f}")

        # Aggregate results
        avg_score = sum(r["overall_score"] for r in results_all_reps) / len(results_all_reps)

        summary = {
            "benchmark": "HumanEval",
            "framework": "LangGraph",
            "agent_type": "single_agent",
            "date": datetime.date.today().isoformat(),
            "config": {
                "n_generations": self.n_problems,
                "pass_at_k": k,
                "temperature": self.temperature,
                "repetitions": self.repetitions
            },
            "results": {
                "average_score": avg_score,
                "all_repetitions": results_all_reps
            }
        }

        # Save results
        self._save_results(summary, "humaneval")

        print(f"\n✓ HumanEval Evaluation Complete: Average Pass@{k} = {avg_score:.3f}")
        return summary

    def evaluate_with_custom_metrics(
        self,
        test_cases: List[LLMTestCase],
        benchmark_name: str
    ) -> Dict[str, Any]:
        """
        Evaluate test cases with custom GEval metrics.
        Used for benchmarks that need custom evaluation logic.
        """
        print(f"\n" + "=" * 70)
        print(f"EVALUATING LANGGRAPH WITH CUSTOM GEVAL METRICS: {benchmark_name.upper()}")
        print("=" * 70)

        # Get appropriate metrics for this benchmark
        metrics = get_metrics_for_benchmark(benchmark_name, is_multi_agent=False)

        results_all_reps = []

        for rep in range(self.repetitions):
            print(f"\nRepetition {rep + 1}/{self.repetitions}")

            # Run evaluation
            eval_results = evaluate(
                test_cases=test_cases,
                metrics=metrics,
                print_results=True
            )

            # Collect scores for each metric
            metric_scores = {}
            for metric in metrics:
                scores = [tc.metrics_data[metric.name].score for tc in test_cases if metric.name in tc.metrics_data]
                if scores:
                    metric_scores[metric.name] = {
                        "average": sum(scores) / len(scores),
                        "min": min(scores),
                        "max": max(scores)
                    }

            result = {
                "repetition": rep + 1,
                "metric_scores": metric_scores,
                "benchmark": benchmark_name
            }
            results_all_reps.append(result)

        summary = {
            "benchmark": benchmark_name,
            "framework": "LangGraph",
            "agent_type": "single_agent",
            "date": datetime.date.today().isoformat(),
            "config": {
                "n_problems": len(test_cases),
                "temperature": self.temperature,
                "repetitions": self.repetitions
            },
            "results": {
                "all_repetitions": results_all_reps
            }
        }

        # Save results
        self._save_results(summary, benchmark_name.lower())

        print(f"\n✓ {benchmark_name} Custom Evaluation Complete")
        return summary

    def _save_results(self, summary: Dict[str, Any], benchmark_name: str):
        """Save evaluation results to JSON file."""
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        filename = f"langgraph_{benchmark_name}_results_{timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Results saved to: {filepath}")

    def run_all_benchmarks(self):
        """Run all benchmark evaluations."""
        print("\n" + "=" * 70)
        print("LANGGRAPH FRAMEWORK - COMPREHENSIVE BENCHMARK EVALUATION")
        print("=" * 70)
        print(f"Configuration:")
        print(f"  - Problems per benchmark: {self.n_problems}")
        print(f"  - Repetitions: {self.repetitions}")
        print(f"  - Temperature: {self.temperature}")
        print(f"  - Model: {self.model.get_model_name()}")
        print("=" * 70)

        all_results = {}

        # Run GSM8K
        try:
            all_results["gsm8k"] = self.evaluate_gsm8k()
        except Exception as e:
            print(f"\n✗ GSM8K Evaluation Failed: {e}")
            all_results["gsm8k"] = {"error": str(e)}

        # Run HumanEval
        try:
            all_results["humaneval"] = self.evaluate_humaneval(k=10)
        except Exception as e:
            print(f"\n✗ HumanEval Evaluation Failed: {e}")
            all_results["humaneval"] = {"error": str(e)}

        # Save consolidated results
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        summary_file = self.output_dir / f"langgraph_all_benchmarks_{timestamp}.json"

        with open(summary_file, "w") as f:
            json.dump(all_results, f, indent=2)

        print("\n" + "=" * 70)
        print("EVALUATION COMPLETE")
        print("=" * 70)
        print(f"Consolidated results saved to: {summary_file}")

        return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LangGraph framework on multiple benchmarks using DeepEval"
    )
    parser.add_argument(
        "--n-problems",
        type=int,
        default=10,
        help="Number of problems to evaluate per benchmark (default: 10)"
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Number of times to repeat each benchmark (default: 3)"
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
        default="results/single_agent",
        help="Output directory for results (default: results/single_agent)"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["all", "gsm8k", "humaneval"],
        default="all",
        help="Which benchmark to run (default: all)"
    )

    args = parser.parse_args()

    # Create evaluator
    evaluator = LangGraphBenchmarkEvaluator(
        n_problems=args.n_problems,
        n_shots=3,
        temperature=args.temperature,
        repetitions=args.repetitions,
        output_dir=args.output_dir
    )

    # Run requested benchmarks
    if args.benchmark == "all":
        evaluator.run_all_benchmarks()
    elif args.benchmark == "gsm8k":
        evaluator.evaluate_gsm8k()
    elif args.benchmark == "humaneval":
        evaluator.evaluate_humaneval()


if __name__ == "__main__":
    main()
