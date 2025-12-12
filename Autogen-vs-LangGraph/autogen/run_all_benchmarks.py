"""
AutoGen Framework - Complete Benchmark Evaluation Suite
=======================================================
Evaluates AutoGen on GSM8K, HumanEval, ARC, and MATH benchmarks
using DeepEval with custom GEval metrics.
"""

import os
import sys
import json
import datetime
import argparse
from pathlib import Path
from typing import List, Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.benchmarks import GSM8K, HumanEval

try:
    from deepeval.benchmarks import ARC, MMLU
    ARC_AVAILABLE = True
except ImportError:
    ARC_AVAILABLE = False
    print("⚠️  ARC benchmark not available. Install with: pip install deepeval[arc]")

# Import custom components
from deepeval_autogen_model import create_autogen_model
from deepeval_metrics import (
    get_metrics_for_benchmark,
    SINGLE_AGENT_METRICS
)

load_dotenv(dotenv_path=Path("../.env").expanduser(), override=True)


class CompleteBenchmarkEvaluator:
    """Complete benchmark evaluation suite for AutoGen."""

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

        self.model = create_autogen_model(temperature=temperature)

    def evaluate_gsm8k(self) -> Dict[str, Any]:
        """Evaluate on GSM8K - Math reasoning."""
        print("\n" + "="*70)
        print("BENCHMARK: GSM8K (Math Reasoning)")
        print("="*70)

        results_all_reps = []

        for rep in range(self.repetitions):
            print(f"\nRepetition {rep + 1}/{self.repetitions}")

            benchmark = GSM8K(
                n_problems=self.n_problems,
                n_shots=self.n_shots,
                enable_cot=True
            )

            benchmark.evaluate(model=self.model)

            result = {
                "repetition": rep + 1,
                "overall_score": benchmark.overall_score,
                "n_problems": self.n_problems
            }
            results_all_reps.append(result)
            print(f"✓ Score: {benchmark.overall_score:.3f}")

        avg_score = sum(r["overall_score"] for r in results_all_reps) / len(results_all_reps)

        summary = {
            "benchmark": "GSM8K",
            "framework": "AutoGen",
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

        self._save_results(summary, "gsm8k")
        print(f"\n✅ GSM8K Complete: Avg = {avg_score:.3f}")
        return summary

    def evaluate_humaneval(self, k: int = 10) -> Dict[str, Any]:
        """Evaluate on HumanEval - Code generation."""
        print("\n" + "="*70)
        print("BENCHMARK: HumanEval (Code Generation)")
        print("="*70)

        results_all_reps = []

        for rep in range(self.repetitions):
            print(f"\nRepetition {rep + 1}/{self.repetitions}")

            benchmark = HumanEval(n=self.n_problems)
            benchmark.evaluate(model=self.model, k=k)

            result = {
                "repetition": rep + 1,
                "overall_score": benchmark.overall_score,
                "pass_at_k": k
            }
            results_all_reps.append(result)
            print(f"✓ Pass@{k}: {benchmark.overall_score:.3f}")

        avg_score = sum(r["overall_score"] for r in results_all_reps) / len(results_all_reps)

        summary = {
            "benchmark": "HumanEval",
            "framework": "AutoGen",
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

        self._save_results(summary, "humaneval")
        print(f"\n✅ HumanEval Complete: Avg Pass@{k} = {avg_score:.3f}")
        return summary

    def evaluate_arc(self) -> Dict[str, Any]:
        """Evaluate on ARC - Multi-step reasoning."""
        if not ARC_AVAILABLE:
            print("\n⚠️  ARC benchmark not available")
            return {"status": "skipped", "reason": "ARC not installed"}

        print("\n" + "="*70)
        print("BENCHMARK: ARC (Multi-step Reasoning)")
        print("="*70)

        results_all_reps = []

        for rep in range(self.repetitions):
            print(f"\nRepetition {rep + 1}/{self.repetitions}")

            # ARC has Challenge and Easy sets
            benchmark = ARC(
                n_problems=self.n_problems,
                n_shots=self.n_shots,
                # tasks can be specified if needed
            )

            benchmark.evaluate(model=self.model)

            result = {
                "repetition": rep + 1,
                "overall_score": benchmark.overall_score,
                "n_problems": self.n_problems
            }
            results_all_reps.append(result)
            print(f"✓ Score: {benchmark.overall_score:.3f}")

        avg_score = sum(r["overall_score"] for r in results_all_reps) / len(results_all_reps)

        summary = {
            "benchmark": "ARC",
            "framework": "AutoGen",
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

        self._save_results(summary, "arc")
        print(f"\n✅ ARC Complete: Avg = {avg_score:.3f}")
        return summary

    def evaluate_math_custom(self) -> Dict[str, Any]:
        """
        Evaluate on MATH dataset (custom implementation).
        Since DeepEval's MATH might need specific setup, we implement custom loading.
        """
        print("\n" + "="*70)
        print("BENCHMARK: MATH (Complex Problem Solving)")
        print("="*70)

        try:
            from datasets import load_dataset

            # Load MATH dataset from HuggingFace
            print("Loading MATH dataset...")
            dataset = load_dataset("hendrycks/math", split="test")
            dataset = dataset.select(range(min(self.n_problems, len(dataset))))

            results_all_reps = []

            for rep in range(self.repetitions):
                print(f"\nRepetition {rep + 1}/{self.repetitions}")

                correct = 0
                test_cases = []

                for i, item in enumerate(dataset):
                    problem = item["problem"]
                    solution = item["solution"]

                    # Generate response using our model
                    prompt = f"Solve this math problem step by step:\n\n{problem}"
                    response = self.model.generate(prompt)

                    # Create test case for GEval metrics
                    test_case = LLMTestCase(
                        input=problem,
                        actual_output=response,
                        expected_output=solution
                    )
                    test_cases.append(test_case)

                    # Simple correctness check (can be improved)
                    if solution.strip().lower() in response.strip().lower():
                        correct += 1

                    if (i + 1) % 10 == 0:
                        print(f"  Progress: {i + 1}/{len(dataset)}")

                accuracy = correct / len(dataset)

                # Evaluate with GEval metrics
                metrics = get_metrics_for_benchmark("math", is_multi_agent=False)
                evaluate(test_cases=test_cases, metrics=metrics, print_results=False)

                # Collect metric scores
                metric_scores = {}
                for metric in metrics:
                    scores = [tc.metrics_data[metric.name].score
                             for tc in test_cases if metric.name in tc.metrics_data]
                    if scores:
                        metric_scores[metric.name] = sum(scores) / len(scores)

                result = {
                    "repetition": rep + 1,
                    "accuracy": accuracy,
                    "metric_scores": metric_scores
                }
                results_all_reps.append(result)
                print(f"✓ Accuracy: {accuracy:.3f}")

            avg_accuracy = sum(r["accuracy"] for r in results_all_reps) / len(results_all_reps)

            summary = {
                "benchmark": "MATH",
                "framework": "AutoGen",
                "date": datetime.date.today().isoformat(),
                "config": {
                    "n_problems": self.n_problems,
                    "temperature": self.temperature,
                    "repetitions": self.repetitions
                },
                "results": {
                    "average_accuracy": avg_accuracy,
                    "all_repetitions": results_all_reps
                }
            }

            self._save_results(summary, "math")
            print(f"\n✅ MATH Complete: Avg Accuracy = {avg_accuracy:.3f}")
            return summary

        except Exception as e:
            print(f"\n✗ MATH evaluation failed: {e}")
            return {"status": "failed", "error": str(e)}

    def _save_results(self, summary: Dict[str, Any], benchmark_name: str):
        """Save results to JSON file."""
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        filename = f"autogen_{benchmark_name}_results_{timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"💾 Saved: {filepath}")

    def run_all_benchmarks(self):
        """Run all available benchmarks."""
        print("\n" + "="*70)
        print("AUTOGEN - COMPLETE BENCHMARK SUITE")
        print("="*70)
        print(f"Configuration:")
        print(f"  Model: {self.model.get_model_name()}")
        print(f"  Problems: {self.n_problems}")
        print(f"  Repetitions: {self.repetitions}")
        print(f"  Temperature: {self.temperature}")
        print("="*70)

        all_results = {}

        # GSM8K
        try:
            all_results["gsm8k"] = self.evaluate_gsm8k()
        except Exception as e:
            print(f"\n✗ GSM8K failed: {e}")
            all_results["gsm8k"] = {"error": str(e)}

        # HumanEval
        try:
            all_results["humaneval"] = self.evaluate_humaneval(k=10)
        except Exception as e:
            print(f"\n✗ HumanEval failed: {e}")
            all_results["humaneval"] = {"error": str(e)}

        # ARC
        try:
            all_results["arc"] = self.evaluate_arc()
        except Exception as e:
            print(f"\n✗ ARC failed: {e}")
            all_results["arc"] = {"error": str(e)}

        # MATH
        try:
            all_results["math"] = self.evaluate_math_custom()
        except Exception as e:
            print(f"\n✗ MATH failed: {e}")
            all_results["math"] = {"error": str(e)}

        # Save consolidated results
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        summary_file = self.output_dir / f"autogen_all_benchmarks_{timestamp}.json"

        with open(summary_file, "w") as f:
            json.dump(all_results, f, indent=2)

        print("\n" + "="*70)
        print("✅ EVALUATION COMPLETE")
        print("="*70)
        print(f"Summary: {summary_file}")

        return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Complete benchmark evaluation for AutoGen framework"
    )
    parser.add_argument("--n-problems", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--n-shots", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="results/single_agent")
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["all", "gsm8k", "humaneval", "arc", "math"],
        default="all"
    )

    args = parser.parse_args()

    evaluator = CompleteBenchmarkEvaluator(
        n_problems=args.n_problems,
        n_shots=args.n_shots,
        temperature=args.temperature,
        repetitions=args.repetitions,
        output_dir=args.output_dir
    )

    if args.benchmark == "all":
        evaluator.run_all_benchmarks()
    elif args.benchmark == "gsm8k":
        evaluator.evaluate_gsm8k()
    elif args.benchmark == "humaneval":
        evaluator.evaluate_humaneval()
    elif args.benchmark == "arc":
        evaluator.evaluate_arc()
    elif args.benchmark == "math":
        evaluator.evaluate_math_custom()


if __name__ == "__main__":
    main()
