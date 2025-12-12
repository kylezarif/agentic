"""
Results Visualization and Analysis
==================================
Creates visualizations and statistical analysis for the experimental study
comparing AutoGen and LangGraph frameworks.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any
import statistics


def load_results(results_dir: Path, pattern: str) -> List[Dict[str, Any]]:
    """Load all result JSON files matching pattern."""
    result_files = list(results_dir.glob(pattern))

    if not result_files:
        print(f"⚠️  No result files found matching: {pattern}")
        return []

    results = []
    for file_path in sorted(result_files):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"✗ Error loading {file_path}: {e}")

    return results


def analyze_single_agent_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze single-agent evaluation results."""
    analysis = {}

    for result in results:
        benchmark = result.get("benchmark", "unknown")
        framework = result.get("framework", "unknown")

        key = f"{framework}_{benchmark}"

        if "results" in result:
            if "average_score" in result["results"]:
                score = result["results"]["average_score"]
            elif "average_accuracy" in result["results"]:
                score = result["results"]["average_accuracy"]
            else:
                continue

            if key not in analysis:
                analysis[key] = {
                    "framework": framework,
                    "benchmark": benchmark,
                    "scores": []
                }

            analysis[key]["scores"].append(score)

    # Calculate statistics
    for key, data in analysis.items():
        scores = data["scores"]
        if scores:
            data["mean"] = statistics.mean(scores)
            data["stdev"] = statistics.stdev(scores) if len(scores) > 1 else 0
            data["min"] = min(scores)
            data["max"] = max(scores)
            data["count"] = len(scores)

    return analysis


def analyze_multi_agent_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze multi-agent evaluation results."""
    analysis = {}

    for result in results:
        benchmark = result.get("benchmark", "unknown")
        framework = result.get("framework", "unknown")

        key = f"{framework}_{benchmark}"

        if "results" in result and "all_repetitions" in result["results"]:
            for rep_data in result["results"]["all_repetitions"]:
                if "metric_scores" in rep_data:
                    for metric_name, metric_data in rep_data["metric_scores"].items():
                        metric_key = f"{key}_{metric_name}"

                        if metric_key not in analysis:
                            analysis[metric_key] = {
                                "framework": framework,
                                "benchmark": benchmark,
                                "metric": metric_name,
                                "scores": []
                            }

                        score = metric_data.get("mean", metric_data) if isinstance(metric_data, dict) else metric_data
                        analysis[metric_key]["scores"].append(score)

    # Calculate statistics
    for key, data in analysis.items():
        scores = data["scores"]
        if scores:
            data["mean"] = statistics.mean(scores)
            data["stdev"] = statistics.stdev(scores) if len(scores) > 1 else 0
            data["min"] = min(scores)
            data["max"] = max(scores)
            data["count"] = len(scores)

    return analysis


def compare_frameworks(
    autogen_analysis: Dict[str, Any],
    langgraph_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare AutoGen and LangGraph results."""

    comparison = {}

    # Find common benchmarks
    autogen_benchmarks = {data["benchmark"] for data in autogen_analysis.values()}
    langgraph_benchmarks = {data["benchmark"] for data in langgraph_analysis.values()}
    common_benchmarks = autogen_benchmarks & langgraph_benchmarks

    for benchmark in common_benchmarks:
        # Get AutoGen data for this benchmark
        autogen_data = next(
            (data for key, data in autogen_analysis.items()
             if data["benchmark"] == benchmark),
            None
        )

        # Get LangGraph data for this benchmark
        langgraph_data = next(
            (data for key, data in langgraph_analysis.items()
             if data["benchmark"] == benchmark),
            None
        )

        if autogen_data and langgraph_data:
            autogen_mean = autogen_data.get("mean", 0)
            langgraph_mean = langgraph_data.get("mean", 0)

            comparison[benchmark] = {
                "autogen": {
                    "mean": autogen_mean,
                    "stdev": autogen_data.get("stdev", 0),
                    "count": autogen_data.get("count", 0)
                },
                "langgraph": {
                    "mean": langgraph_mean,
                    "stdev": langgraph_data.get("stdev", 0),
                    "count": langgraph_data.get("count", 0)
                },
                "difference": langgraph_mean - autogen_mean,
                "winner": "LangGraph" if langgraph_mean > autogen_mean else "AutoGen",
                "percent_improvement": ((langgraph_mean - autogen_mean) / autogen_mean * 100)
                                      if autogen_mean > 0 else 0
            }

    return comparison


def generate_markdown_report(
    single_agent_analysis: Dict[str, Any],
    multi_agent_analysis: Dict[str, Any],
    comparison: Dict[str, Any],
    output_file: Path
):
    """Generate a markdown report of results."""

    with open(output_file, "w") as f:
        f.write("# Experimental Study Results: AutoGen vs LangGraph\n\n")
        f.write(f"Generated: {Path(__file__).parent.name}\n\n")

        # Single-Agent Results
        f.write("## Single-Agent Evaluation Results\n\n")
        f.write("| Framework | Benchmark | Mean Score | Std Dev | Min | Max | N |\n")
        f.write("|-----------|-----------|------------|---------|-----|-----|---|\n")

        for key, data in sorted(single_agent_analysis.items()):
            f.write(f"| {data['framework']} | {data['benchmark']} | "
                   f"{data.get('mean', 0):.3f} | {data.get('stdev', 0):.3f} | "
                   f"{data.get('min', 0):.3f} | {data.get('max', 0):.3f} | "
                   f"{data.get('count', 0)} |\n")

        # Framework Comparison
        f.write("\n## Framework Comparison\n\n")
        f.write("| Benchmark | AutoGen | LangGraph | Difference | Winner | % Improvement |\n")
        f.write("|-----------|---------|-----------|------------|--------|---------------|\n")

        for benchmark, comp_data in sorted(comparison.items()):
            f.write(f"| {benchmark} | "
                   f"{comp_data['autogen']['mean']:.3f} | "
                   f"{comp_data['langgraph']['mean']:.3f} | "
                   f"{comp_data['difference']:.3f} | "
                   f"{comp_data['winner']} | "
                   f"{comp_data['percent_improvement']:.1f}% |\n")

        # Multi-Agent Results (if available)
        if multi_agent_analysis:
            f.write("\n## Multi-Agent Evaluation Results\n\n")
            f.write("| Framework | Benchmark | Metric | Mean | Std Dev |\n")
            f.write("|-----------|-----------|--------|------|----------|\n")

            for key, data in sorted(multi_agent_analysis.items()):
                f.write(f"| {data['framework']} | {data['benchmark']} | "
                       f"{data['metric']} | {data.get('mean', 0):.3f} | "
                       f"{data.get('stdev', 0):.3f} |\n")

        # Summary
        f.write("\n## Summary\n\n")

        if comparison:
            overall_winner = statistics.mode([c["winner"] for c in comparison.values()])
            avg_improvement = statistics.mean([abs(c["percent_improvement"]) for c in comparison.values()])

            f.write(f"- **Overall Winner**: {overall_winner}\n")
            f.write(f"- **Average Performance Difference**: {avg_improvement:.1f}%\n")
            f.write(f"- **Benchmarks Evaluated**: {len(comparison)}\n")

    print(f"✅ Markdown report saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Visualize and analyze experimental results")
    parser.add_argument(
        "--autogen-dir",
        type=str,
        default="autogen/results/single_agent",
        help="AutoGen results directory"
    )
    parser.add_argument(
        "--langgraph-dir",
        type=str,
        default="langgraph/results/single_agent",
        help="LangGraph results directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/analysis_report.md",
        help="Output markdown file"
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("RESULTS ANALYSIS AND VISUALIZATION")
    print("="*70)

    # Load AutoGen results
    print("\nLoading AutoGen results...")
    autogen_dir = Path(args.autogen_dir)
    autogen_results = load_results(autogen_dir, "autogen_*_results_*.json")
    print(f"  Found {len(autogen_results)} AutoGen result files")

    # Load LangGraph results
    print("Loading LangGraph results...")
    langgraph_dir = Path(args.langgraph_dir)
    langgraph_results = load_results(langgraph_dir, "langgraph_*_results_*.json")
    print(f"  Found {len(langgraph_results)} LangGraph result files")

    # Analyze single-agent results
    print("\nAnalyzing single-agent results...")
    autogen_single = [r for r in autogen_results if r.get("agent_type") != "multi_agent"]
    langgraph_single = [r for r in langgraph_results if r.get("agent_type") != "multi_agent"]

    autogen_analysis = analyze_single_agent_results(autogen_single)
    langgraph_analysis = analyze_single_agent_results(langgraph_single)

    # Analyze multi-agent results
    print("Analyzing multi-agent results...")
    autogen_multi = [r for r in autogen_results if r.get("agent_type") == "multi_agent"]
    langgraph_multi = [r for r in langgraph_results if r.get("agent_type") == "multi_agent"]

    autogen_multi_analysis = analyze_multi_agent_results(autogen_multi)
    langgraph_multi_analysis = analyze_multi_agent_results(langgraph_multi)

    # Compare frameworks
    print("Comparing frameworks...")
    comparison = compare_frameworks(autogen_analysis, langgraph_analysis)

    # Generate report
    print("\nGenerating report...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generate_markdown_report(
        {**autogen_analysis, **langgraph_analysis},
        {**autogen_multi_analysis, **langgraph_multi_analysis},
        comparison,
        output_path
    )

    # Print summary to console
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if comparison:
        for benchmark, comp_data in comparison.items():
            print(f"\n{benchmark}:")
            print(f"  AutoGen:   {comp_data['autogen']['mean']:.3f}")
            print(f"  LangGraph: {comp_data['langgraph']['mean']:.3f}")
            print(f"  Winner:    {comp_data['winner']}")
            print(f"  Improvement: {comp_data['percent_improvement']:.1f}%")

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
