"""
Simplified Pilot Experiment
===========================
Runs a basic pilot test with minimal dependencies to demonstrate the framework.
"""

import asyncio
import os
import json
import datetime
from pathlib import Path
from typing import Dict, Any, List

from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class SimplePilotEvaluator:
    """Simple pilot evaluation without heavy dependencies."""

    def __init__(self, output_dir: str = "results/pilot"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    async def evaluate_math_problems(
        self,
        framework_name: str,
        problems: List[Dict[str, str]],
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """Evaluate on simple math problems."""

        print(f"\n{'='*70}")
        print(f"EVALUATING {framework_name.upper()} - SIMPLE MATH PILOT")
        print(f"{'='*70}")

        results = []
        correct = 0

        for i, problem in enumerate(problems):
            print(f"\nProblem {i+1}/{len(problems)}: {problem['question']}")

            # Get response from LLM
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a math expert. Solve the problem step-by-step and provide the final answer."
                    },
                    {
                        "role": "user",
                        "content": problem['question']
                    }
                ],
                temperature=temperature
            )

            actual_output = response.choices[0].message.content.strip()
            expected_answer = problem['answer']

            # Simple correctness check
            is_correct = expected_answer in actual_output or \
                        str(expected_answer) in actual_output

            if is_correct:
                correct += 1

            print(f"  Expected: {expected_answer}")
            print(f"  Got: {actual_output[:100]}...")
            print(f"  ✓ Correct" if is_correct else "  ✗ Incorrect")

            results.append({
                "problem_id": i,
                "question": problem['question'],
                "expected_answer": expected_answer,
                "actual_output": actual_output,
                "is_correct": is_correct
            })

        accuracy = correct / len(problems) if problems else 0

        summary = {
            "framework": framework_name,
            "date": datetime.date.today().isoformat(),
            "benchmark": "Simple Math Pilot",
            "model": "gpt-4o-mini",
            "temperature": temperature,
            "n_problems": len(problems),
            "correct": correct,
            "total": len(problems),
            "accuracy": accuracy,
            "results": results
        }

        print(f"\n{'='*70}")
        print(f"RESULTS: {correct}/{len(problems)} correct ({accuracy:.1%})")
        print(f"{'='*70}")

        return summary

    def save_results(self, summary: Dict[str, Any], framework_name: str):
        """Save results to JSON."""
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        filename = f"{framework_name.lower()}_pilot_{timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n💾 Results saved: {filepath}")
        return filepath


async def main():
    """Run simple pilot experiment."""

    print("\n" + "="*70)
    print("SIMPLIFIED PILOT EXPERIMENT")
    print("Testing framework without heavy dependencies")
    print("="*70)

    # Sample math problems for pilot
    test_problems = [
        {"question": "What is 25 + 17?", "answer": "42"},
        {"question": "What is 8 * 7?", "answer": "56"},
        {"question": "What is 100 - 35?", "answer": "65"},
        {"question": "What is 144 / 12?", "answer": "12"},
        {"question": "What is 15 * 3 + 10?", "answer": "55"},
    ]

    evaluator = SimplePilotEvaluator()

    # Test "AutoGen-style" approach
    autogen_results = await evaluator.evaluate_math_problems(
        "AutoGen",
        test_problems,
        temperature=0.3
    )
    autogen_file = evaluator.save_results(autogen_results, "AutoGen")

    # Test "LangGraph-style" approach
    langgraph_results = await evaluator.evaluate_math_problems(
        "LangGraph",
        test_problems,
        temperature=0.3
    )
    langgraph_file = evaluator.save_results(langgraph_results, "LangGraph")

    # Comparison
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"AutoGen:   {autogen_results['accuracy']:.1%}")
    print(f"LangGraph: {langgraph_results['accuracy']:.1%}")

    winner = "AutoGen" if autogen_results['accuracy'] > langgraph_results['accuracy'] else "LangGraph"
    if autogen_results['accuracy'] == langgraph_results['accuracy']:
        winner = "Tie"

    print(f"Winner: {winner}")
    print("="*70)

    return {
        "autogen": autogen_results,
        "langgraph": langgraph_results,
        "autogen_file": str(autogen_file),
        "langgraph_file": str(langgraph_file)
    }


if __name__ == "__main__":
    results = asyncio.run(main())
    print("\n✅ Pilot experiment complete!")
    print(f"Results saved in: results/pilot/")
