"""
LangGraph Multi-Agent Evaluation
================================
Evaluates multi-agent collaboration in LangGraph framework using GEval metrics
for measuring coordination, communication, and task delegation.
"""

import asyncio
import os
import sys
import json
import datetime
import argparse
from pathlib import Path
from typing import List, Dict, Any, TypedDict

sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase

from openai import AsyncOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from deepeval_metrics import (
    correctness_metric,
    collaboration_consistency_metric,
    reasoning_quality_metric,
    MULTI_AGENT_METRICS
)

load_dotenv(dotenv_path=Path("../.env").expanduser(), override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ========== MULTI-AGENT STATE ==========

class AgentState(TypedDict):
    """State shared across all agents."""
    task: str
    task_type: str
    specialist_output: str
    coordinator_decision: str
    final_result: str
    agent_trace: List[str]


# ========== MULTI-AGENT NODES ==========

async def coordinator_node(state: AgentState) -> AgentState:
    """Coordinator analyzes task and routes to specialist."""
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    system_prompt = (
        "You are a coordinator. Analyze the task and decide which specialist to use:\n"
        "- Mathematician: for math problems\n"
        "- Programmer: for code-related tasks\n"
        "- Reasoner: for logic and reasoning tasks\n"
        "Respond with just the specialist name."
    )

    completion = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state["task"]}
        ],
        temperature=0.3
    )

    decision = completion.choices[0].message.content.strip()

    state["coordinator_decision"] = decision
    state["task_type"] = decision
    state["agent_trace"] = state.get("agent_trace", []) + [f"Coordinator: {decision}"]

    return state


async def mathematician_node(state: AgentState) -> AgentState:
    """Specialist for mathematics."""
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    system_prompt = "You are a mathematics expert. Solve math problems step-by-step."

    completion = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state["task"]}
        ],
        temperature=0.3
    )

    output = completion.choices[0].message.content.strip()

    state["specialist_output"] = output
    state["agent_trace"] = state.get("agent_trace", []) + [f"Mathematician: solved"]

    return state


async def programmer_node(state: AgentState) -> AgentState:
    """Specialist for programming."""
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    system_prompt = "You are a programming expert. Write clean, efficient code."

    completion = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state["task"]}
        ],
        temperature=0.3
    )

    output = completion.choices[0].message.content.strip()

    state["specialist_output"] = output
    state["agent_trace"] = state.get("agent_trace", []) + [f"Programmer: implemented"]

    return state


async def reasoner_node(state: AgentState) -> AgentState:
    """Specialist for reasoning."""
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    system_prompt = "You are a reasoning expert. Think logically through problems."

    completion = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state["task"]}
        ],
        temperature=0.3
    )

    output = completion.choices[0].message.content.strip()

    state["specialist_output"] = output
    state["agent_trace"] = state.get("agent_trace", []) + [f"Reasoner: analyzed"]

    return state


async def aggregator_node(state: AgentState) -> AgentState:
    """Aggregates specialist output into final result."""
    state["final_result"] = state.get("specialist_output", "")
    state["agent_trace"] = state.get("agent_trace", []) + ["Aggregator: finalized"]

    return state


def route_to_specialist(state: AgentState) -> str:
    """Route to appropriate specialist based on coordinator decision."""
    task_type = state.get("task_type", "Reasoner")

    if "Mathematician" in task_type:
        return "mathematician"
    elif "Programmer" in task_type:
        return "programmer"
    else:
        return "reasoner"


# ========== EVALUATION ORCHESTRATOR ==========

class MultiAgentEvaluator:
    """Evaluates multi-agent collaboration in LangGraph."""

    def __init__(
        self,
        n_problems: int = 10,
        repetitions: int = 3,
        temperature: float = 0.3,
        output_dir: str = "results/multi_agent"
    ):
        self.n_problems = n_problems
        self.repetitions = repetitions
        self.temperature = temperature
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_multi_agent_graph(self) -> StateGraph:
        """Build LangGraph multi-agent workflow."""

        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("coordinator", coordinator_node)
        workflow.add_node("mathematician", mathematician_node)
        workflow.add_node("programmer", programmer_node)
        workflow.add_node("reasoner", reasoner_node)
        workflow.add_node("aggregator", aggregator_node)

        # Add edges
        workflow.add_edge(START, "coordinator")

        # Conditional routing from coordinator to specialists
        workflow.add_conditional_edges(
            "coordinator",
            route_to_specialist,
            {
                "mathematician": "mathematician",
                "programmer": "programmer",
                "reasoner": "reasoner"
            }
        )

        # All specialists go to aggregator
        workflow.add_edge("mathematician", "aggregator")
        workflow.add_edge("programmer", "aggregator")
        workflow.add_edge("reasoner", "aggregator")

        # Aggregator to end
        workflow.add_edge("aggregator", END)

        return workflow

    async def evaluate_collaboration(
        self,
        tasks: List[Dict[str, str]],
        benchmark_name: str
    ) -> Dict[str, Any]:
        """Evaluate multi-agent system on tasks."""

        print(f"\n" + "="*70)
        print(f"MULTI-AGENT EVALUATION: {benchmark_name}")
        print("="*70)

        results_all_reps = []

        for rep in range(self.repetitions):
            print(f"\nRepetition {rep + 1}/{self.repetitions}")

            test_cases = []

            # Build graph once per repetition
            workflow = self.build_multi_agent_graph()
            memory = MemorySaver()
            graph = workflow.compile(checkpointer=memory)

            for i, task in enumerate(tasks):
                # Execute multi-agent workflow
                final_state = await graph.ainvoke(
                    {"task": task["question"], "agent_trace": []},
                    config={
                        "configurable": {
                            "thread_id": f"task-{rep}-{i}",
                            "checkpoint_ns": "multi-agent-eval"
                        }
                    }
                )

                # Create test case for evaluation
                test_case = LLMTestCase(
                    input=task["question"],
                    actual_output=final_state.get("final_result", ""),
                    expected_output=task.get("answer", ""),
                    additional_metadata={
                        "agent_trace": final_state.get("agent_trace", []),
                        "multi_agent": True
                    }
                )
                test_cases.append(test_case)

                if (i + 1) % 5 == 0:
                    print(f"  Progress: {i + 1}/{len(tasks)}")

            # Evaluate with GEval metrics (including collaboration metric)
            evaluate(
                test_cases=test_cases,
                metrics=MULTI_AGENT_METRICS,
                print_results=False
            )

            # Collect metric scores
            metric_scores = {}
            for metric in MULTI_AGENT_METRICS:
                scores = [tc.metrics_data[metric.name].score
                         for tc in test_cases if metric.name in tc.metrics_data]
                if scores:
                    metric_scores[metric.name] = {
                        "mean": sum(scores) / len(scores),
                        "min": min(scores),
                        "max": max(scores)
                    }

            result = {
                "repetition": rep + 1,
                "n_tasks": len(test_cases),
                "metric_scores": metric_scores
            }
            results_all_reps.append(result)

            print(f"✓ Repetition {rep + 1} complete")

        summary = {
            "benchmark": benchmark_name,
            "framework": "LangGraph",
            "agent_type": "multi_agent",
            "date": datetime.date.today().isoformat(),
            "config": {
                "n_problems": self.n_problems,
                "repetitions": self.repetitions,
                "temperature": self.temperature,
                "agents": ["Coordinator", "Mathematician", "Programmer", "Reasoner", "Aggregator"]
            },
            "results": {
                "all_repetitions": results_all_reps
            }
        }

        self._save_results(summary, benchmark_name)
        return summary

    def _save_results(self, summary: Dict[str, Any], benchmark_name: str):
        """Save results to JSON file."""
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        filename = f"langgraph_multiagent_{benchmark_name}_{timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"💾 Saved: {filepath}")


async def main_async():
    parser = argparse.ArgumentParser(description="LangGraph Multi-Agent Evaluation")
    parser.add_argument("--n-problems", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--output-dir", type=str, default="results/multi_agent")

    args = parser.parse_args()

    evaluator = MultiAgentEvaluator(
        n_problems=args.n_problems,
        repetitions=args.repetitions,
        temperature=args.temperature,
        output_dir=args.output_dir
    )

    # Sample tasks for testing
    sample_tasks = [
        {"question": "What is 15 * 23 + 17?", "answer": "362"},
        {"question": "Write a Python function to check if a number is prime.", "answer": "def is_prime..."},
        {"question": "If all A are B, and all B are C, are all A also C?", "answer": "Yes"},
    ] * (args.n_problems // 3)

    await evaluator.evaluate_collaboration(sample_tasks[:args.n_problems], "sample_benchmark")

    print("\n✅ Multi-Agent Evaluation Complete")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
