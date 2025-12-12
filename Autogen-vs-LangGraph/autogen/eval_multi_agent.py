"""
AutoGen Multi-Agent Evaluation
==============================
Evaluates multi-agent collaboration in AutoGen framework using GEval metrics
for measuring coordination, communication, and task delegation.
"""

import asyncio
import os
import sys
import json
import datetime
import argparse
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase

from autogen_core import (
    AgentId,
    MessageContext,
    RoutedAgent,
    TopicId,
    TypeSubscription,
    message_handler,
    SingleThreadedAgentRuntime,
)
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    SystemMessage,
    UserMessage,
)
from autogen_ext.models.openai import OpenAIChatCompletionClient

from deepeval_metrics import (
    correctness_metric,
    collaboration_consistency_metric,
    reasoning_quality_metric,
    MULTI_AGENT_METRICS
)

load_dotenv(dotenv_path=Path("../.env").expanduser(), override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ========== MULTI-AGENT ARCHITECTURE ==========

@dataclass
class TaskMessage:
    """Message containing a task to solve."""
    content: str
    task_id: int

@dataclass
class ResultMessage:
    """Message containing results from an agent."""
    content: str
    agent_name: str
    task_id: int

class CoordinatorAgent(RoutedAgent):
    """Coordinates task delegation among specialist agents."""

    def __init__(self, model_client: ChatCompletionClient):
        super().__init__("Coordinator")
        self._model = model_client
        self._results = []

    @message_handler
    async def on_task_message(self, message: TaskMessage, ctx: MessageContext) -> None:
        """Receive task and delegate to appropriate specialist."""

        # Analyze task and decide which specialist to use
        system_prompt = (
            "You are a coordinator. Analyze the task and decide which specialist to use:\n"
            "- Mathematician: for math problems\n"
            "- Programmer: for code-related tasks\n"
            "- Reasoner: for logic and reasoning tasks\n"
            "Respond with just the specialist name."
        )

        completion = await self._model.create(
            [SystemMessage(content=system_prompt),
             UserMessage(content=message.content, source="user")],
            cancellation_token=ctx.cancellation_token
        )

        specialist = completion.content.strip() if isinstance(completion.content, str) else "Reasoner"

        # Forward to appropriate specialist
        await self.publish_message(
            TaskMessage(content=message.content, task_id=message.task_id),
            TopicId(type=specialist, source="default")
        )

    @message_handler
    async def on_result_message(self, message: ResultMessage, ctx: MessageContext) -> None:
        """Collect results from specialists."""
        self._results.append(message)


class SpecialistAgent(RoutedAgent):
    """Specialist agent for specific task types."""

    def __init__(self, name: str, specialty: str, model_client: ChatCompletionClient):
        super().__init__(name)
        self._specialty = specialty
        self._model = model_client

    @message_handler
    async def on_task_message(self, message: TaskMessage, ctx: MessageContext) -> None:
        """Process task according to specialty."""

        system_prompts = {
            "Mathematician": "You are a mathematics expert. Solve math problems step-by-step.",
            "Programmer": "You are a programming expert. Write clean, efficient code.",
            "Reasoner": "You are a reasoning expert. Think logically through problems."
        }

        system_prompt = system_prompts.get(self._specialty, system_prompts["Reasoner"])

        completion = await self._model.create(
            [SystemMessage(content=system_prompt),
             UserMessage(content=message.content, source="user")],
            cancellation_token=ctx.cancellation_token
        )

        result = completion.content if isinstance(completion.content, str) else ""

        # Send result back to coordinator
        await self.publish_message(
            ResultMessage(content=result, agent_name=self.id.type, task_id=message.task_id),
            TopicId(type="Coordinator", source="default")
        )


# ========== EVALUATION ORCHESTRATOR ==========

class MultiAgentEvaluator:
    """Evaluates multi-agent collaboration."""

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

            for i, task in enumerate(tasks):
                # Set up multi-agent runtime
                runtime = SingleThreadedAgentRuntime()
                model = OpenAIChatCompletionClient(
                    model="gpt-4o-mini",
                    api_key=OPENAI_API_KEY,
                    temperature=self.temperature
                )

                # Register agents
                coordinator_type = await CoordinatorAgent.register(
                    runtime,
                    "Coordinator",
                    lambda: CoordinatorAgent(model)
                )

                mathematician_type = await SpecialistAgent.register(
                    runtime,
                    "Mathematician",
                    lambda: SpecialistAgent("Mathematician", "Mathematician", model)
                )

                programmer_type = await SpecialistAgent.register(
                    runtime,
                    "Programmer",
                    lambda: SpecialistAgent("Programmer", "Programmer", model)
                )

                reasoner_type = await SpecialistAgent.register(
                    runtime,
                    "Reasoner",
                    lambda: SpecialistAgent("Reasoner", "Reasoner", model)
                )

                # Subscribe to topics
                await runtime.add_subscription(
                    TypeSubscription(topic_type="Coordinator", agent_type=coordinator_type.type)
                )
                await runtime.add_subscription(
                    TypeSubscription(topic_type="Mathematician", agent_type=mathematician_type.type)
                )
                await runtime.add_subscription(
                    TypeSubscription(topic_type="Programmer", agent_type=programmer_type.type)
                )
                await runtime.add_subscription(
                    TypeSubscription(topic_type="Reasoner", agent_type=reasoner_type.type)
                )

                runtime.start()

                # Send task to coordinator
                await runtime.publish_message(
                    TaskMessage(content=task["question"], task_id=i),
                    TopicId(type="Coordinator", source="default")
                )

                await runtime.stop_when_idle()

                # Get results
                coordinator = await runtime._get_agent(AgentId("Coordinator", "default"))
                coordinator_instance = coordinator.instance if hasattr(coordinator, "instance") else coordinator

                if coordinator_instance._results:
                    result = coordinator_instance._results[0]

                    # Create test case for evaluation
                    test_case = LLMTestCase(
                        input=task["question"],
                        actual_output=result.content,
                        expected_output=task.get("answer", ""),
                        additional_metadata={
                            "agent_name": result.agent_name,
                            "multi_agent": True
                        }
                    )
                    test_cases.append(test_case)

                if (i + 1) % 5 == 0:
                    print(f"  Progress: {i + 1}/{len(tasks)}")

                await model.close()

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
            "framework": "AutoGen",
            "agent_type": "multi_agent",
            "date": datetime.date.today().isoformat(),
            "config": {
                "n_problems": self.n_problems,
                "repetitions": self.repetitions,
                "temperature": self.temperature,
                "agents": ["Coordinator", "Mathematician", "Programmer", "Reasoner"]
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
        filename = f"autogen_multiagent_{benchmark_name}_{timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"💾 Saved: {filepath}")


async def main_async():
    parser = argparse.ArgumentParser(description="AutoGen Multi-Agent Evaluation")
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
