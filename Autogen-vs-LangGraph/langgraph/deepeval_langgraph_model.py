"""
DeepEval Custom Model Wrapper for LangGraph Framework
=====================================================
This wrapper allows DeepEval benchmarks to evaluate LangGraph agents.
"""

import asyncio
import os
from pathlib import Path
from typing import Optional, Any

from dotenv import load_dotenv
from deepeval.models import DeepEvalBaseLLM

from openai import AsyncOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


# Load environment
load_dotenv(dotenv_path=Path("../.env").expanduser(), override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class LangGraphModelWrapper(DeepEvalBaseLLM):
    """
    DeepEval-compatible wrapper for LangGraph agents.
    Implements the DeepEvalBaseLLM interface required by DeepEval benchmarks.
    """

    def __init__(
        self,
        model_name: str = "langgraph-gpt-4o-mini",
        temperature: float = 0.3,
        agent_type: str = "single"
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.agent_type = agent_type
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    def load_model(self):
        """Load model (required by DeepEvalBaseLLM)."""
        return self

    def generate(self, prompt: str) -> str:
        """
        Generate response using LangGraph agent (synchronous wrapper).
        Required by DeepEvalBaseLLM interface.
        """
        return asyncio.run(self.a_generate(prompt))

    async def a_generate(
        self,
        prompt: str,
        schema: Optional[Any] = None
    ) -> str:
        """
        Asynchronous generation using LangGraph agent.
        Required by DeepEvalBaseLLM interface.
        """
        # Define LangGraph nodes
        async def solve_task(state: dict):
            """Main reasoning node."""
            user_input = state["input"]

            system_prompt = (
                "You are an expert reasoning assistant. "
                "Solve the given problem step-by-step and provide a clear final answer. "
                "Show your reasoning process."
            )

            completion = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ],
                temperature=self.temperature,
            )

            output = completion.choices[0].message.content.strip()
            return {"output": output, "reasoning": output}

        async def validate_output(state: dict):
            """Validation node (optional)."""
            return {"validated": True}

        # Build workflow
        workflow = StateGraph(dict)
        workflow.add_node("solve", solve_task)
        workflow.add_node("validate", validate_output)

        workflow.add_edge(START, "solve")
        workflow.add_edge("solve", "validate")
        workflow.add_edge("validate", END)

        # Compile with memory
        memory = MemorySaver()
        graph = workflow.compile(checkpointer=memory)

        # Execute graph
        final_state = await graph.ainvoke(
            {"input": prompt},
            config={
                "configurable": {
                    "thread_id": "deepeval-test",
                    "checkpoint_ns": "evaluation"
                }
            }
        )

        return final_state.get("output", "")

    def get_model_name(self) -> str:
        """Return model name (required by DeepEvalBaseLLM)."""
        return self.model_name


# ========== Factory Function ==========

def create_langgraph_model(
    temperature: float = 0.3,
    agent_type: str = "single"
) -> LangGraphModelWrapper:
    """
    Create a LangGraph model wrapper for DeepEval evaluation.

    Args:
        temperature: Model temperature for consistency
        agent_type: "single" or "multi" for agent configuration

    Returns:
        LangGraphModelWrapper instance ready for DeepEval benchmarks
    """
    return LangGraphModelWrapper(
        model_name=f"langgraph-gpt-4o-mini-{agent_type}",
        temperature=temperature,
        agent_type=agent_type
    )


if __name__ == "__main__":
    # Test the wrapper
    print("Testing LangGraph Model Wrapper...")

    model = create_langgraph_model(temperature=0.3)
    test_prompt = "What is 25 * 4 + 10?"

    print(f"\nPrompt: {test_prompt}")
    response = model.generate(test_prompt)
    print(f"Response: {response}")
    print("\n✓ LangGraph wrapper test complete")
