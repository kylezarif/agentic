"""
DeepEval Custom Model Wrapper for AutoGen Framework
===================================================
This wrapper allows DeepEval benchmarks to evaluate AutoGen agents.
"""

import asyncio
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from dotenv import load_dotenv
from deepeval.models import DeepEvalBaseLLM

from autogen_core import (
    AgentId,
    MessageContext,
    RoutedAgent,
    message_handler,
    SingleThreadedAgentRuntime,
)
from openai import AsyncOpenAI


# Load environment
load_dotenv(dotenv_path=Path("../.env").expanduser(), override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


@dataclass
class TaskMessage:
    """Message containing task input and tools."""
    content: str
    tools_available: List[str] = None


@dataclass
class ResultMessage:
    """Message containing agent result."""
    output: str
    reasoning_trace: str = ""
    tools_used: List[Dict[str, Any]] = None


class AutoGenSingleAgent(RoutedAgent):
    """AutoGen agent for single-agent evaluation."""

    def __init__(self, name: str = "AutoGenAgent", temperature: float = 0.3) -> None:
        super().__init__(name)
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.temperature = temperature
        self.result = None

    @message_handler
    async def on_task_message(self, message: TaskMessage, ctx: MessageContext) -> None:
        """Handle incoming task and generate response."""

        # System prompt based on task type
        system_prompt = (
            "You are an expert reasoning assistant. "
            "Solve the given problem step-by-step and provide a clear final answer. "
            "Show your reasoning process."
        )

        completion = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message.content},
            ],
            temperature=self.temperature,
        )

        output = completion.choices[0].message.content.strip()

        self.result = ResultMessage(
            output=output,
            reasoning_trace=output,  # For single agent, output contains reasoning
            tools_used=[]
        )


class AutoGenModelWrapper(DeepEvalBaseLLM):
    """
    DeepEval-compatible wrapper for AutoGen agents.
    Implements the DeepEvalBaseLLM interface required by DeepEval benchmarks.
    """

    def __init__(
        self,
        model_name: str = "autogen-gpt-4o-mini",
        temperature: float = 0.3,
        agent_type: str = "single"
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.agent_type = agent_type

    def load_model(self):
        """Load model (required by DeepEvalBaseLLM)."""
        return self

    def generate(self, prompt: str) -> str:
        """
        Generate response using AutoGen agent (synchronous wrapper).
        Required by DeepEvalBaseLLM interface.
        """
        return asyncio.run(self.a_generate(prompt))

    async def a_generate(
        self,
        prompt: str,
        schema: Optional[Any] = None
    ) -> str:
        """
        Asynchronous generation using AutoGen agent.
        Required by DeepEvalBaseLLM interface.
        """
        # Create AutoGen runtime
        runtime = SingleThreadedAgentRuntime()

        # Register agent
        await AutoGenSingleAgent.register(
            runtime,
            "AutoGenAgent",
            lambda: AutoGenSingleAgent("AutoGenAgent", self.temperature)
        )

        runtime.start()

        # Send task to agent
        agent_id = AgentId("AutoGenAgent", "default")
        await runtime.send_message(
            TaskMessage(content=prompt, tools_available=[]),
            agent_id
        )

        await runtime.stop_when_idle()

        # Retrieve result
        maybe_agent = await runtime._get_agent(AgentId("AutoGenAgent", "default"))
        agent_instance = maybe_agent.instance if hasattr(maybe_agent, "instance") else maybe_agent

        if agent_instance.result:
            return agent_instance.result.output
        else:
            return ""

    def get_model_name(self) -> str:
        """Return model name (required by DeepEvalBaseLLM)."""
        return self.model_name


# ========== Factory Function ==========

def create_autogen_model(
    temperature: float = 0.3,
    agent_type: str = "single"
) -> AutoGenModelWrapper:
    """
    Create an AutoGen model wrapper for DeepEval evaluation.

    Args:
        temperature: Model temperature for consistency
        agent_type: "single" or "multi" for agent configuration

    Returns:
        AutoGenModelWrapper instance ready for DeepEval benchmarks
    """
    return AutoGenModelWrapper(
        model_name=f"autogen-gpt-4o-mini-{agent_type}",
        temperature=temperature,
        agent_type=agent_type
    )


if __name__ == "__main__":
    # Test the wrapper
    print("Testing AutoGen Model Wrapper...")

    model = create_autogen_model(temperature=0.3)
    test_prompt = "What is 25 * 4 + 10?"

    print(f"\nPrompt: {test_prompt}")
    response = model.generate(test_prompt)
    print(f"Response: {response}")
    print("\n✓ AutoGen wrapper test complete")
