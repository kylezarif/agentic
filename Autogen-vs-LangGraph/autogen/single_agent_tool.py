import asyncio
import json
import os
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List

import aiohttp
from dotenv import load_dotenv

from autogen_core import (
    AgentId,
    CancellationToken,
    FunctionCall,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    message_handler,
)
from autogen_core.models import (
    AssistantMessage,
    FunctionExecutionResult,
    FunctionExecutionResultMessage,
    SystemMessage,
    UserMessage,
    LLMMessage,
)
from autogen_core.tools import FunctionTool, Tool
from autogen_ext.models.openai import OpenAIChatCompletionClient


load_dotenv(dotenv_path=Path("../.env").expanduser(), override=True, verbose=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not found in .env file!")


async def get_weather(city: str) -> str:
    """Fetch current weather from wttr.in API."""
    url = f"https://wttr.in/{city}?format=j1"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Failed to fetch weather for {city}: {resp.status}")
            data = await resp.json()
            temp_c = data["current_condition"][0]["temp_C"]
            desc = data["current_condition"][0]["weatherDesc"][0]["value"]
            feels = data["current_condition"][0].get("FeelsLikeC", temp_c)
            return json.dumps(
                {
                    "city": city,
                    "date": datetime.date.today().isoformat(),
                    "description": desc,
                    "temperature_C": temp_c,
                    "feels_like_C": feels,
                },
                indent=2,
            )


weather_tool = FunctionTool(
    get_weather,
    description="Get the current weather in a specified city. Returns structured JSON.",
)


@dataclass
class Message:
    content: str


class ToolUseAgent(RoutedAgent):
    def __init__(self, model_client, tool_schema: List[Tool]) -> None:
        super().__init__("ToolUseAgent")
        schema_str = json.dumps(
            {
                "city": "string (the name of the city)",
                "date": "string (ISO date)",
                "description": "string (short weather summary)",
                "temperature_C": "number (current temperature)",
                "feels_like_C": "number (feels-like temperature)",
                "notes": "string (brief summary or advice based on weather)",
            },
            indent=2,
        )
        self._system_messages: List[LLMMessage] = [
            SystemMessage(
                content=(
                    "You are a helpful AI assistant that can use tools.\n"
                    "When asked about weather, call the get_weather tool.\n"
                    "Always produce a JSON object strictly following this schema:\n"
                    f"{schema_str}\n"
                    "Rules:\n"
                    "- Output valid JSON only (no markdown, no natural language).\n"
                    "- Use today's date automatically.\n"
                    "- Add a short 'notes' summary in plain text."
                )
            )
        ]
        self._model_client = model_client
        self._tools = tool_schema

    @message_handler
    async def handle_user_message(self, message: Message, ctx: MessageContext) -> Message:
        session: List[LLMMessage] = self._system_messages + [
            UserMessage(content=message.content, source="user")
        ]

        create_result = await self._model_client.create(
            messages=session,
            tools=self._tools,
            cancellation_token=ctx.cancellation_token,
        )

        if isinstance(create_result.content, str):
            return Message(content=create_result.content)

        assert isinstance(create_result.content, list) and all(
            isinstance(call, FunctionCall) for call in create_result.content
        )
        session.append(AssistantMessage(content=create_result.content, source="assistant"))

        results = await asyncio.gather(
            *[self._execute_tool_call(call, ctx.cancellation_token) for call in create_result.content]
        )

        session.append(FunctionExecutionResultMessage(content=results))

        create_result = await self._model_client.create(
            messages=session,
            cancellation_token=ctx.cancellation_token,
        )

        assert isinstance(create_result.content, str)
        try:
            parsed = json.loads(create_result.content)
            formatted = json.dumps(parsed, indent=2)
        except Exception:
            formatted = json.dumps({"error": "invalid JSON", "raw_output": create_result.content}, indent=2)
        return Message(content=formatted)

    async def _execute_tool_call(
        self, call: FunctionCall, cancellation_token: CancellationToken
    ) -> FunctionExecutionResult:
        tool = next((t for t in self._tools if t.name == call.name), None)
        assert tool is not None
        try:
            args = json.loads(call.arguments)
            result = await tool.run_json(args, cancellation_token)
            return FunctionExecutionResult(
                call_id=call.id,
                content=tool.return_value_as_string(result),
                is_error=False,
                name=tool.name,
            )
        except Exception as e:
            return FunctionExecutionResult(
                call_id=call.id, content=str(e), is_error=True, name=tool.name
            )


async def main():
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
    )

    runtime = SingleThreadedAgentRuntime()
    tools: List[Tool] = [weather_tool]

    await ToolUseAgent.register(
        runtime,
        "tool_agent",
        lambda: ToolUseAgent(model_client=model_client, tool_schema=tools),
    )

    runtime.start()
    agent_id = AgentId("tool_agent", "default")

    response = await runtime.send_message(
        Message("Get me today's weather in Austin, Texas."),
        agent_id,
    )

    print("\nAgent Response (Structured JSON):")
    print(response.content)

    await runtime.stop_when_idle()
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
