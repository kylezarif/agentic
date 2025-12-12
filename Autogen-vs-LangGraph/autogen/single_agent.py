import asyncio
import os
import json
import datetime
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv
from autogen_core import (
    AgentId,
    MessageContext,
    RoutedAgent,
    message_handler,
    SingleThreadedAgentRuntime,
)
from openai import AsyncOpenAI


load_dotenv(dotenv_path=Path("../.env").expanduser(), override=True, verbose=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not found in .env file!")


@dataclass
class TextMessage:
    content: str
    source: str


@dataclass
class LogMessage:
    content: str


class MyAssistant(RoutedAgent):
    def __init__(self, name: str = "MyAssistant") -> None:
        super().__init__(name)
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        schema_str = json.dumps(
            {
                "query": "string (user query)",
                "date": "string (ISO format, current date)",
                "summary": "string (short 1-2 sentence summary answer)",
                "key_points": ["string (important concept 1)", "string (concept 2)", "..."],
                "further_reading": ["string (topic or suggestion)"]
            },
            indent=2,
        )
        self.system_prompt = (
            "You are a structured assistant. "
            "When replying, output only valid JSON following this schema:\n"
            f"{schema_str}\n"
            "Rules:\n"
            "- Always include today's date.\n"
            "- Keep answers concise.\n"
            "- No markdown or natural language outside JSON."
        )

    @message_handler
    async def on_text_message(self, message: TextMessage, ctx: MessageContext) -> None:
        today = datetime.date.today().isoformat()
        user_query = message.content.strip()

        completion = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"User asked: {user_query}. Today's date is {today}."}
            ],
            temperature=0.3,
        )

        reply = completion.choices[0].message.content.strip()
        try:
            parsed = json.loads(reply)
            formatted = json.dumps(parsed, indent=2)
        except Exception:
            formatted = json.dumps({"error": "invalid JSON", "raw_output": reply}, indent=2)

        print(f"[{self.id.type}] Structured reply:\n{formatted}")

        logger_id = AgentId("LoggerAgent", "default")
        await self.send_message(LogMessage(f"Assistant JSON Output:\n{formatted}"), logger_id)


class LoggerAgent(RoutedAgent):
    def __init__(self, name: str = "LoggerAgent") -> None:
        super().__init__(name)

    @message_handler
    async def on_log_message(self, message: LogMessage, ctx: MessageContext) -> None:
        print(f"[{self.id.type}] LOG RECEIVED:\n{message.content}")


async def main():
    runtime = SingleThreadedAgentRuntime()

    await MyAssistant.register(runtime, "MyAssistant", lambda: MyAssistant("MyAssistant"))
    await LoggerAgent.register(runtime, "LoggerAgent", lambda: LoggerAgent("LoggerAgent"))

    runtime.start()

    assistant_id = AgentId("MyAssistant", "default")
    await runtime.send_message(
        TextMessage(content="Explain how AutoGen Core runtimes work.", source="User"),
        assistant_id,
    )

    await runtime.stop_when_idle()


if __name__ == "__main__":
    asyncio.run(main())
