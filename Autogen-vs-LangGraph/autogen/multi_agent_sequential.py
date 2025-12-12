import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

from autogen_core import (
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    message_handler,
    type_subscription,
)
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient


load_dotenv(dotenv_path=Path("../.env").expanduser(), override=True, verbose=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not found in .env file!")

weather_topic_type = "WeatherRetrieverAgent"
planner_topic_type = "TravelPlannerAgent"
format_topic_type = "FormatProofAgent"
user_topic_type = "UserAgent"


@dataclass
class Message:
    content: str


@type_subscription(topic_type=weather_topic_type)
class WeatherRetrieverAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient):
        super().__init__("Weather Retriever Agent")
        self._model_client = model_client
        self._system_msg = SystemMessage(
            content="You are a weather assistant. Summarize weather conditions clearly and briefly."
        )

    async def _get_weather(self, city: str) -> str:
        url = f"https://wttr.in/{city}?format=j1"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return f"Could not retrieve weather for {city}."
                data = await resp.json()
                temp_c = data["current_condition"][0]["temp_C"]
                desc = data["current_condition"][0]["weatherDesc"][0]["value"]
                return f"The weather in {city} is {desc} with temperature around {temp_c}°C."

    @message_handler
    async def handle_request(self, message: Message, ctx: MessageContext) -> None:
        city = message.content.strip()
        weather_info = await self._get_weather(city)
        print(f"\n{self.id.type}:\n{weather_info}")
        await self.publish_message(
            Message(weather_info),
            topic_id=TopicId(planner_topic_type, source=self.id.key),
        )


@type_subscription(topic_type=planner_topic_type)
class TravelPlannerAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient):
        super().__init__("Travel Planner Agent")
        self._model_client = model_client
        schema_str = json.dumps({
            "city": "string (city name)",
            "date": "string (day or date of travel)",
            "itinerary": {
                "morning": "string (morning plan)",
                "afternoon": "string (afternoon plan)",
                "evening": "string (evening plan)"
            },
            "dining_suggestions": ["string (restaurant 1)", "string (restaurant 2)"],
            "notes": "string (additional remarks)"
        }, indent=2)
        self._system_msg = SystemMessage(
            content=(
                "You are a travel planner. Given the weather, plan a one-day trip as valid JSON.\n"
                "Output must strictly follow this schema:\n"
                f"{schema_str}\n"
                "Rules:\n"
                "- Output valid JSON only, no text or markdown.\n"
                "- Keep it concise and realistic."
            )
        )

    @message_handler
    async def handle_weather_info(self, message: Message, ctx: MessageContext) -> None:
        prompt = f"Plan a one-day trip based on this weather info:\n{message.content}"
        llm_result = await self._model_client.create(
            messages=[self._system_msg, UserMessage(content=prompt, source=self.id.key)],
            cancellation_token=ctx.cancellation_token,
        )
        response = llm_result.content
        assert isinstance(response, str)
        try:
            parsed = json.loads(response)
            formatted = json.dumps(parsed, indent=2)
        except Exception:
            formatted = json.dumps({"error": "invalid JSON", "raw_output": response}, indent=2)
        print(f"\n{self.id.type}:\n{formatted}")
        await self.publish_message(
            Message(formatted),
            topic_id=TopicId(format_topic_type, source=self.id.key),
        )


@type_subscription(topic_type=format_topic_type)
class FormatProofAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient):
        super().__init__("Format & Proof Agent")
        self._model_client = model_client
        schema_str = json.dumps({
            "validated_itinerary": "object (same structure as planner JSON, refined)",
            "summary": "string (short readable summary)",
            "approval_request": "string (ask user to APPROVE or suggest edits)"
        }, indent=2)
        self._system_msg = SystemMessage(
            content=(
                "You are an editor. Validate and polish the given itinerary JSON.\n"
                f"Respond only with JSON matching this schema:\n{schema_str}\n"
                "Rules:\n"
                "- Never output plain text.\n"
                "- Keep formatting clean and compact."
            )
        )

    @message_handler
    async def handle_itinerary(self, message: Message, ctx: MessageContext) -> None:
        prompt = f"Polish and validate this itinerary JSON:\n{message.content}"
        llm_result = await self._model_client.create(
            messages=[self._system_msg, UserMessage(content=prompt, source=self.id.key)],
            cancellation_token=ctx.cancellation_token,
        )
        response = llm_result.content
        assert isinstance(response, str)
        try:
            parsed = json.loads(response)
            formatted = json.dumps(parsed, indent=2)
        except Exception:
            formatted = json.dumps({"error": "invalid JSON", "raw_output": response}, indent=2)
        print(f"\n{self.id.type}:\n{formatted}")
        await self.publish_message(
            Message(formatted),
            topic_id=TopicId(user_topic_type, source=self.id.key),
        )


@type_subscription(topic_type=user_topic_type)
class UserAgent(RoutedAgent):
    def __init__(self):
        super().__init__("User Agent")

    @message_handler
    async def handle_final_itinerary(self, message: Message, ctx: MessageContext) -> None:
        print(f"\n{self.id.type} received final trip plan:\n{message.content}\n")


async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    runtime = SingleThreadedAgentRuntime()

    await WeatherRetrieverAgent.register(
        runtime, type=weather_topic_type, factory=lambda: WeatherRetrieverAgent(model_client)
    )
    await TravelPlannerAgent.register(
        runtime, type=planner_topic_type, factory=lambda: TravelPlannerAgent(model_client)
    )
    await FormatProofAgent.register(
        runtime, type=format_topic_type, factory=lambda: FormatProofAgent(model_client)
    )
    await UserAgent.register(runtime, type=user_topic_type, factory=lambda: UserAgent())

    runtime.start()
    await runtime.publish_message(
        Message(content="Austin, Texas"),
        topic_id=TopicId(weather_topic_type, source="default"),
    )
    await runtime.stop_when_idle()
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
