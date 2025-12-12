import asyncio
import json
import os
import string
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import aiohttp
from dotenv import load_dotenv

from autogen_core import (
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
    message_handler,
)
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import FunctionTool, Tool
from autogen_ext.models.openai import OpenAIChatCompletionClient


# --- ENV SETUP ---
load_dotenv(dotenv_path=Path("../.env").expanduser(), override=True, verbose=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not found in .env file!")


# --- TOPIC DEFINITIONS ---
group_chat_topic_type = "trip_chat"
weather_topic_type = "Weather"
planner_topic_type = "Planner"
editor_topic_type = "Editor"
user_topic_type = "User"


# --- MESSAGE SCHEMA ---
@dataclass
class GroupChatMessage:
    content: str
    source: str


@dataclass
class RequestToSpeak:
    pass


# --- BASE AGENT CLASS ---
class BaseGroupChatAgent(RoutedAgent):
    def __init__(self, description: str, model_client: ChatCompletionClient, role_system_message: str):
        super().__init__(description)
        self._model = model_client
        self._system = SystemMessage(content=role_system_message)
        self._history: List[LLMMessage] = []

    @message_handler
    async def on_group_message(self, message: GroupChatMessage, ctx: MessageContext) -> None:
        self._history.extend([
            UserMessage(content=f"Transferred to {message.source}", source="system"),
            UserMessage(content=message.content, source=message.source),
        ])


# --- FUNCTION TOOL: LIVE WEATHER ---
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


# --- WEATHER AGENT USING TOOL ---
class WeatherAgent(BaseGroupChatAgent):
    def __init__(self, model_client: ChatCompletionClient):
        super().__init__(
            description="Weather agent that uses FunctionTool for live weather.",
            model_client=model_client,
            role_system_message="You summarize live weather data using the get_weather tool.",
        )
        self._tools: List[Tool] = [weather_tool]

    async def _extract_city(self) -> Optional[str]:
        """Extract city name from user messages."""
        for msg in reversed(self._history):
            if isinstance(msg, UserMessage) and msg.source in ("User",):
                text = (msg.content or "").strip()
                if text:
                    lower = text.lower()
                    if " in " in lower:
                        after = text[lower.index(" in ") + 4 :]
                        for stop in [",", ".", "?", "!", ";", " for "]:
                            idx = after.find(stop)
                            if idx != -1:
                                after = after[:idx]
                        return after.strip().title()
                    if len(text.split()) <= 4:
                        return text.strip()
        return None

    async def _get_weather_tool(self, city: str, ctx: MessageContext) -> str:
        """Invoke the FunctionTool directly (no FunctionCall interface)."""
        try:
            result = await weather_tool.run_json({"city": city}, ctx.cancellation_token)
            return weather_tool.return_value_as_string(result)
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)

    @message_handler
    async def on_request_to_speak(self, message: RequestToSpeak, ctx: MessageContext) -> None:
        city = await self._extract_city() or "Austin, Texas"
        weather_json = await self._get_weather_tool(city, ctx)
        self._history.append(AssistantMessage(content=weather_json, source=self.id.type))
        await self.publish_message(
            GroupChatMessage(content=weather_json, source=self.id.type),
            TopicId(type=group_chat_topic_type, source=self.id.key),
        )


# --- PLANNER AGENT ---
class PlannerAgent(BaseGroupChatAgent):
    def __init__(self, model_client: ChatCompletionClient):
        schema_str = json.dumps({
            "city": "string (name of the city)",
            "date": "string (day or date of the trip)",
            "itinerary": {
                "morning": "string (morning activity)",
                "afternoon": "string (afternoon activity)",
                "evening": "string (evening activity)"
            },
            "dining_suggestions": ["string (restaurant 1)", "string (restaurant 2)"],
            "notes": "string (summary of the plan)"
        }, indent=2)

        super().__init__(
            description="Planner agent that makes a one-day itinerary in JSON.",
            model_client=model_client,
            role_system_message=(
                "You are a travel planner AI.\n"
                "Given the weather info and user preferences, output a one-day travel itinerary "
                "as a valid JSON object matching the schema below:\n\n"
                f"{schema_str}\n\n"
                "Rules:\n"
                "- Output only valid JSON (no markdown, no prose)\n"
                "- Fill every field with a short, realistic summary"
            ),
        )

    @message_handler
    async def on_request_to_speak(self, message: RequestToSpeak, ctx: MessageContext) -> None:
        completion = await self._model.create([self._system] + self._history, cancellation_token=ctx.cancellation_token)
        assert isinstance(completion.content, str)
        try:
            parsed = json.loads(completion.content)
            formatted = json.dumps(parsed, indent=2)
        except Exception:
            formatted = json.dumps({"error": "invalid JSON", "raw_output": completion.content}, indent=2)
        self._history.append(AssistantMessage(content=formatted, source=self.id.type))
        await self.publish_message(
            GroupChatMessage(content=formatted, source=self.id.type),
            TopicId(type=group_chat_topic_type, source=self.id.key),
        )


# --- EDITOR AGENT ---
class EditorAgent(BaseGroupChatAgent):
    def __init__(self, model_client: ChatCompletionClient):
        schema_str = json.dumps({
            "validated_itinerary": "object (same structure as planner's JSON, but cleaned up)",
            "feedback": "string (brief human-readable summary)",
            "approval_request": "string (ask user to APPROVE or suggest changes)"
        }, indent=2)

        super().__init__(
            description="Editor agent that polishes the itinerary JSON.",
            model_client=model_client,
            role_system_message=(
                "You are an editor. Given a JSON itinerary, validate and polish it for readability. "
                "Respond only with JSON matching this schema:\n\n"
                f"{schema_str}\n\n"
                "Rules:\n"
                "- Never output raw text or markdown.\n"
                "- Preserve the structure.\n"
                "- Keep JSON valid and compact."
            ),
        )

    @message_handler
    async def on_request_to_speak(self, message: RequestToSpeak, ctx: MessageContext) -> None:
        completion = await self._model.create([self._system] + self._history, cancellation_token=ctx.cancellation_token)
        assert isinstance(completion.content, str)
        try:
            parsed = json.loads(completion.content)
            formatted = json.dumps(parsed, indent=2)
        except Exception:
            formatted = json.dumps({"error": "invalid JSON", "raw_output": completion.content}, indent=2)
        self._history.append(AssistantMessage(content=formatted, source=self.id.type))
        await self.publish_message(
            GroupChatMessage(content=formatted, source=self.id.type),
            TopicId(type=group_chat_topic_type, source=self.id.key),
        )


# --- USER AGENT ---
class UserAgent(RoutedAgent):
    def __init__(self):
        super().__init__("User agent that relays console input to the group chat.")

    @message_handler
    async def on_group_message(self, message: GroupChatMessage, ctx: MessageContext) -> None:
        print(f"\n[{message.source}] {message.content}")

    @message_handler
    async def on_request_to_speak(self, message: RequestToSpeak, ctx: MessageContext) -> None:
        user_input = input("\nYou: ")
        await self.publish_message(
            GroupChatMessage(content=user_input, source=self.id.type),
            TopicId(type=group_chat_topic_type, source=self.id.key),
        )


# --- GROUP CHAT MANAGER ---
class GroupChatManager(RoutedAgent):
    def __init__(self, participants: List[str]):
        super().__init__("Group chat manager")
        self._participants = participants
        self._last_idx: Optional[int] = None
        self._ended = False

    def _next_topic(self) -> str:
        if self._last_idx is None:
            self._last_idx = 0
        else:
            self._last_idx = (self._last_idx + 1) % len(self._participants)
        return self._participants[self._last_idx]

    @message_handler
    async def on_group_message(self, message: GroupChatMessage, ctx: MessageContext) -> None:
        if self._ended:
            return
        if message.source == "User":
            txt = message.content.lower().strip(string.whitespace + string.punctuation)
            if txt == "approve":
                print("\n[Manager] Conversation ended by user approval.")
                self._ended = True
                return
        topic = self._next_topic()
        await self.publish_message(RequestToSpeak(), TopicId(type=topic, source="default"))


# --- MAIN RUNTIME ---
async def main():
    model = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    runtime = SingleThreadedAgentRuntime()

    weather_type = await WeatherAgent.register(runtime, weather_topic_type, lambda: WeatherAgent(model))
    planner_type = await PlannerAgent.register(runtime, planner_topic_type, lambda: PlannerAgent(model))
    editor_type = await EditorAgent.register(runtime, editor_topic_type, lambda: EditorAgent(model))
    user_type = await UserAgent.register(runtime, user_topic_type, lambda: UserAgent())
    manager_type = await GroupChatManager.register(
        runtime,
        "group_chat_manager",
        lambda: GroupChatManager(participants=[weather_topic_type, planner_topic_type, editor_topic_type, user_topic_type]),
    )

    for agent_type, topic in [
        (weather_type, weather_topic_type),
        (planner_type, planner_topic_type),
        (editor_type, editor_topic_type),
        (user_type, user_topic_type),
    ]:
        await runtime.add_subscription(TypeSubscription(topic_type=topic, agent_type=agent_type.type))
        await runtime.add_subscription(TypeSubscription(topic_type=group_chat_topic_type, agent_type=agent_type.type))

    await runtime.add_subscription(TypeSubscription(topic_type=group_chat_topic_type, agent_type=manager_type.type))

    runtime.start()

    initial_request = "Plan a one-day trip in Flower Mound, Texas today. I love nature and burgers."
    await runtime.publish_message(
        GroupChatMessage(content=initial_request, source="User"),
        TopicId(type=group_chat_topic_type, source="default"),
    )

    await runtime.stop_when_idle()
    await model.close()


if __name__ == "__main__":
    asyncio.run(main())
