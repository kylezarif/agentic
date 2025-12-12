# import asyncio
# import os
# import json
# import datetime
# from pathlib import Path
# from dataclasses import dataclass
# from dotenv import load_dotenv

# from autogen_core import (
#     AgentId,
#     MessageContext,
#     RoutedAgent,
#     message_handler,
#     SingleThreadedAgentRuntime,
# )
# from openai import AsyncOpenAI
# from datasets import load_dataset


# # ========= ENV SETUP =========
# load_dotenv(dotenv_path=Path("../.env").expanduser(), override=True, verbose=True)
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# if not OPENAI_API_KEY:
#     raise EnvironmentError("OPENAI_API_KEY not found!")

# client = AsyncOpenAI(api_key=OPENAI_API_KEY)


# # ========= DATA CLASSES =========
# @dataclass
# class MathProblem:
#     question: str
#     gold_answer: str
#     problem_id: int = 0


# @dataclass
# class MathAnswer:
#     problem_id: int
#     predicted: str
#     gold: str


# # ========= AGENTS =========
# class MyAssistant(RoutedAgent):
#     """Agent that solves GSM8K-style math problems."""

#     def __init__(self, name="MyAssistant") -> None:
#         super().__init__(name)
#         self.client = client
#         self.system_prompt = (
#             "You are a reasoning assistant that solves grade-school math problems. "
#             "Show your reasoning step-by-step, then output the final numeric answer "
#             "in the format: 'Answer: <number>'."
#         )

#     @message_handler
#     async def on_math_problem(self, message: MathProblem, ctx: MessageContext) -> None:
#         completion = await self.client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": self.system_prompt},
#                 {"role": "user", "content": message.question},
#             ],
#             temperature=0.3,
#         )

#         reply = completion.choices[0].message.content.strip()

#         await self.send_message(
#             MathAnswer(
#                 problem_id=message.problem_id,
#                 predicted=reply,
#                 gold=message.gold_answer,
#             ),
#             AgentId("ValidatorAgent", "default"),
#         )


# class ValidatorAgent(RoutedAgent):
#     """Receives and evaluates math answers."""

#     def __init__(self, name="ValidatorAgent") -> None:
#         super().__init__(name)
#         self.results = []

#     @message_handler
#     async def on_math_answer(self, message: MathAnswer, ctx: MessageContext) -> None:
#         self.results.append(message)
#         print(f"[Validator] Problem {message.problem_id}")
#         print(f"Predicted:\n{message.predicted}\nGold:\n{message.gold}\n")


# # ========= MAIN EVALUATION FUNCTION =========
# async def evaluate_autogen_on_gsm8k(n_problems: int = 5):
#     """Evaluate the AutoGen agent on GSM8K and store JSON results."""

#     print("📚 Loading GSM8K dataset...")
#     dataset = load_dataset("gsm8k", "main")["test"]
#     problems = dataset.select(range(min(n_problems, len(dataset))))

#     runtime = SingleThreadedAgentRuntime()

#     await MyAssistant.register(runtime, "MyAssistant", lambda: MyAssistant("MyAssistant"))
#     await ValidatorAgent.register(runtime, "ValidatorAgent", lambda: ValidatorAgent("ValidatorAgent"))

#     runtime.start()
#     assistant_id = AgentId("MyAssistant", "default")

#     for i, item in enumerate(problems):
#         await runtime.send_message(
#             MathProblem(
#                 question=item["question"],
#                 gold_answer=item["answer"],
#                 problem_id=i,
#             ),
#             assistant_id,
#         )

#     await runtime.stop_when_idle()

#     # ✅ Retrieve validator instance (handle both possible return types)
#     maybe_agent = await runtime._get_agent(AgentId("ValidatorAgent", "default"))
#     validator_instance = maybe_agent.instance if hasattr(maybe_agent, "instance") else maybe_agent
#     results = validator_instance.results

#     # ✅ Build results summary
#     detailed_results = []
#     correct = 0
#     total = len(results)

#     for r in results:
#         gold = str(r.gold).strip()
#         pred = (
#             str(r.predicted).split("Answer:")[-1].strip()
#             if "Answer:" in r.predicted
#             else r.predicted.strip()
#         )
#         is_correct = gold == pred
#         if is_correct:
#             correct += 1

#         reasoning_length = len(r.predicted.split())

#         detailed_results.append(
#             {
#                 "problem_id": r.problem_id,
#                 "question": problems[r.problem_id]["question"],
#                 "gold_answer": gold,
#                 "predicted_answer": pred,
#                 "raw_model_output": r.predicted,
#                 "is_correct": is_correct,
#                 "reasoning_length": reasoning_length,
#             }
#         )

#     accuracy = correct / total if total else 0.0

#     results_summary = {
#         "date": datetime.date.today().isoformat(),
#         "benchmark": "GSM8K",
#         "framework": "AutoGen",
#         "model": "gpt-4o-mini",
#         "n_problems": n_problems,
#         "accuracy": accuracy,
#         "correct": correct,
#         "total": total,
#         "results": detailed_results,
#     }

#     # ✅ Save to structured JSON file with timestamp
#     timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
#     output_path = Path(f"autogen_gsm8k_results_{timestamp}.json")
#     with open(output_path, "w") as f:
#         json.dump(results_summary, f, indent=2)

#     print(f"\n✅ AutoGen GSM8K Accuracy: {accuracy:.2f} ({correct}/{total})")
#     print(f"Results saved to {output_path.resolve()}")


# # ========= ENTRY POINT =========
# if __name__ == "__main__":
#     asyncio.run(evaluate_autogen_on_gsm8k(n_problems=5))

import asyncio
import os
import re
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
from datasets import load_dataset


# ========= ENV SETUP =========
load_dotenv(dotenv_path=Path("../.env").expanduser(), override=True, verbose=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not found!")

client = AsyncOpenAI(api_key=OPENAI_API_KEY)


# ========= DATA CLASSES =========
@dataclass
class MathProblem:
    question: str
    gold_answer: str
    problem_id: int = 0


@dataclass
class MathAnswer:
    problem_id: int
    predicted: str
    gold: str


# ========= AGENTS =========
class MyAssistant(RoutedAgent):
    """Agent that solves GSM8K-style math problems."""

    def __init__(self, name="MyAssistant") -> None:
        super().__init__(name)
        self.client = client
        self.system_prompt = (
            "You are a reasoning assistant that solves grade-school math problems. "
            "Show your reasoning step-by-step, then output the final numeric answer "
            "in the format: 'Answer: <number>'."
        )

    @message_handler
    async def on_math_problem(self, message: MathProblem, ctx: MessageContext) -> None:
        completion = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": message.question},
            ],
            temperature=0.3,
        )

        reply = completion.choices[0].message.content.strip()

        await self.send_message(
            MathAnswer(
                problem_id=message.problem_id,
                predicted=reply,
                gold=message.gold_answer,
            ),
            AgentId("ValidatorAgent", "default"),
        )


class ValidatorAgent(RoutedAgent):
    """Receives and evaluates math answers."""

    def __init__(self, name="ValidatorAgent") -> None:
        super().__init__(name)
        self.results = []

    @message_handler
    async def on_math_answer(self, message: MathAnswer, ctx: MessageContext) -> None:
        self.results.append(message)
        print(f"[Validator] Problem {message.problem_id}")
        print(f"Predicted:\n{message.predicted}\nGold:\n{message.gold}\n")


# ========= MAIN EVALUATION FUNCTION =========
async def evaluate_autogen_on_gsm8k(n_problems: int = 5):
    """Evaluate the AutoGen agent on GSM8K and store JSON results."""

    print("📚 Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main")["test"]
    problems = dataset.select(range(min(n_problems, len(dataset))))

    runtime = SingleThreadedAgentRuntime()

    await MyAssistant.register(runtime, "MyAssistant", lambda: MyAssistant("MyAssistant"))
    await ValidatorAgent.register(runtime, "ValidatorAgent", lambda: ValidatorAgent("ValidatorAgent"))

    runtime.start()
    assistant_id = AgentId("MyAssistant", "default")

    for i, item in enumerate(problems):
        await runtime.send_message(
            MathProblem(
                question=item["question"],
                gold_answer=item["answer"],
                problem_id=i,
            ),
            assistant_id,
        )

    await runtime.stop_when_idle()

    # ✅ Retrieve validator instance (works for new & old autogen versions)
    maybe_agent = await runtime._get_agent(AgentId("ValidatorAgent", "default"))
    validator_instance = maybe_agent.instance if hasattr(maybe_agent, "instance") else maybe_agent
    results = validator_instance.results

    # ✅ Build results summary with correct numeric matching
    detailed_results = []
    correct = 0
    total = len(results)

    for r in results:
        # Extract numeric gold answer from GSM8K pattern "#### <number>"
        gold_match = re.search(r"####\s*([\d.,+-]+)", r.gold)
        gold = gold_match.group(1).replace(",", "") if gold_match else str(r.gold).strip()

        # Extract numeric predicted answer from model output "Answer: <number>"
        pred_match = re.search(r"Answer:\s*([\d.,+-]+)", r.predicted)
        pred = pred_match.group(1).replace(",", "") if pred_match else r.predicted.strip()

        is_correct = gold == pred
        if is_correct:
            correct += 1

        reasoning_length = len(r.predicted.split())

        detailed_results.append(
            {
                "problem_id": r.problem_id,
                "question": problems[r.problem_id]["question"],
                "gold_answer": gold,
                "predicted_answer": pred,
                "raw_model_output": r.predicted,
                "is_correct": is_correct,
                "reasoning_length": reasoning_length,
            }
        )

    accuracy = correct / total if total else 0.0

    results_summary = {
        "date": datetime.date.today().isoformat(),
        "benchmark": "GSM8K",
        "framework": "AutoGen",
        "model": "gpt-4o-mini",
        "n_problems": n_problems,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": detailed_results,
    }

    # ✅ Save to structured JSON file with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    output_path = Path(f"autogen_gsm8k_results_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n✅ AutoGen GSM8K Accuracy: {accuracy:.2f} ({correct}/{total})")
    print(f"Results saved to {output_path.resolve()}")


# ========= ENTRY POINT =========
if __name__ == "__main__":
    asyncio.run(evaluate_autogen_on_gsm8k(n_problems=5))


