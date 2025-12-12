import asyncio
import os
import re
import json
import datetime
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

from datasets import load_dataset
from openai import AsyncOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


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


# ========= GRAPH NODES =========
async def solve_math_problem(state: dict):
    """LangGraph node: LLM reasoning and answer generation"""
    problem: MathProblem = state["problem"]

    system_prompt = (
        "You are a reasoning assistant that solves grade-school math problems. "
        "Show your reasoning step-by-step, then output the final numeric answer "
        "in the format: 'Answer: <number>'."
    )

    completion = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem.question},
        ],
        temperature=0.3,
    )

    reply = completion.choices[0].message.content.strip()
    return {"answer": MathAnswer(problem.problem_id, reply, problem.gold_answer)}


async def validate_answer(state: dict):
    """LangGraph node: Compare predicted and gold answers"""
    answer: MathAnswer = state["answer"]
    print(f"[Validator] Problem {answer.problem_id}")
    print(f"Predicted:\n{answer.predicted}\nGold:\n{answer.gold}\n")
    return {"validated": answer}


# ========= MAIN EVALUATION FUNCTION =========
async def evaluate_langgraph_on_gsm8k(n_problems: int = 5):
    """Evaluate the LangGraph agentic pipeline on GSM8K and store JSON results."""

    print("📚 Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main")["test"]
    problems = dataset.select(range(min(n_problems, len(dataset))))

    # ✅ FIX 1: specify state_schema
    workflow = StateGraph(dict)
    workflow.add_node("solve", solve_math_problem)
    workflow.add_node("validate", validate_answer)

    workflow.add_edge(START, "solve")
    workflow.add_edge("solve", "validate")
    workflow.add_edge("validate", END)

    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    results = []

    # ✅ FIX 2: provide thread_id *and* checkpoint_ns
    for i, item in enumerate(problems):
        problem = MathProblem(
            question=item["question"], gold_answer=item["answer"], problem_id=i
        )
        final_state = await graph.ainvoke(
            {"problem": problem},
            configurable={
                "thread_id": f"problem-{i}",
                "checkpoint_ns": "gsm8k-eval"
            }
        )
        results.append(final_state["validated"])

    # ✅ Build results summary
    detailed_results = []
    correct = 0
    total = len(results)

    for r in results:
        gold_match = re.search(r"####\s*([\d.,+-]+)", r.gold)
        gold = gold_match.group(1).replace(",", "") if gold_match else str(r.gold).strip()

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
        "framework": "LangGraph",
        "model": "gpt-4o-mini",
        "n_problems": n_problems,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": detailed_results,
    }

    # ✅ Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    output_path = Path(f"langgraph_gsm8k_results_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n✅ LangGraph GSM8K Accuracy: {accuracy:.2f} ({correct}/{total})")
    print(f"Results saved to {output_path.resolve()}")


# ========= ENTRY POINT =========
if __name__ == "__main__":
    asyncio.run(evaluate_langgraph_on_gsm8k(n_problems=5))
