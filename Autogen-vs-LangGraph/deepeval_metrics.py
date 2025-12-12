"""
DeepEval Custom Metrics using GEval
====================================
Custom evaluation metrics aligned with the study's research questions:
1. Task Success Rate (end-to-end completion)
2. Intent Maintenance (focus and goal adherence)
3. Tool Selection and Execution (correct tool use)
4. Multi-Agent Collaboration Consistency
"""

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams


# ========== METRIC 1: TASK SUCCESS RATE ==========
task_success_metric = GEval(
    name="TaskSuccessRate",
    criteria=(
        "Evaluate whether the agent successfully completed the task from start to finish. "
        "Consider correctness of the final answer, completeness of the solution, "
        "and whether all required steps were executed properly."
    ),
    evaluation_steps=[
        "Verify the actual output matches the expected output or achieves the stated goal",
        "Check if all necessary sub-tasks were completed",
        "Ensure no critical steps were skipped or incorrectly executed",
        "Heavily penalize incomplete or incorrect final answers",
        "Award full marks only for complete, correct solutions"
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT
    ],
    threshold=0.7,
    model="gpt-4o-mini"
)


# ========== METRIC 2: INTENT MAINTENANCE ==========
intent_maintenance_metric = GEval(
    name="IntentMaintenance",
    criteria=(
        "Assess how well the agent maintains focus on the original goal throughout execution. "
        "The agent should consistently pursue the intended objective without getting distracted "
        "or deviating into irrelevant reasoning paths."
    ),
    evaluation_steps=[
        "Identify the original goal or intent from the input query",
        "Trace through the reasoning steps in the actual output",
        "Check if each reasoning step contributes toward the original goal",
        "Identify any digressions, tangential reasoning, or loss of focus",
        "Penalize heavily if the agent changes objectives mid-execution",
        "Reward clear, direct paths toward the stated goal"
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT
    ],
    threshold=0.7,
    model="gpt-4o-mini"
)


# ========== METRIC 3: TOOL SELECTION AND EXECUTION ==========
tool_usage_metric = GEval(
    name="ToolUsageAccuracy",
    criteria=(
        "Evaluate whether the agent selects the correct tools for the task and executes them properly. "
        "Correct tool selection means choosing the most appropriate tool for each sub-task. "
        "Proper execution means using tools with correct parameters and interpreting results accurately."
    ),
    evaluation_steps=[
        "Identify what tools should be used based on the task requirements",
        "Check if the agent selected the appropriate tools from available options",
        "Verify that tool calls used correct parameters and syntax",
        "Assess whether tool outputs were correctly interpreted and used",
        "Penalize incorrect tool selection, misuse of tools, or ignoring tool outputs",
        "Reward efficient tool usage that directly contributes to task completion"
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.TOOLS_CALLED  # Will contain tool usage information
    ],
    threshold=0.7,
    model="gpt-4o-mini"
)


# ========== METRIC 4: REASONING QUALITY ==========
reasoning_quality_metric = GEval(
    name="ReasoningQuality",
    criteria=(
        "Assess the quality of step-by-step reasoning demonstrated in solving the problem. "
        "High-quality reasoning should be logical, coherent, and follow sound problem-solving principles."
    ),
    evaluation_steps=[
        "Evaluate if the reasoning steps follow a logical progression",
        "Check for mathematical or logical correctness in intermediate steps",
        "Assess whether the reasoning is clear and well-explained",
        "Identify any logical fallacies or incorrect assumptions",
        "Verify that conclusions follow from the reasoning provided",
        "Penalize jumps in logic or unexplained leaps to conclusions"
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT
    ],
    threshold=0.7,
    model="gpt-4o-mini"
)


# ========== METRIC 5: MULTI-AGENT COLLABORATION (for multi-agent trials) ==========
collaboration_consistency_metric = GEval(
    name="CollaborationConsistency",
    criteria=(
        "Evaluate how effectively multiple agents collaborate to solve the task. "
        "Assess coordination, communication quality, task delegation, and consistency "
        "in achieving the goal across multiple agents."
    ),
    evaluation_steps=[
        "Identify how tasks were divided among agents",
        "Assess whether task delegation was logical and efficient",
        "Check if agents communicated effectively and shared necessary information",
        "Verify that agents built upon each other's work rather than duplicating effort",
        "Evaluate if the final output represents coherent collaboration",
        "Penalize poor coordination, miscommunication, or redundant work",
        "Reward seamless integration of contributions from multiple agents"
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT
    ],
    threshold=0.7,
    model="gpt-4o-mini"
)


# ========== METRIC 6: ANSWER CORRECTNESS (Primary metric) ==========
correctness_metric = GEval(
    name="AnswerCorrectness",
    criteria=(
        "Determine whether the actual output is factually correct based on the expected output. "
        "This is the primary success metric for evaluating agent performance."
    ),
    evaluation_steps=[
        "Extract the final answer from the actual output",
        "Compare the final answer to the expected output",
        "Check whether facts in the actual output contradict the expected output",
        "Heavily penalize omission of critical details",
        "Vague language or differing presentation is acceptable if the core answer is correct",
        "Award full marks only if the answer is completely correct"
    ],
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT
    ],
    threshold=0.8,
    model="gpt-4o-mini"
)


# ========== METRIC COLLECTIONS ==========

# Single-agent evaluation metrics
SINGLE_AGENT_METRICS = [
    correctness_metric,
    task_success_metric,
    intent_maintenance_metric,
    tool_usage_metric,
    reasoning_quality_metric
]

# Multi-agent evaluation metrics (includes collaboration)
MULTI_AGENT_METRICS = [
    correctness_metric,
    task_success_metric,
    intent_maintenance_metric,
    tool_usage_metric,
    reasoning_quality_metric,
    collaboration_consistency_metric
]


# ========== BENCHMARK-SPECIFIC METRICS ==========

# GSM8K: Math reasoning with calculator tool
gsm8k_metrics = [
    correctness_metric,
    reasoning_quality_metric,
    tool_usage_metric,  # For calculator usage
]

# HumanEval: Code generation
humaneval_metrics = [
    correctness_metric,  # Does code pass tests?
    task_success_metric,  # Is function complete?
]

# ARC: Multi-step reasoning
arc_metrics = [
    correctness_metric,
    reasoning_quality_metric,
    intent_maintenance_metric,
]

# MATH: Complex problem solving with tools
math_metrics = [
    correctness_metric,
    reasoning_quality_metric,
    tool_usage_metric,  # For symbolic computation
]


def get_metrics_for_benchmark(benchmark_name: str, is_multi_agent: bool = False):
    """Get appropriate metrics for a specific benchmark."""

    base_metrics = {
        "gsm8k": gsm8k_metrics,
        "humaneval": humaneval_metrics,
        "arc": arc_metrics,
        "math": math_metrics
    }.get(benchmark_name.lower(), SINGLE_AGENT_METRICS)

    # Add collaboration metric for multi-agent scenarios
    if is_multi_agent and collaboration_consistency_metric not in base_metrics:
        base_metrics = base_metrics + [collaboration_consistency_metric]

    return base_metrics


if __name__ == "__main__":
    print("DeepEval Custom Metrics for Agentic AI Framework Evaluation")
    print("=" * 70)
    print("\nSingle-Agent Metrics:")
    for metric in SINGLE_AGENT_METRICS:
        print(f"  - {metric.name}: threshold={metric.threshold}")

    print("\nMulti-Agent Metrics:")
    for metric in MULTI_AGENT_METRICS:
        print(f"  - {metric.name}: threshold={metric.threshold}")

    print("\nBenchmark-Specific Configurations:")
    for benchmark in ["gsm8k", "humaneval", "arc", "math"]:
        metrics = get_metrics_for_benchmark(benchmark)
        print(f"  {benchmark.upper()}: {len(metrics)} metrics")
