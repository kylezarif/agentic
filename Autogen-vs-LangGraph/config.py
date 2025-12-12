"""
Shared Experimental Configuration
==================================
This file contains standardized settings for both AutoGen and LangGraph frameworks
to ensure consistent experimental conditions across all evaluations.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv


# Load environment variables
load_dotenv(dotenv_path=Path(".env").expanduser(), override=True, verbose=True)


@dataclass
class ExperimentalConfig:
    """Standardized experimental configuration for all benchmarks."""

    # === API Keys ===
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    langfuse_public_key: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    langfuse_secret_key: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    langfuse_host: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    # === Model Configuration ===
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.3  # Standardized to minimize stochastic variation
    max_tokens: Optional[int] = None

    # === Experimental Design Parameters ===
    # Single-agent trials
    single_agent_n_problems: int = 100  # 50-100 as per design
    single_agent_repetitions: int = 10  # 5-10 as per design

    # Multi-agent trials
    multi_agent_n_problems: int = 50  # 25-50 as per design
    multi_agent_repetitions: int = 10  # 5-10 as per design

    # === Benchmark-Specific Settings ===
    # HumanEval
    humaneval_timeout_seconds: int = 10
    humaneval_k: int = 1  # pass@k metric

    # ARC
    arc_challenge_only: bool = False  # Use both easy and challenge sets

    # GSM8K
    gsm8k_use_calculator: bool = True  # Allow calculator tool

    # MATH
    math_use_sympy: bool = True  # Allow symbolic computation
    math_difficulty_levels: list = None  # None = all levels

    # === Logging Configuration ===
    enable_langfuse: bool = True
    log_intermediate_steps: bool = True
    save_reasoning_traces: bool = True

    # === Output Configuration ===
    results_base_dir: str = "results"
    save_json_results: bool = True
    save_csv_summary: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.openai_api_key:
            raise EnvironmentError("OPENAI_API_KEY not found in environment!")

        if self.enable_langfuse:
            if not self.langfuse_public_key or not self.langfuse_secret_key:
                raise EnvironmentError(
                    "Langfuse keys not found! Set LANGFUSE_PUBLIC_KEY and "
                    "LANGFUSE_SECRET_KEY or disable Langfuse logging."
                )

        # Set default difficulty levels for MATH if not specified
        if self.math_difficulty_levels is None:
            self.math_difficulty_levels = list(range(1, 6))  # Levels 1-5

    def get_results_dir(self, framework: str, agent_type: str, benchmark: str) -> Path:
        """Get the results directory path for a specific experiment."""
        results_dir = Path(self.results_base_dir) / framework / agent_type / benchmark
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir

    def to_dict(self) -> dict:
        """Convert config to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "single_agent_n_problems": self.single_agent_n_problems,
            "single_agent_repetitions": self.single_agent_repetitions,
            "multi_agent_n_problems": self.multi_agent_n_problems,
            "multi_agent_repetitions": self.multi_agent_repetitions,
        }


# === Global Configuration Instance ===
CONFIG = ExperimentalConfig()


# === Benchmark-Specific Prompt Templates ===
PROMPTS = {
    "gsm8k": {
        "system": (
            "You are a reasoning assistant that solves grade-school math problems. "
            "Show your reasoning step-by-step, then output the final numeric answer "
            "in the format: 'Answer: <number>'."
        ),
        "user": "Solve this problem:\n{question}"
    },

    "humaneval": {
        "system": (
            "You are a Python programming assistant. "
            "Complete the given function according to its docstring. "
            "Output only the complete function implementation."
        ),
        "user": "Complete this function:\n{prompt}"
    },

    "arc": {
        "system": (
            "You are a reasoning assistant that answers multiple-choice science questions. "
            "Think step-by-step about the question and select the best answer. "
            "Output your answer in the format: 'Answer: <letter>'"
        ),
        "user": "Question: {question}\n\nChoices:\n{choices}\n\nThink carefully and select the best answer."
    },

    "math": {
        "system": (
            "You are a mathematics expert that solves complex math problems. "
            "Show detailed reasoning and all steps in your solution. "
            "Output the final answer in the format: 'Answer: <answer>' "
            "where <answer> is in LaTeX format if necessary."
        ),
        "user": "Solve this problem:\n{problem}"
    }
}


# === Tool Definitions ===
TOOL_DEFINITIONS = {
    "calculator": {
        "name": "calculator",
        "description": "Perform basic arithmetic operations. Input should be a mathematical expression as a string.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A mathematical expression to evaluate (e.g., '2 + 2', '10 * 5 + 3')"
                }
            },
            "required": ["expression"]
        }
    },

    "python_executor": {
        "name": "python_executor",
        "description": "Execute Python code and return the result. Useful for complex calculations or logic.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Should include print statements for output."
                }
            },
            "required": ["code"]
        }
    }
}


# === Validation Functions ===
def validate_config():
    """Validate that all required configuration is present."""
    try:
        CONFIG.__post_init__()
        print("✓ Configuration validated successfully")
        return True
    except EnvironmentError as e:
        print(f"✗ Configuration validation failed: {e}")
        return False


if __name__ == "__main__":
    print("Experimental Configuration")
    print("=" * 50)
    print(f"Model: {CONFIG.model_name}")
    print(f"Temperature: {CONFIG.temperature}")
    print(f"Single-agent trials: {CONFIG.single_agent_n_problems} problems × {CONFIG.single_agent_repetitions} reps")
    print(f"Multi-agent trials: {CONFIG.multi_agent_n_problems} problems × {CONFIG.multi_agent_repetitions} reps")
    print(f"Langfuse logging: {'Enabled' if CONFIG.enable_langfuse else 'Disabled'}")
    print("=" * 50)
    validate_config()
