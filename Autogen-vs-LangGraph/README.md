# Agentic AI Frameworks Comparative Study

## Study Objectives and Research Questions

The study will evaluate and compare existing agentic AI frameworks powered by large language model (LLM)-based systems where integrated reasoning, planning, memory integration, and tool-use tasks across diverse architectural designs are present. To be more specific in scope of the work centers on answering the following questions:

1. **Task Success Rate**: What is the end-to-end task success rate on single agents for multi-step tasks?
2. **Intent Maintenance**: How accurately does each framework detect, pursue, and maintain intent throughout the task execution?
3. **Tool Selection and Execution**: When using external tools or APIs, does the agent select the correct tool and execute calls properly?
4. **Multi-Agent Consistency**: How consistently do agents collaborate to solve tasks across repeated trials for each multi-agent framework?

## Experimental Design

This study adopts a controlled experimental design that considers LangGraph and AutoGen agentic AI frameworks. Each framework will be tested using publicly available datasets designed for structured reasoning and tool-based task execution.

### Datasets and Benchmarks

The following datasets provide established ground-truth answers for reliable performance comparison:

1. **HumanEval** - Programming-based verification (Chen et al., 2021)
2. **ARC** - Multi-step reasoning (Clark et al., 2018)
3. **GSM8K** - Interpretable reasoning sequences (Cobbe et al., 2021)
4. **MATH** - Complex problem-solving (Hendrycks et al., 2021)

### Framework Selection

**LangGraph and AutoGen** were selected because:
- Both represent leading frameworks for agentic AI
- Allow for controlled comparison of planning and intent maintenance
- Support both single-agent and multi-agent architectures
- Provide tool-use capabilities
- Have active communities and documentation

### Evaluation Conditions

#### Single-Agent Trials
Single-agent trials assess how well an individual agent can:
- Plan for multiple steps
- Stay focused on correctly completing the intended goal
- **Sample size**: 50-100 task items per framework
- **Repetitions**: 5-10 times per task to measure consistency

#### Multi-Agent Trials
Multi-agent trials evaluate:
- Whether coordination mechanisms support consistent collaboration
- Communication and task delegation between agents
- **Sample size**: 25-50 task items per framework
- **Repetitions**: 5-10 times per task to reduce random variation

### Experimental Controls

All experiments will be conducted with standardized conditions:

| Parameter | Value/Description |
|-----------|------------------|
| **Environment** | Local computational system with containerized environments |
| **Model** | gpt-4o-mini (consistent across both frameworks) |
| **Temperature** | 0.3 (standardized to minimize stochastic variation) |
| **Sampling** | Consistent configurations across all trials |
| **Tool Permissions** | Constant across all trials |
| **Logging** | Automated using Langfuse observatory framework |

### Data Collection and Reproducibility

- **Reasoning traces**, **intermediate steps**, and **final outputs** are automatically logged using Langfuse
- All experiments, including code and results, will be archived in a public GitHub repository
- Follows open evaluation practices (DeepEval Team, 2024)

## Project Structure

```
experimental_design/
├── README.md                    # This file - project overview
├── config.py                    # Shared experimental configuration
├── .env                         # Environment variables (not in git)
├── autogen/                     # AutoGen framework implementation
│   ├── README.md                # AutoGen-specific documentation
│   ├── evaluate_autogen_with_gsm8k.py
│   ├── evaluate_autogen_with_humaneval.py
│   ├── evaluate_autogen_with_arc.py
│   ├── evaluate_autogen_with_math.py
│   ├── multi_agent_*.py         # Multi-agent evaluation scripts
│   ├── run_all_experiments.py   # Orchestration script
│   └── results/                 # JSON outputs
│       ├── single_agent/
│       └── multi_agent/
└── langgraph/                   # LangGraph framework implementation
    ├── README.md                # LangGraph-specific documentation
    ├── evaluate_langgraph_with_gsm8k.py
    ├── evaluate_langgraph_with_humaneval.py
    ├── evaluate_langgraph_with_arc.py
    ├── evaluate_langgraph_with_math.py
    ├── multi_agent_*.py         # Multi-agent evaluation scripts
    ├── run_all_experiments.py   # Orchestration script
    └── results/                 # JSON outputs
        ├── single_agent/
        └── multi_agent/
```

## Installation and Setup

### Prerequisites

- Python 3.10+
- pip or uv package manager
- OpenAI API key
- Langfuse account (optional, for logging)

### Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd experimental_design
```

2. Create environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

Required environment variables:
```
OPENAI_API_KEY=your_openai_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key  # Optional
LANGFUSE_SECRET_KEY=your_langfuse_secret_key  # Optional
LANGFUSE_HOST=https://cloud.langfuse.com      # Optional
```

3. Install dependencies for each framework:

**AutoGen:**
```bash
cd autogen
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install autogen-core autogen-ext datasets openai langfuse python-dotenv
```

**LangGraph:**
```bash
cd langgraph
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install langgraph langchain langchain-openai datasets openai langfuse python-dotenv
```

## Running Experiments

### Quick Start

Run a single benchmark on a single framework:

```bash
# AutoGen + GSM8K
cd autogen
python evaluate_autogen_with_gsm8k.py

# LangGraph + GSM8K
cd langgraph
python evaluate_langgraph_with_gsm8k.py
```

### Full Experimental Suite

Run all benchmarks with full experimental parameters:

```bash
# AutoGen - all experiments
cd autogen
python run_all_experiments.py --n-problems 100 --repetitions 10

# LangGraph - all experiments
cd langgraph
python run_all_experiments.py --n-problems 100 --repetitions 10
```

### Custom Configuration

You can modify the experimental parameters in `config.py`:

```python
from config import CONFIG

# Adjust parameters
CONFIG.single_agent_n_problems = 50  # Reduce for faster testing
CONFIG.single_agent_repetitions = 5
CONFIG.temperature = 0.0  # More deterministic
```

## Results Format

Results are saved as JSON files with standardized structure:

```json
{
  "metadata": {
    "date": "2024-11-04",
    "benchmark": "GSM8K",
    "framework": "AutoGen",
    "agent_type": "single_agent",
    "model": "gpt-4o-mini",
    "temperature": 0.3,
    "repetition": 1
  },
  "summary": {
    "n_problems": 100,
    "accuracy": 0.85,
    "correct": 85,
    "total": 100,
    "avg_reasoning_length": 145.2
  },
  "results": [
    {
      "problem_id": 0,
      "question": "...",
      "gold_answer": "...",
      "predicted_answer": "...",
      "is_correct": true,
      "reasoning_trace": "...",
      "tools_used": [],
      "execution_time_ms": 1234
    }
  ]
}
```

## Project Timeline

| Period | Activities |
|--------|-----------|
| **Weeks 1-2** | Environment setup and pilot testing of task execution |
| **Weeks 3-4** | Data collection (single-agent and multi-agent trials) |
| **Weeks 5-6** | Organizing results, statistical summaries, and written report |

**Total Duration**: 4-6 weeks

## Analysis Plan

### Metrics to Measure

1. **Task Success Rate**
   - Accuracy on each benchmark
   - Comparison between frameworks
   - Single-agent vs. multi-agent performance

2. **Intent Maintenance**
   - Analyze reasoning traces
   - Track goal adherence across steps
   - Measure deviation from intended solution path

3. **Tool Usage**
   - Tool selection accuracy
   - Tool execution success rate
   - Comparison of tool-use patterns

4. **Consistency**
   - Variance across repetitions
   - Stability of answers
   - Framework reliability scores

### Statistical Methods

- Descriptive statistics (mean, median, std dev)
- Comparative analysis (t-tests, effect sizes)
- Consistency measures (coefficient of variation)
- Visualization (accuracy plots, consistency heatmaps)

## References

- Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. D. O., Kaplan, J., ... & Zaremba, W. (2021). Evaluating Large Language Models Trained on Code. *arXiv preprint arXiv:2107.03374*.

- Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., & Tafjord, O. (2018). Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge. *arXiv preprint arXiv:1803.05457*.

- Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., ... & Schulman, J. (2021). Training Verifiers to Solve Math Word Problems. *arXiv preprint arXiv:2110.14168*.

- Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., ... & Steinhardt, J. (2021). Measuring Mathematical Problem Solving With the MATH Dataset. *arXiv preprint arXiv:2103.03874*.

- Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., & Zhou, D. (2022). Self-Consistency Improves Chain of Thought Reasoning in Language Models. *arXiv preprint arXiv:2203.11171*.

- DeepEval Team (2024). Open Evaluation Practices for LLM Systems. Retrieved from https://docs.deepeval.com

## License

This experimental design is intended for academic research and follows open evaluation practices with public GitHub archival for reproducibility and transparency.

## Contributing

This is a research project. For questions or collaboration inquiries, please open an issue on the GitHub repository.

## Citation

If you use this experimental design or codebase, please cite:

```bibtex
@misc{agentic_frameworks_comparison_2024,
  title={Comparative Evaluation of Agentic AI Frameworks: LangGraph vs AutoGen},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[username]/experimental_design}
}
```
