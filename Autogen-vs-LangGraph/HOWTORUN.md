# How to Run the Experimental Study

## Complete Guide to Running AutoGen vs LangGraph Comparative Evaluation

This guide provides step-by-step instructions for running the full experimental study comparing AutoGen and LangGraph frameworks using DeepEval benchmarks and custom GEval metrics.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Initial Setup](#initial-setup)
3. [Running a Pilot Study](#running-a-pilot-study)
4. [Running Individual Framework Evaluations](#running-individual-framework-evaluations)
5. [Running the Full Comparative Study](#running-the-full-comparative-study)
6. [Understanding the Results](#understanding-the-results)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

- Python 3.10 or higher
- At least 8GB RAM
- Stable internet connection (for API calls)
- ~5GB free disk space for results

### Required Accounts and API Keys

1. **OpenAI API Account**
   - Sign up at https://platform.openai.com/
   - Generate an API key
   - Ensure you have sufficient credits

2. **Langfuse Account** (Optional, for advanced logging)
   - Sign up at https://cloud.langfuse.com/
   - Get public and secret keys

---

## Initial Setup

### Step 1: Clone and Navigate to Project

```bash
cd /path/to/experimental_design
```

### Step 2: Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# Copy the example
cp .env.example .env

# Edit with your keys
nano .env  # or use your preferred editor
```

Add the following to `.env`:

```env
# Required
OPENAI_API_KEY=sk-your-actual-openai-key-here

# Optional (for Langfuse logging)
LANGFUSE_PUBLIC_KEY=pk-lf-your-key-here
LANGFUSE_SECRET_KEY=sk-lf-your-secret-here
LANGFUSE_HOST=https://cloud.langfuse.com
```

### Step 3: Install Dependencies for AutoGen

```bash
cd autogen

# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install autogen-core autogen-ext datasets openai langfuse python-dotenv deepeval

# Verify installation
python -c "import autogen_core; import deepeval; print('✓ AutoGen setup complete')"
```

### Step 4: Install Dependencies for LangGraph

```bash
cd ../langgraph  # From project root

# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install langgraph langchain langchain-openai datasets openai langfuse python-dotenv deepeval

# Verify installation
python -c "import langgraph; import deepeval; print('✓ LangGraph setup complete')"
```

---

## Running a Pilot Study

Before running the full study (which can take hours and cost significant API credits), run a pilot study:

### Quick Pilot Test (Recommended First Step)

```bash
# From project root
python run_full_experiment.py --pilot
```

This will run:
- 5 problems per benchmark
- 2 repetitions
- Both frameworks
- Estimated time: 15-30 minutes
- Estimated cost: $2-5 in API credits

### Custom Pilot Configuration

```bash
python run_full_experiment.py \
  --n-problems 5 \
  --repetitions 2 \
  --temperature 0.3
```

---

## Running Individual Framework Evaluations

### AutoGen Only

```bash
cd autogen
source .venv/bin/activate

# Run all benchmarks
python run_deepeval_benchmarks.py --n-problems 10 --repetitions 3

# Or run specific benchmark
python run_deepeval_benchmarks.py --benchmark gsm8k --n-problems 10 --repetitions 3
python run_deepeval_benchmarks.py --benchmark humaneval --n-problems 10 --repetitions 3
```

### LangGraph Only

```bash
cd langgraph
source .venv/bin/activate

# Run all benchmarks
python run_deepeval_benchmarks.py --n-problems 10 --repetitions 3

# Or run specific benchmark
python run_deepeval_benchmarks.py --benchmark gsm8k --n-problems 10 --repetitions 3
python run_deepeval_benchmarks.py --benchmark humaneval --n-problems 10 --repetitions 3
```

---

## Running the Full Comparative Study

### Full Study (As Per Experimental Design)

According to the experimental design:
- **Single-agent trials**: 50-100 problems, 5-10 repetitions
- **Multi-agent trials**: 25-50 problems, 5-10 repetitions

```bash
# From project root
python run_full_experiment.py \
  --n-problems 100 \
  --repetitions 10 \
  --temperature 0.3
```

**⚠️ Warning**: This will take several hours and cost significant API credits (~$50-100)

### Recommended Phases

#### Phase 1: Small Scale (Validation)
```bash
python run_full_experiment.py --n-problems 10 --repetitions 3
```
- **Duration**: ~1 hour
- **Cost**: ~$5-10
- **Purpose**: Validate setup

#### Phase 2: Medium Scale (Preliminary Results)
```bash
python run_full_experiment.py --n-problems 50 --repetitions 5
```
- **Duration**: ~3-5 hours
- **Cost**: ~$25-40
- **Purpose**: Get preliminary comparative results

#### Phase 3: Full Scale (Final Study)
```bash
python run_full_experiment.py --n-problems 100 --repetitions 10
```
- **Duration**: ~8-12 hours
- **Cost**: ~$50-100
- **Purpose**: Complete experimental design

---

## Understanding the Results

### Result Files Location

After running experiments, find results in:

```
experimental_design/
├── autogen/results/single_agent/
│   ├── autogen_gsm8k_results_TIMESTAMP.json
│   ├── autogen_humaneval_results_TIMESTAMP.json
│   └── autogen_all_benchmarks_TIMESTAMP.json
├── langgraph/results/single_agent/
│   ├── langgraph_gsm8k_results_TIMESTAMP.json
│   ├── langgraph_humaneval_results_TIMESTAMP.json
│   └── langgraph_all_benchmarks_TIMESTAMP.json
└── results/comparative_study/
    ├── framework_comparison_TIMESTAMP.json
    └── full_study_report_TIMESTAMP.json
```

### Reading the Comparison Report

The `framework_comparison_TIMESTAMP.json` file contains:

```json
{
  "study_metadata": {
    "date": "2024-12-12",
    "n_problems_per_benchmark": 100,
    "repetitions_per_task": 10,
    "temperature": 0.3
  },
  "benchmarks": {
    "gsm8k": {
      "autogen": {
        "mean": 0.85,
        "stdev": 0.03,
        "min": 0.80,
        "max": 0.90
      },
      "langgraph": {
        "mean": 0.87,
        "stdev": 0.02,
        "min": 0.84,
        "max": 0.91
      },
      "comparison": {
        "mean_difference": 0.02,
        "winner": "LangGraph"
      }
    }
  }
}
```

### Key Metrics

1. **Mean Score**: Average accuracy across all repetitions
2. **Standard Deviation**: Consistency of results (lower is more consistent)
3. **Winner**: Framework with higher mean score
4. **Difference**: Magnitude of performance gap

---

## Monitoring Progress

### Real-time Monitoring

```bash
# In a separate terminal, watch the results directory
watch -n 5 'ls -lht autogen/results/single_agent/ | head -10'

# Or monitor log output
tail -f nohup.out  # if running in background
```

### Checking Costs

Monitor your OpenAI usage at: https://platform.openai.com/usage

### Estimated Costs by Scale

| Scale | Problems | Reps | Est. API Calls | Est. Cost |
|-------|----------|------|----------------|-----------|
| Pilot | 5 | 2 | ~20 | $2-5 |
| Small | 10 | 3 | ~60 | $5-10 |
| Medium | 50 | 5 | ~500 | $25-40 |
| Full | 100 | 10 | ~2,000 | $50-100 |

---

## Running in Background (Recommended for Long Studies)

### Using nohup

```bash
nohup python run_full_experiment.py --n-problems 100 --repetitions 10 > experiment.log 2>&1 &

# Check progress
tail -f experiment.log

# Check if still running
ps aux | grep run_full_experiment
```

### Using screen

```bash
# Start screen session
screen -S experiment

# Run experiment
python run_full_experiment.py --n-problems 100 --repetitions 10

# Detach: Ctrl+A, then D
# Reattach later: screen -r experiment
```

---

## Troubleshooting

### Issue: "Module not found" errors

**Solution**: Ensure virtual environment is activated

```bash
# Check which Python you're using
which python

# Should show path with .venv
# If not, activate virtual environment
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

### Issue: "OPENAI_API_KEY not found"

**Solution**: Check .env file

```bash
# Verify .env exists
ls -la .env

# Check it has the key
cat .env | grep OPENAI_API_KEY

# Verify it's being loaded
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('OPENAI_API_KEY'))"
```

### Issue: Rate limit errors from OpenAI

**Solution**: Add delays or reduce parallelism

```python
# In the evaluation scripts, add:
import time
time.sleep(1)  # Between API calls
```

### Issue: Out of memory errors

**Solution**: Reduce batch size or n_problems

```bash
# Run with fewer problems
python run_full_experiment.py --n-problems 10 --repetitions 3
```

### Issue: Results files not being created

**Solution**: Check permissions

```bash
# Ensure results directories exist and are writable
mkdir -p autogen/results/single_agent
mkdir -p langgraph/results/single_agent
mkdir -p results/comparative_study

chmod -R 755 autogen/results
chmod -R 755 langgraph/results
chmod -R 755 results
```

---

## Best Practices

1. **Always start with a pilot study** before running the full experiment
2. **Monitor costs closely** during initial runs
3. **Save intermediate results** frequently
4. **Document any modifications** to experimental parameters
5. **Back up results** to multiple locations
6. **Use version control** for code changes

---

## Advanced: Customizing the Evaluation

### Modifying GEval Metrics

Edit `deepeval_metrics.py` to customize evaluation criteria:

```python
# Add your own custom metric
my_custom_metric = GEval(
    name="MyMetric",
    criteria="Your evaluation criteria here",
    evaluation_steps=[
        "Step 1",
        "Step 2"
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7
)
```

### Adding New Benchmarks

To add ARC or MATH benchmarks:

1. Install benchmark: `pip install deepeval[arc]` or `pip install deepeval[math]`
2. Update evaluation scripts to include new benchmarks
3. Add benchmark-specific metrics in `deepeval_metrics.py`

---

## Support and Documentation

- DeepEval Documentation: https://docs.deepeval.com
- AutoGen Documentation: https://microsoft.github.io/autogen/
- LangGraph Documentation: https://langchain-ai.github.io/langgraph/

---

## Citation

If you use this experimental framework, please cite:

```bibtex
@misc{agentic_frameworks_comparison_2024,
  title={Comparative Evaluation of Agentic AI Frameworks: AutoGen vs LangGraph},
  author={[Your Name]},
  year={2024},
  note={DeepEval-based evaluation framework}
}
```

---

**Last Updated**: December 2024
**Version**: 1.0
