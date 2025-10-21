# Meta-Learning PEFT Benchmarks

Comprehensive benchmark and validation suite for the meta-learning PEFT system.

## Overview

This directory contains benchmarking and validation tools for evaluating:
- Meta-learning algorithms (MAML, Reptile, FOMAML, Meta-SGD)
- PEFT methods (LoRA, AdaLoRA, Prompt Tuning)
- Few-shot learning capabilities
- Cross-task transfer
- System correctness and robustness

## Benchmark Modules

### 1. Meta-Learning Benchmark (`meta_learning_benchmark.py`)

Compares meta-learning algorithms across multiple dimensions:

**Metrics:**
- **Adaptation Speed**: How quickly algorithms adapt to new tasks (1-shot, 5-shot, 10-shot)
- **Training Time**: Time per meta-training iteration
- **Memory Usage**: Model parameter count and memory footprint
- **Convergence Rate**: Learning curve during meta-training

**Usage:**
```bash
# Quick benchmark (20 iterations, 5 eval tasks)
python benchmarks/meta_learning_benchmark.py --quick

# Full benchmark (100 iterations, 20 eval tasks)
python benchmarks/meta_learning_benchmark.py
```

**Output:**
- Console report with formatted tables
- JSON results: `outputs/benchmarks/meta_learning_benchmark.json`

### 2. PEFT Comparison Benchmark (`peft_comparison_benchmark.py`)

Compares PEFT methods for parameter efficiency and performance:

**Metrics:**
- **Parameter Efficiency**: Trainable parameters vs total parameters
- **Training Speed**: Seconds per meta-training iteration
- **Inference Speed**: Milliseconds per sample
- **Accuracy**: Final accuracy after meta-training
- **Adaptation Capability**: K-shot learning performance (1, 3, 5-shot)
- **Memory Footprint**: Total memory usage in MB

**Usage:**
```bash
# Quick benchmark
python benchmarks/peft_comparison_benchmark.py --quick

# Full benchmark
python benchmarks/peft_comparison_benchmark.py
```

**Output:**
- Comparison table with all methods side-by-side
- JSON results: `outputs/benchmarks/peft_comparison_benchmark.json`

### 3. Validation Suite (`validation_suite.py`)

Validates system correctness and functionality:

**Tests:**
- **Convergence**: Meta-learning algorithms converge during training
- **Few-Shot Learning**: K-shot improvement (5-shot > 1-shot)
- **Parameter Efficiency**: PEFT reduces parameters by >2x
- **Task Embeddings**: Embeddings generated correctly
- **Task Similarity**: Similarity metrics work correctly
- **Cross-Task Transfer**: Knowledge transfers between tasks
- **Save/Load**: Checkpoints persist correctly

**Usage:**
```bash
# Run validation suite
python benchmarks/validation_suite.py
```

**Output:**
- Pass/Fail report for each component
- JSON results: `outputs/validation/validation_results.json`

## Quick Start

### Run All Benchmarks

```bash
# From project root
cd projects/05_Meta_Learning_PEFT

# Run validation first
python benchmarks/validation_suite.py

# Run meta-learning benchmark
python benchmarks/meta_learning_benchmark.py --quick

# Run PEFT comparison
python benchmarks/peft_comparison_benchmark.py --quick
```

### Interpret Results

**Meta-Learning Benchmark:**
- Look for algorithms with best adaptation speed (highest K-shot accuracy)
- Compare training time vs accuracy tradeoffs
- Check convergence rates (earlier convergence = better)

**PEFT Comparison:**
- LoRA: Best balance of accuracy and efficiency
- AdaLoRA: Most parameter efficient (adaptive ranking)
- Prompt Tuning: Fastest training, fewest parameters

**Validation Suite:**
- All tests should pass (✓ PASS)
- Failed tests indicate system issues requiring investigation

## Configuration

### Benchmark Configuration

Adjust benchmark parameters in code:

```python
from benchmarks import BenchmarkConfig

config = BenchmarkConfig(
    num_meta_iterations=100,  # Meta-training iterations
    meta_batch_size=4,        # Tasks per batch
    num_eval_tasks=20,        # Evaluation tasks
    num_inner_steps=5,        # Gradient steps per task
    inner_lr=0.01,            # Inner loop learning rate
    outer_lr=0.001,           # Meta learning rate
)
```

### Hardware Considerations

- **Memory**: Benchmarks use small models (32 hidden dim) to fit on typical hardware
- **Time**: Full benchmarks take 10-30 minutes depending on hardware
- **Quick Mode**: Use `--quick` flag for faster development testing

## Output Structure

```
outputs/
├── benchmarks/
│   ├── meta_learning_benchmark.json    # Algorithm comparison
│   └── peft_comparison_benchmark.json  # PEFT method comparison
└── validation/
    └── validation_results.json         # Validation test results
```

## Expected Results

### Meta-Learning Algorithms

| Algorithm | 5-shot Accuracy | Training Time | Memory |
|-----------|----------------|---------------|--------|
| MAML      | 0.75-0.85      | ~5-10s/iter   | ~50MB  |
| Reptile   | 0.70-0.80      | ~3-6s/iter    | ~50MB  |
| FOMAML    | 0.70-0.80      | ~3-5s/iter    | ~50MB  |
| Meta-SGD  | 0.75-0.85      | ~6-12s/iter   | ~60MB  |

### PEFT Methods

| Method         | Param Reduction | Accuracy | Training Speed |
|----------------|----------------|----------|----------------|
| LoRA           | 3-5x           | 0.70-0.80| ~4-8s/iter     |
| AdaLoRA        | 5-8x           | 0.70-0.80| ~5-10s/iter    |
| Prompt Tuning  | 10-20x         | 0.65-0.75| ~2-5s/iter     |

*Results vary based on task complexity and hardware*

## Development

### Adding New Benchmarks

1. Create new benchmark class inheriting from base structure
2. Implement metric collection methods
3. Add to `__init__.py` exports
4. Update this README

### Modifying Existing Benchmarks

- Adjust `BenchmarkConfig` or `PEFTBenchmarkConfig` for different settings
- Add new metrics by extending result dataclasses
- Update report generation to include new metrics

## Troubleshooting

**Benchmark fails with memory error:**
- Reduce `hidden_dim` in config
- Reduce `meta_batch_size`
- Use `--quick` mode

**Validation tests fail:**
- Check implementation changes didn't break functionality
- Review test expectations (may need adjustment)
- Check MLX installation and compatibility

**Results vary significantly between runs:**
- Set random seed for reproducibility
- Increase `num_eval_tasks` for more stable estimates
- Average results across multiple runs

## References

- **MAML**: Finn et al. (2017) "Model-Agnostic Meta-Learning"
- **Reptile**: Nichol et al. (2018) "On First-Order Meta-Learning Algorithms"
- **LoRA**: Hu et al. (2021) "LoRA: Low-Rank Adaptation"
- **AdaLoRA**: Zhang et al. (2023) "Adaptive Budget Allocation"
- **Prompt Tuning**: Lester et al. (2021) "The Power of Scale"
