# Benchmark on Model Update Hook

## Description
Automatically run performance benchmarks when model files or training scripts are updated to track performance regressions and improvements.

## Trigger
File save events for model-related files

## Actions
1. Run performance benchmarks for the updated model
2. Compare results with previous benchmarks
3. Generate performance reports
4. Alert on significant performance changes

## Configuration
```yaml
name: "Benchmark on Model Update"
trigger:
  event: "file_save"
  pattern: 
    - "**/models/*.py"
    - "**/training/*.py"
    - "**/src/train.py"
    - "**/src/inference.py"
conditions:
  - file_exists: "benchmarks/"
  - apple_silicon_available: true
actions:
  - run_command: "uv run python -m utils.benchmark_runner --model ${project_name} --quick"
  - generate_report:
      template: "benchmark_report.md"
      output: "benchmarks/reports/${timestamp}_${project_name}.md"
  - compare_with_baseline:
      baseline_file: "benchmarks/baseline_${project_name}.json"
      threshold: 0.05  # 5% performance change threshold
notifications:
  performance_improvement: "🚀 Performance improved by ${improvement_percent}% in ${project_name}"
  performance_regression: "⚠️ Performance regression detected in ${project_name}: ${regression_percent}%"
  benchmark_complete: "📊 Benchmark complete for ${project_name}"
```

## Expected Behavior
- Automatic performance tracking on model changes
- Quick benchmarks for fast feedback
- Performance regression detection
- Historical performance tracking