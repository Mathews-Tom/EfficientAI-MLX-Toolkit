# Usage Examples

Practical examples for using the Adaptive Diffusion Optimizer framework.

## Table of Contents

- [Basic Usage](#basic-usage)
- [Adaptive Scheduling](#adaptive-scheduling)
- [Quality-Guided Sampling](#quality-guided-sampling)
- [RL-Based Optimization](#rl-based-optimization)
- [Domain Adaptation](#domain-adaptation)
- [Production Workflows](#production-workflows)
- [Benchmarking](#benchmarking)

---

## Basic Usage

### Simple Image Generation

```python
from adaptive_diffusion import DiffusionPipeline

# Create pipeline with default DDIM scheduler
pipeline = DiffusionPipeline(
    scheduler="ddim",
    image_size=(256, 256),
    in_channels=3
)

# Generate images
images = pipeline.generate(
    batch_size=4,
    num_inference_steps=50,
    seed=42
)

print(f"Generated images shape: {images.shape}")  # (4, 256, 256, 3)

# Get scheduler information
info = pipeline.get_scheduler_info()
print(f"Using {info['scheduler_type']} on {info['device']}")
```

### Using Different Schedulers

```python
from adaptive_diffusion import DiffusionPipeline

# DDPM - High quality, slower
pipeline_ddpm = DiffusionPipeline(scheduler="ddpm")
images_ddpm = pipeline_ddpm.generate(num_inference_steps=1000)

# DDIM - Faster, deterministic
pipeline_ddim = DiffusionPipeline(scheduler="ddim")
images_ddim = pipeline_ddim.generate(num_inference_steps=50)

# DPM-Solver - Very fast, good quality
pipeline_dpm = DiffusionPipeline(scheduler="dpm-solver")
images_dpm = pipeline_dpm.generate(num_inference_steps=20)
```

### Image Denoising

```python
import mlx.core as mx
from adaptive_diffusion import DiffusionPipeline

pipeline = DiffusionPipeline()

# Load or create a noisy image
noisy_image = mx.random.normal((1, 256, 256, 3))

# Denoise starting from timestep 500
denoised = pipeline.denoise_image(
    noisy_image=noisy_image,
    timestep=500,
    num_inference_steps=50
)

print(f"Denoised image shape: {denoised.shape}")
```

---

## Adaptive Scheduling

### Basic Adaptive Scheduling

```python
from adaptive_diffusion.schedulers.adaptive import AdaptiveScheduler
from adaptive_diffusion.baseline.pipeline import DiffusionPipeline

# Create adaptive scheduler
scheduler = AdaptiveScheduler(
    num_inference_steps=50,
    adaptive_threshold=0.5,
    progress_power=2.0,
    min_step_ratio=0.5,
    max_step_ratio=2.0
)

# Create pipeline with adaptive scheduler
pipeline = DiffusionPipeline(scheduler=scheduler)

# Generate with adaptive scheduling
images, intermediates = pipeline.generate(
    batch_size=1,
    return_intermediates=True
)

# Inspect schedule
schedule_info = scheduler.get_schedule_info()
print(f"Average quality: {schedule_info['avg_quality']}")
print(f"Average complexity: {schedule_info['avg_complexity']}")
print(f"Timesteps used: {len(schedule_info['timesteps'])}")
```

### Complexity-Adaptive Scheduling

```python
from adaptive_diffusion.schedulers.adaptive import AdaptiveScheduler
import mlx.core as mx

scheduler = AdaptiveScheduler(num_inference_steps=50)

# Set timesteps based on content complexity
sample = mx.random.normal((1, 256, 256, 3))
complexity = scheduler.estimate_complexity(sample)

scheduler.set_timesteps(
    num_inference_steps=50,
    complexity=complexity
)

print(f"Content complexity: {complexity:.4f}")
print(f"Adapted timesteps: {scheduler.timesteps[:5]}...")  # Show first 5
```

### Quality-Aware Scheduling

```python
from adaptive_diffusion.schedulers.adaptive import AdaptiveScheduler
from adaptive_diffusion.baseline.pipeline import DiffusionPipeline
import mlx.core as mx

scheduler = AdaptiveScheduler(
    num_inference_steps=50,
    adaptive_threshold=0.3  # Lower threshold = more adaptive
)

pipeline = DiffusionPipeline(scheduler=scheduler)

# Generate with quality monitoring
x_t = mx.random.normal((1, 256, 256, 3))

for i, t in enumerate(scheduler.timesteps):
    # Predict noise
    t_batch = mx.array([t])
    model_output = pipeline.model(x_t, t_batch)

    # Estimate quality
    quality = scheduler.estimate_complexity(x_t)

    # Adaptive step with quality feedback
    x_t = scheduler.step(
        model_output=model_output,
        timestep=i,
        sample=x_t,
        quality_estimate=quality
    )

    print(f"Step {i}: t={t}, quality={quality:.4f}")

# View quality history
schedule_info = scheduler.get_schedule_info()
print(f"Quality progression: {schedule_info['quality_history']}")
```

---

## Quality-Guided Sampling

### Basic Quality-Guided Sampling

```python
from adaptive_diffusion.sampling.quality_guided import QualityGuidedSampler
from adaptive_diffusion.schedulers.adaptive import AdaptiveScheduler
from adaptive_diffusion.baseline.pipeline import DiffusionPipeline

# Create scheduler and sampler
scheduler = AdaptiveScheduler(num_inference_steps=50)
sampler = QualityGuidedSampler(
    scheduler=scheduler,
    quality_threshold=0.6,
    early_stopping=True,
    patience=5
)

# Create pipeline
pipeline = DiffusionPipeline(scheduler=scheduler)

# Sample with quality guidance
images = sampler.sample(
    model=pipeline.model,
    batch_size=1,
    image_size=(256, 256),
    seed=42
)

# Analyze quality progression
quality_history = sampler.get_quality_history()
print(f"Steps taken: {len(quality_history)}")
print(f"Final quality: {quality_history[-1]:.4f}")
print(f"Quality improved by: {(quality_history[-1] - quality_history[0]) * 100:.1f}%")
```

### Early Stopping Based on Quality

```python
from adaptive_diffusion.sampling.quality_guided import QualityGuidedSampler
from adaptive_diffusion.schedulers.adaptive import AdaptiveScheduler

scheduler = AdaptiveScheduler(num_inference_steps=100)
sampler = QualityGuidedSampler(
    scheduler=scheduler,
    quality_threshold=0.7,  # Stop when quality reaches 0.7
    early_stopping=True,
    patience=3,  # Stop if quality doesn't improve for 3 steps
    min_steps=20,  # Always run at least 20 steps
    max_steps=100
)

# Generate multiple samples
for i in range(5):
    from adaptive_diffusion.baseline.pipeline import DiffusionPipeline
    pipeline = DiffusionPipeline(scheduler=scheduler)

    images = sampler.sample(
        model=pipeline.model,
        batch_size=1,
        image_size=(256, 256),
        seed=i
    )

    history = sampler.get_quality_history()
    print(f"Sample {i}: {len(history)} steps, quality={history[-1]:.4f}")

    sampler.reset_history()
```

### Quality Metrics Monitoring

```python
from adaptive_diffusion.sampling.quality_guided import QualityGuidedSampler
from adaptive_diffusion.schedulers.adaptive import AdaptiveScheduler
import mlx.core as mx

scheduler = AdaptiveScheduler(num_inference_steps=50)
sampler = QualityGuidedSampler(scheduler=scheduler, quality_threshold=0.6)

# Monitor quality during generation
sample = mx.random.normal((1, 256, 256, 3))
quality = sampler.estimate_quality(sample)

print(f"Initial quality: {quality:.4f}")

# Simulate progressive denoising
for step in range(10):
    # Add some refinement (simulation)
    sample = sample * 0.9 + mx.random.normal(sample.shape) * 0.1

    quality = sampler.estimate_quality(sample)
    print(f"Step {step}: quality={quality:.4f}")
```

---

## RL-Based Optimization

### Training the RL Agent

```python
from adaptive_diffusion.rl.ppo_agent import HyperparameterTuningAgent
from adaptive_diffusion.rl.environment import DiffusionHyperparameterEnv
from adaptive_diffusion.rl.reward import create_reward_function

# Create reward function
reward_fn = create_reward_function(
    "quality_speed",
    quality_weight=0.7,
    speed_weight=0.3
)

# Create environment
env = DiffusionHyperparameterEnv(
    reward_function=reward_fn,
    max_steps=20,
    target_quality=0.8,
    target_speed=2.0
)

# Create agent
agent = HyperparameterTuningAgent(
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    verbose=1
)

# Train agent
print("Training RL agent...")
training_stats = agent.train(
    total_timesteps=10000,
    eval_freq=2000,
    n_eval_episodes=5
)

print(f"Training completed:")
print(f"  Total timesteps: {training_stats['total_timesteps']}")
print(f"  Convergence achieved: {training_stats['convergence_achieved']}")
print(f"  Best reward: {training_stats.get('best_reward', 'N/A')}")

# Save trained agent
agent.save("models/ppo_agent.zip")
```

### Using Trained Agent for Optimization

```python
from adaptive_diffusion.rl.ppo_agent import HyperparameterTuningAgent
from adaptive_diffusion.rl.environment import DiffusionHyperparameterEnv

# Load pre-trained agent
env = DiffusionHyperparameterEnv()
agent = HyperparameterTuningAgent(env=env)
agent.load("models/ppo_agent.zip")

# Optimize hyperparameters
print("Optimizing hyperparameters...")
results = agent.optimize_hyperparameters(
    num_episodes=10,
    deterministic=True
)

print(f"\nOptimization Results:")
print(f"  Best quality: {results['best_quality']:.4f}")
print(f"  Best speed: {results['best_speed']:.2f}x")
print(f"  Mean quality: {results['mean_quality']:.4f}")
print(f"  Mean speed: {results['mean_speed']:.2f}x")
print(f"\nBest hyperparameters:")
for key, value in results['best_hyperparameters'].items():
    print(f"  {key}: {value}")
```

### Custom Reward Functions

```python
from adaptive_diffusion.rl.environment import DiffusionHyperparameterEnv
from adaptive_diffusion.rl.reward import RewardFunction
from adaptive_diffusion.rl.ppo_agent import HyperparameterTuningAgent

# Define custom reward function
class CustomReward(RewardFunction):
    def __call__(self, quality: float, speed: float, num_steps: int) -> float:
        # Prioritize quality over speed
        quality_reward = quality * 10.0

        # Penalize excessive steps
        step_penalty = (num_steps - 30) * 0.1 if num_steps > 30 else 0

        # Bonus for speedup
        speed_bonus = (speed - 1.0) * 2.0 if speed > 1.0 else 0

        return quality_reward + speed_bonus - step_penalty

# Use custom reward
reward_fn = CustomReward()
env = DiffusionHyperparameterEnv(reward_function=reward_fn)
agent = HyperparameterTuningAgent(env=env)

# Train with custom reward
training_stats = agent.train(total_timesteps=5000)
```

---

## Domain Adaptation

### Using Pre-configured Domains

```python
from adaptive_diffusion.optimization.domain_adapter import (
    DomainAdapter,
    DomainType
)

adapter = DomainAdapter()

# Get configurations for different domains
portrait_config = adapter.get_config(domain_type=DomainType.PORTRAITS)
landscape_config = adapter.get_config(domain_type=DomainType.LANDSCAPES)
abstract_config = adapter.get_config(domain_type=DomainType.ABSTRACT)

print("Portrait settings:")
print(f"  Steps: {portrait_config.num_steps}")
print(f"  Threshold: {portrait_config.adaptive_threshold}")
print(f"  Quality weight: {portrait_config.quality_weight}")

print("\nLandscape settings:")
print(f"  Steps: {landscape_config.num_steps}")
print(f"  Threshold: {landscape_config.adaptive_threshold}")
print(f"  Speed weight: {landscape_config.speed_weight}")
```

### Automatic Domain Detection

```python
from adaptive_diffusion.optimization.domain_adapter import DomainAdapter
import mlx.core as mx

adapter = DomainAdapter()

# Detect from prompt
domain = adapter.detect_domain(
    prompt="A photorealistic portrait of a person"
)
print(f"Detected domain: {domain.value}")

# Detect from sample
sample = mx.random.normal((1, 256, 256, 3))
domain = adapter.detect_domain(sample=sample)
print(f"Detected from sample: {domain.value}")

# Get optimal config for detected domain
config = adapter.get_config(
    prompt="A beautiful mountain landscape"
)
print(f"Auto-detected: {config.domain_type.value}")
```

### Learning from Results

```python
from adaptive_diffusion.optimization.domain_adapter import (
    DomainAdapter,
    DomainType
)

adapter = DomainAdapter()

# Simulate optimization results
for i in range(5):
    # Run optimization (simulated)
    quality = 0.75 + i * 0.02
    speed = 1.5 + i * 0.1
    hyperparameters = {
        "num_steps": 40 - i,
        "adaptive_threshold": 0.5 + i * 0.02,
        "progress_power": 2.0 + i * 0.05
    }

    # Learn from results
    adapter.learn_from_results(
        domain_type=DomainType.PORTRAITS,
        quality=quality,
        speed=speed,
        hyperparameters=hyperparameters
    )

    print(f"Trial {i}: quality={quality:.4f}, speed={speed:.2f}x")

# Get updated config
updated_config = adapter.get_config(domain_type=DomainType.PORTRAITS)
print(f"\nLearned configuration:")
print(f"  Steps: {updated_config.num_steps}")
print(f"  Threshold: {updated_config.adaptive_threshold}")
```

---

## Production Workflows

### Complete Optimization Pipeline

```python
from adaptive_diffusion.optimization.pipeline import OptimizationPipeline
from adaptive_diffusion.optimization.domain_adapter import DomainType
from adaptive_diffusion.baseline.pipeline import DiffusionPipeline

# Create optimization pipeline
opt_pipeline = OptimizationPipeline(
    use_domain_adaptation=True,
    use_rl_optimization=True,
    verbose=1
)

# Optimize for specific domain
print("Running optimization...")
results = opt_pipeline.optimize(
    domain_type=DomainType.PORTRAITS,
    num_training_steps=5000,
    num_optimization_episodes=10
)

print(f"\nOptimization complete!")
print(f"Domain: {results['domain_type']}")
print(f"Final config: {results['final_config']}")

# Create optimized components
scheduler = opt_pipeline.create_optimized_scheduler(
    config=results['final_config']
)
sampler = opt_pipeline.create_optimized_sampler(
    scheduler=scheduler
)

# Generate images
pipeline = DiffusionPipeline(scheduler=scheduler)
images = sampler.sample(
    model=pipeline.model,
    batch_size=4,
    image_size=(256, 256)
)

print(f"Generated {images.shape[0]} optimized images")

# Save pipeline
opt_pipeline.save("models/optimization_pipeline")
```

### Batch Processing with Optimization

```python
from adaptive_diffusion.optimization.pipeline import OptimizationPipeline
from adaptive_diffusion.optimization.domain_adapter import DomainType
from adaptive_diffusion.baseline.pipeline import DiffusionPipeline

# Setup
opt_pipeline = OptimizationPipeline()
domains = [
    DomainType.PORTRAITS,
    DomainType.LANDSCAPES,
    DomainType.ABSTRACT
]

# Optimize for each domain
optimized_samplers = {}

for domain in domains:
    print(f"\nOptimizing for {domain.value}...")

    # Run optimization
    results = opt_pipeline.optimize(
        domain_type=domain,
        num_training_steps=3000,
        num_optimization_episodes=5
    )

    # Create optimized sampler
    sampler = opt_pipeline.create_optimized_sampler(
        domain_type=domain,
        config=results['final_config']
    )

    optimized_samplers[domain] = sampler

    print(f"  Config: {results['final_config']}")

# Use optimized samplers for generation
for domain, sampler in optimized_samplers.items():
    pipeline = DiffusionPipeline(scheduler=sampler.scheduler)
    images = sampler.sample(
        model=pipeline.model,
        batch_size=2,
        image_size=(256, 256)
    )
    print(f"{domain.value}: generated {images.shape[0]} images")
```

### Progressive Enhancement

```python
from adaptive_diffusion.optimization.pipeline import OptimizationPipeline
from adaptive_diffusion.baseline.pipeline import DiffusionPipeline
import mlx.core as mx

opt_pipeline = OptimizationPipeline()

# Stage 1: Quick optimization
print("Stage 1: Quick optimization")
results_quick = opt_pipeline.optimize(
    num_training_steps=1000,
    num_optimization_episodes=3
)

# Stage 2: Refined optimization
print("\nStage 2: Refined optimization")
results_refined = opt_pipeline.optimize(
    num_training_steps=5000,
    num_optimization_episodes=10
)

# Compare results
print(f"\nQuick: {results_quick['final_config']}")
print(f"Refined: {results_refined['final_config']}")

# Use refined configuration
sampler = opt_pipeline.create_optimized_sampler(
    config=results_refined['final_config']
)

pipeline = DiffusionPipeline(scheduler=sampler.scheduler)
images = sampler.sample(
    model=pipeline.model,
    batch_size=4,
    image_size=(256, 256)
)
```

---

## Benchmarking

### Performance Benchmarking

```python
from adaptive_diffusion.baseline.pipeline import DiffusionPipeline
from adaptive_diffusion.schedulers.adaptive import AdaptiveScheduler
from adaptive_diffusion.sampling.quality_guided import QualityGuidedSampler
import time

# Baseline
print("Benchmarking baseline...")
pipeline_baseline = DiffusionPipeline(scheduler="ddim")

start = time.time()
images_baseline = pipeline_baseline.generate(
    batch_size=4,
    num_inference_steps=50
)
time_baseline = time.time() - start

# Adaptive
print("Benchmarking adaptive...")
scheduler_adaptive = AdaptiveScheduler(num_inference_steps=50)
pipeline_adaptive = DiffusionPipeline(scheduler=scheduler_adaptive)

start = time.time()
images_adaptive = pipeline_adaptive.generate(
    batch_size=4,
    num_inference_steps=50
)
time_adaptive = time.time() - start

# Quality-guided
print("Benchmarking quality-guided...")
sampler = QualityGuidedSampler(
    scheduler=scheduler_adaptive,
    quality_threshold=0.6,
    early_stopping=True
)

start = time.time()
images_guided = sampler.sample(
    model=pipeline_adaptive.model,
    batch_size=4,
    image_size=(256, 256)
)
time_guided = time.time() - start

# Results
print(f"\nBenchmark Results:")
print(f"Baseline: {time_baseline:.2f}s")
print(f"Adaptive: {time_adaptive:.2f}s ({time_baseline/time_adaptive:.2f}x)")
print(f"Quality-guided: {time_guided:.2f}s ({time_baseline/time_guided:.2f}x)")
print(f"Steps: {len(sampler.get_quality_history())}")
```

### Quality Metrics Comparison

```python
from adaptive_diffusion.baseline.pipeline import DiffusionPipeline
from adaptive_diffusion.optimization.pipeline import OptimizationPipeline
from adaptive_diffusion.sampling.quality_guided import QualityGuidedSampler

# Generate with different methods
methods = {
    "baseline": DiffusionPipeline(scheduler="ddim"),
    "optimized": None  # Will create via optimization
}

# Run optimization
opt_pipeline = OptimizationPipeline()
results = opt_pipeline.optimize(num_training_steps=2000)

sampler = opt_pipeline.create_optimized_sampler(
    config=results['final_config']
)

# Generate samples
print("Generating samples...")
samples = {}

# Baseline
samples["baseline"] = methods["baseline"].generate(
    batch_size=4,
    num_inference_steps=50
)

# Optimized
pipeline_opt = DiffusionPipeline(scheduler=sampler.scheduler)
samples["optimized"] = sampler.sample(
    model=pipeline_opt.model,
    batch_size=4,
    image_size=(256, 256)
)

# Compare
for method, images in samples.items():
    quality = sampler.estimate_quality(images)
    print(f"{method}: quality={quality:.4f}")
```

---

## Advanced Examples

### Custom Scheduler Configuration

```python
from adaptive_diffusion.schedulers.adaptive import AdaptiveScheduler

# Create highly adaptive scheduler
scheduler = AdaptiveScheduler(
    num_inference_steps=50,
    adaptive_threshold=0.3,  # Very sensitive
    progress_power=3.0,  # Aggressive progress weighting
    min_step_ratio=0.3,  # Allow very small steps
    max_step_ratio=3.0   # Allow very large steps
)

# Use for specific content
scheduler.set_timesteps(
    num_inference_steps=50,
    complexity=0.8  # High complexity content
)

info = scheduler.get_schedule_info()
print(f"Adaptive configuration:")
print(f"  Timesteps: {len(info['timesteps'])}")
print(f"  Weight distribution: min={min(info['step_weights']):.4f}, max={max(info['step_weights']):.4f}")
```

### Combining Multiple Techniques

```python
from adaptive_diffusion.optimization.pipeline import OptimizationPipeline
from adaptive_diffusion.optimization.domain_adapter import DomainType
from adaptive_diffusion.sampling.quality_guided import QualityGuidedSampler

# Full optimization stack
opt_pipeline = OptimizationPipeline(
    use_domain_adaptation=True,
    use_rl_optimization=True
)

# Optimize
results = opt_pipeline.optimize(
    domain_type=DomainType.PORTRAITS,
    num_training_steps=5000,
    num_optimization_episodes=10
)

# Create components
scheduler = opt_pipeline.create_optimized_scheduler(
    config=results['final_config']
)

sampler = QualityGuidedSampler(
    scheduler=scheduler,
    quality_threshold=0.7,
    early_stopping=True,
    patience=3
)

# Generate
from adaptive_diffusion.baseline.pipeline import DiffusionPipeline
pipeline = DiffusionPipeline(scheduler=scheduler)

images = sampler.sample(
    model=pipeline.model,
    batch_size=1,
    image_size=(256, 256)
)

print(f"Fully optimized generation:")
print(f"  Domain: {results['domain_type']}")
print(f"  Steps taken: {len(sampler.get_quality_history())}")
print(f"  Final quality: {sampler.get_quality_history()[-1]:.4f}")
```

---

For complete API documentation, see [api.md](api.md).
