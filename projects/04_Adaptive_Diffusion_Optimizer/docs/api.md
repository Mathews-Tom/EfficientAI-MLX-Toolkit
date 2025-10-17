# API Reference

Complete API documentation for the Adaptive Diffusion Optimizer framework.

## Table of Contents

- [Baseline Pipeline](#baseline-pipeline)
- [Schedulers](#schedulers)
- [Sampling](#sampling)
- [Reinforcement Learning](#reinforcement-learning)
- [Optimization](#optimization)
- [Metrics](#metrics)

---

## Baseline Pipeline

### DiffusionPipeline

MLX-optimized diffusion pipeline for image generation.

```python
from adaptive_diffusion.baseline.pipeline import DiffusionPipeline
```

#### Constructor

```python
DiffusionPipeline(
    model: nn.Module | None = None,
    scheduler: BaseScheduler | str | None = None,
    image_size: tuple[int, int] = (256, 256),
    in_channels: int = 3
)
```

**Parameters:**
- `model`: U-Net model instance. If None, uses `SimpleUNet`
- `scheduler`: Scheduler instance or name ('ddpm', 'ddim', 'dpm-solver')
- `image_size`: Generated image size as (height, width)
- `in_channels`: Number of image channels (3 for RGB)

**Attributes:**
- `model`: U-Net model
- `scheduler`: Noise scheduler
- `image_size`: Image dimensions
- `in_channels`: Channel count
- `device`: Detected hardware ('apple_silicon' or 'cpu')

#### Methods

##### generate()

Generate images from random noise.

```python
generate(
    batch_size: int = 1,
    num_inference_steps: int | None = None,
    seed: int | None = None,
    return_intermediates: bool = False
) -> mx.array | tuple[mx.array, list[mx.array]]
```

**Parameters:**
- `batch_size`: Number of images to generate
- `num_inference_steps`: Number of denoising steps (overrides scheduler default)
- `seed`: Random seed for reproducibility
- `return_intermediates`: If True, return intermediate denoising steps

**Returns:**
- Images as MLX array [B, H, W, C] (NHWC format)
- If `return_intermediates=True`: tuple of (images, intermediates)

**Example:**
```python
pipeline = DiffusionPipeline(scheduler="ddim")
images = pipeline.generate(batch_size=4, num_inference_steps=50, seed=42)
# images.shape: (4, 256, 256, 3)
```

##### denoise_image()

Denoise an image starting from given timestep.

```python
denoise_image(
    noisy_image: mx.array,
    timestep: int,
    num_inference_steps: int | None = None,
    return_intermediates: bool = False
) -> mx.array | tuple[mx.array, list[mx.array]]
```

**Parameters:**
- `noisy_image`: Noisy input image [B, H, W, C]
- `timestep`: Starting timestep
- `num_inference_steps`: Number of denoising steps
- `return_intermediates`: If True, return intermediate steps

**Returns:**
- Denoised image [B, H, W, C]

##### add_noise()

Add noise to images using forward diffusion process.

```python
add_noise(
    images: mx.array,
    timesteps: mx.array,
    noise: mx.array | None = None
) -> mx.array
```

**Parameters:**
- `images`: Clean images [B, H, W, C]
- `timesteps`: Timesteps for each image [B]
- `noise`: Noise to add (if None, samples random noise)

**Returns:**
- Noisy images [B, H, W, C]

##### save_model() / load_model()

```python
save_model(path: str | Path)
load_model(path: str | Path)
```

Save and load model weights.

##### get_scheduler_info()

```python
get_scheduler_info() -> dict[str, Any]
```

Get information about current scheduler configuration.

**Returns:**
- Dictionary with scheduler metadata

---

## Schedulers

### BaseScheduler

Abstract base class for noise schedulers.

```python
from adaptive_diffusion.baseline.schedulers import BaseScheduler
```

#### Methods

```python
set_timesteps(num_inference_steps: int)
add_noise(original_samples, noise, timesteps) -> mx.array
step(model_output, timestep, sample) -> mx.array
```

### Standard Schedulers

#### DDPMScheduler

Denoising Diffusion Probabilistic Models scheduler.

```python
from adaptive_diffusion.baseline.schedulers import DDPMScheduler

scheduler = DDPMScheduler(
    num_train_timesteps: int = 1000,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    beta_schedule: str = "linear"
)
```

#### DDIMScheduler

Denoising Diffusion Implicit Models scheduler (faster inference).

```python
from adaptive_diffusion.baseline.schedulers import DDIMScheduler

scheduler = DDIMScheduler(
    num_train_timesteps: int = 1000,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    beta_schedule: str = "linear",
    eta: float = 0.0  # DDIM parameter (0.0 = deterministic)
)
```

#### DPMSolverScheduler

Fast ODE solver for diffusion models.

```python
from adaptive_diffusion.baseline.schedulers import DPMSolverScheduler

scheduler = DPMSolverScheduler(
    num_train_timesteps: int = 1000,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    beta_schedule: str = "linear",
    order: int = 2  # Solver order (1, 2, or 3)
)
```

### AdaptiveScheduler

Progress-based adaptive noise scheduler with dynamic scheduling.

```python
from adaptive_diffusion.schedulers.adaptive import AdaptiveScheduler
```

#### Constructor

```python
AdaptiveScheduler(
    num_train_timesteps: int = 1000,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    beta_schedule: str = "linear",
    num_inference_steps: int = 50,
    adaptive_threshold: float = 0.5,
    progress_power: float = 2.0,
    min_step_ratio: float = 0.5,
    max_step_ratio: float = 2.0
)
```

**Parameters:**
- `num_train_timesteps`: Number of training timesteps
- `beta_start`, `beta_end`: Beta schedule range
- `beta_schedule`: Beta schedule type ('linear', 'scaled_linear', 'squaredcos_cap_v2')
- `num_inference_steps`: Base number of inference steps
- `adaptive_threshold`: Threshold for triggering adaptive behavior (0-1)
- `progress_power`: Power for progress-based weighting (higher = more aggressive)
- `min_step_ratio`, `max_step_ratio`: Step size adjustment bounds

**Attributes:**
- `timesteps`: Computed timesteps array
- `step_weights`: Progress-based step weights
- `quality_history`: Historical quality estimates
- `complexity_estimates`: Historical complexity estimates

#### Methods

##### set_timesteps()

```python
set_timesteps(
    num_inference_steps: int,
    complexity: float | None = None
)
```

Set adaptive timesteps based on progress and complexity.

**Parameters:**
- `num_inference_steps`: Number of inference steps
- `complexity`: Estimated content complexity (0-1 scale)

##### step()

```python
step(
    model_output: mx.array,
    timestep: int,
    sample: mx.array,
    quality_estimate: float | None = None
) -> mx.array
```

Perform adaptive reverse diffusion step with quality-based adjustment.

**Parameters:**
- `model_output`: Predicted noise from model
- `timestep`: Current timestep index
- `sample`: Current noisy sample
- `quality_estimate`: Optional quality estimate for current sample

**Returns:**
- Denoised sample at previous timestep

##### estimate_complexity()

```python
estimate_complexity(sample: mx.array) -> float
```

Estimate content complexity using variance and edge information.

**Returns:**
- Complexity estimate (0-1 scale)

##### get_schedule_info()

```python
get_schedule_info() -> dict[str, Any]
```

Get information about current adaptive schedule.

**Returns:**
- Dictionary with schedule statistics including:
  - `num_steps`: Number of timesteps
  - `timesteps`: Timestep array
  - `step_weights`: Weight array
  - `quality_history`: Historical quality values
  - `complexity_estimates`: Historical complexity values
  - `avg_quality`: Average quality
  - `avg_complexity`: Average complexity

##### reset_history()

```python
reset_history()
```

Reset quality and complexity history.

---

## Sampling

### QualityGuidedSampler

Quality-guided sampling with early stopping and adaptive step adjustment.

```python
from adaptive_diffusion.sampling.quality_guided import QualityGuidedSampler
```

#### Constructor

```python
QualityGuidedSampler(
    scheduler: BaseScheduler,
    quality_threshold: float = 0.6,
    early_stopping: bool = True,
    patience: int = 5,
    min_steps: int = 10,
    max_steps: int = 100
)
```

**Parameters:**
- `scheduler`: Base scheduler (typically AdaptiveScheduler)
- `quality_threshold`: Quality threshold for early stopping (0-1)
- `early_stopping`: Whether to enable early stopping
- `patience`: Number of steps to wait before stopping
- `min_steps`: Minimum sampling steps
- `max_steps`: Maximum sampling steps

#### Methods

##### sample()

```python
sample(
    model: nn.Module,
    batch_size: int = 1,
    image_size: tuple[int, int] = (256, 256),
    in_channels: int = 3,
    seed: int | None = None
) -> mx.array
```

Generate samples with quality-guided adaptive sampling.

**Parameters:**
- `model`: U-Net model
- `batch_size`: Number of samples
- `image_size`: Image dimensions
- `in_channels`: Number of channels
- `seed`: Random seed

**Returns:**
- Generated images [B, H, W, C]

##### estimate_quality()

```python
estimate_quality(sample: mx.array) -> float
```

Estimate sample quality using variance and sharpness metrics.

**Returns:**
- Quality estimate (0-1 scale)

##### get_quality_history()

```python
get_quality_history() -> list[float]
```

Get historical quality estimates.

##### reset_history()

```python
reset_history()
```

Reset quality history.

### StepReductionSampler

Intelligent step reduction while maintaining quality.

```python
from adaptive_diffusion.sampling.step_reduction import StepReductionSampler

sampler = StepReductionSampler(
    scheduler=scheduler,
    target_reduction: float = 0.5,  # 50% step reduction
    quality_threshold: float = 0.7
)
```

---

## Reinforcement Learning

### DiffusionHyperparameterEnv

Gymnasium environment for diffusion hyperparameter optimization.

```python
from adaptive_diffusion.rl.environment import DiffusionHyperparameterEnv
```

#### Constructor

```python
DiffusionHyperparameterEnv(
    reward_function: RewardFunction | None = None,
    max_steps: int = 20,
    target_quality: float = 0.8,
    target_speed: float = 2.0
)
```

**Parameters:**
- `reward_function`: Custom reward function
- `max_steps`: Maximum steps per episode
- `target_quality`: Target quality metric
- `target_speed`: Target speedup factor

#### Spaces

**Observation Space:**
```python
gym.spaces.Box(
    low=0, high=1,
    shape=(6,),  # [quality, speed, num_steps_normalized, threshold, power, step]
    dtype=np.float32
)
```

**Action Space:**
```python
gym.spaces.Box(
    low=-1, high=1,
    shape=(3,),  # [num_steps_delta, threshold_delta, power_delta]
    dtype=np.float32
)
```

#### Methods

##### reset()

```python
reset(seed: int | None = None) -> tuple[np.ndarray, dict]
```

Reset environment to initial state.

##### step()

```python
step(action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]
```

Execute action and return (observation, reward, terminated, truncated, info).

### HyperparameterTuningAgent

PPO-based agent for hyperparameter optimization.

```python
from adaptive_diffusion.rl.ppo_agent import HyperparameterTuningAgent
```

#### Constructor

```python
HyperparameterTuningAgent(
    env: gym.Env | None = None,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    normalize_env: bool = True,
    device: str = "auto",
    verbose: int = 1
)
```

**Parameters:**
- Standard PPO hyperparameters
- `normalize_env`: Whether to normalize observations/rewards
- `device`: Training device ('auto', 'cpu', 'cuda')

#### Methods

##### train()

```python
train(
    total_timesteps: int,
    callback: BaseCallback | list[BaseCallback] | None = None,
    eval_env: gym.Env | None = None,
    eval_freq: int = 10000,
    n_eval_episodes: int = 5,
    convergence_threshold: float = 0.01,
    convergence_window: int = 5
) -> dict[str, Any]
```

Train the PPO agent.

**Parameters:**
- `total_timesteps`: Total training timesteps
- `callback`: Optional Stable-Baselines3 callback(s)
- `eval_env`: Optional evaluation environment
- `eval_freq`: Evaluation frequency
- `n_eval_episodes`: Number of evaluation episodes
- `convergence_threshold`: Reward std threshold for convergence
- `convergence_window`: Window for convergence check

**Returns:**
- Training statistics dictionary

##### predict()

```python
predict(
    observation: np.ndarray,
    deterministic: bool = True
) -> tuple[np.ndarray, np.ndarray | None]
```

Predict action for given observation.

**Returns:**
- Tuple of (action, value_estimate)

##### optimize_hyperparameters()

```python
optimize_hyperparameters(
    num_episodes: int = 10,
    deterministic: bool = True
) -> dict[str, Any]
```

Optimize hyperparameters using trained agent.

**Returns:**
- Dictionary with:
  - `episodes`: List of episode results
  - `best_quality`: Best quality achieved
  - `best_speed`: Best speed achieved
  - `best_hyperparameters`: Optimal hyperparameters
  - `mean_quality`: Average quality
  - `mean_speed`: Average speed

##### save() / load()

```python
save(path: str | Path)
load(path: str | Path)
```

Save and load trained agent.

##### get_config()

```python
get_config() -> dict[str, Any]
```

Get agent configuration.

### RewardFunction

Reward function for RL optimization.

```python
from adaptive_diffusion.rl.reward import RewardFunction, create_reward_function

# Create reward function
reward_fn = create_reward_function(
    "quality_speed",  # or "quality", "speed"
    quality_weight=0.7,
    speed_weight=0.3
)

# Compute reward
reward = reward_fn(quality=0.85, speed=1.8, num_steps=35)
```

---

## Optimization

### OptimizationPipeline

Unified optimization pipeline combining domain adaptation and RL.

```python
from adaptive_diffusion.optimization.pipeline import OptimizationPipeline
```

#### Constructor

```python
OptimizationPipeline(
    domain_adapter: DomainAdapter | None = None,
    rl_agent: HyperparameterTuningAgent | None = None,
    reward_function: RewardFunction | None = None,
    use_domain_adaptation: bool = True,
    use_rl_optimization: bool = True,
    verbose: int = 1
)
```

**Parameters:**
- `domain_adapter`: Domain adapter instance
- `rl_agent`: RL agent instance
- `reward_function`: Custom reward function
- `use_domain_adaptation`: Enable domain adaptation
- `use_rl_optimization`: Enable RL optimization
- `verbose`: Verbosity level

#### Methods

##### optimize()

```python
optimize(
    domain_type: DomainType | None = None,
    prompt: str | None = None,
    sample: mx.array | None = None,
    num_training_steps: int = 1000,
    num_optimization_episodes: int = 10
) -> dict[str, Any]
```

Run full optimization pipeline.

**Parameters:**
- `domain_type`: Explicit domain type (auto-detected if None)
- `prompt`: Optional prompt for domain detection
- `sample`: Optional sample for domain detection
- `num_training_steps`: RL training steps
- `num_optimization_episodes`: Optimization episodes

**Returns:**
- Results dictionary with:
  - `domain_type`: Detected domain
  - `domain_config`: Domain-specific configuration
  - `rl_optimized_config`: RL-optimized configuration
  - `final_config`: Combined final configuration
  - `training_stats`: RL training statistics
  - `optimization_results`: Optimization results

##### create_optimized_scheduler()

```python
create_optimized_scheduler(
    config: dict[str, Any] | None = None,
    domain_type: DomainType | None = None,
    prompt: str | None = None
) -> AdaptiveScheduler
```

Create adaptive scheduler with optimized configuration.

**Returns:**
- Configured AdaptiveScheduler

##### create_optimized_sampler()

```python
create_optimized_sampler(
    scheduler: AdaptiveScheduler | None = None,
    config: dict[str, Any] | None = None,
    domain_type: DomainType | None = None,
    prompt: str | None = None
) -> QualityGuidedSampler
```

Create quality-guided sampler with optimized configuration.

**Returns:**
- Configured QualityGuidedSampler

##### save() / load()

```python
save(path: str | Path)
load(path: str | Path)
```

Save and load pipeline state.

##### get_history()

```python
get_history() -> list[dict[str, Any]]
```

Get optimization history.

### DomainAdapter

Domain-specific optimization and adaptation.

```python
from adaptive_diffusion.optimization.domain_adapter import (
    DomainAdapter,
    DomainType,
    DomainConfig
)
```

#### DomainType Enum

```python
class DomainType(Enum):
    PORTRAITS = "portraits"
    LANDSCAPES = "landscapes"
    ABSTRACT = "abstract"
    ARCHITECTURE = "architecture"
    GENERAL = "general"
```

#### Constructor

```python
DomainAdapter()
```

#### Methods

##### get_config()

```python
get_config(
    domain_type: DomainType | None = None,
    sample: mx.array | None = None,
    prompt: str | None = None
) -> DomainConfig
```

Get domain-specific configuration.

**Parameters:**
- `domain_type`: Explicit domain type
- `sample`: Sample for automatic detection
- `prompt`: Prompt for detection

**Returns:**
- DomainConfig with optimized settings

##### learn_from_results()

```python
learn_from_results(
    domain_type: DomainType,
    quality: float,
    speed: float,
    hyperparameters: dict[str, Any]
)
```

Learn from optimization results to improve domain configs.

##### detect_domain()

```python
detect_domain(
    sample: mx.array | None = None,
    prompt: str | None = None
) -> DomainType
```

Automatically detect domain from sample or prompt.

---

## Metrics

### Quality Metrics

```python
from adaptive_diffusion.metrics.quality import (
    compute_fid,
    compute_clip_score,
    compute_perceptual_distance
)

# Frechet Inception Distance
fid = compute_fid(real_images, generated_images)

# CLIP Score
clip_score = compute_clip_score(images, prompts)

# Perceptual Distance (LPIPS)
lpips = compute_perceptual_distance(image1, image2)
```

---

## Complete Example

```python
from adaptive_diffusion import DiffusionPipeline
from adaptive_diffusion.optimization.pipeline import OptimizationPipeline
from adaptive_diffusion.optimization.domain_adapter import DomainType

# Step 1: Create optimization pipeline
opt_pipeline = OptimizationPipeline(
    use_domain_adaptation=True,
    use_rl_optimization=True
)

# Step 2: Optimize for specific domain
results = opt_pipeline.optimize(
    domain_type=DomainType.PORTRAITS,
    num_training_steps=5000,
    num_optimization_episodes=10
)

print(f"Optimized config: {results['final_config']}")

# Step 3: Create optimized components
scheduler = opt_pipeline.create_optimized_scheduler(
    domain_type=DomainType.PORTRAITS
)
sampler = opt_pipeline.create_optimized_sampler(
    scheduler=scheduler,
    domain_type=DomainType.PORTRAITS
)

# Step 4: Generate images
pipeline = DiffusionPipeline(scheduler=scheduler)
images = sampler.sample(
    model=pipeline.model,
    batch_size=4,
    image_size=(256, 256)
)

print(f"Generated {images.shape[0]} images")
print(f"Quality history: {sampler.get_quality_history()}")
```

---

## Error Handling

All components raise standard Python exceptions:

```python
try:
    pipeline = DiffusionPipeline(scheduler="invalid")
except ValueError as e:
    print(f"Invalid scheduler: {e}")

try:
    agent.load("nonexistent.zip")
except FileNotFoundError as e:
    print(f"Model not found: {e}")
```

## Type Hints

All components use full type hints for IDE support:

```python
from typing import Optional, Union, Tuple, List, Dict, Any
import mlx.core as mx
import mlx.nn as nn

def generate(
    self,
    batch_size: int = 1,
    num_inference_steps: Optional[int] = None,
    seed: Optional[int] = None,
    return_intermediates: bool = False
) -> Union[mx.array, Tuple[mx.array, List[mx.array]]]:
    ...
```

---

For more examples, see the [Usage Examples](examples.md) documentation.
