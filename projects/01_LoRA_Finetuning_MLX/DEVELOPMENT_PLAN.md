# LoRA MLX Implementation Development Plan

**Status:** Converting from Demo Mode to Full Implementation
**Created:** 2025-08-19
**Framework:** MLX-Native LoRA Fine-Tuning for Apple Silicon

## 🎯 Project Goal

Transform the LoRA framework from "demo mode" (parameter validation only) to full ML implementation with real training, inference, and serving capabilities.

## 📊 Current Implementation Status

### ✅ **Completed Components (80% of framework)**

- [x] **Architecture & Configuration**: Complete YAML config system with validation
- [x] **CLI Interface**: Full command structure with typer integration
- [x] **Project Structure**: Modular design with proper imports and dependencies
- [x] **MLX Dependencies**: Core MLX framework integration (`mlx>=0.15.0`, `mlx-lm>=0.15.0`)
- [x] **Training Framework**: `LoRATrainer` class structure and callback system
- [x] **Inference Engine**: `LoRAInferenceEngine` class with result containers
- [x] **FastAPI Serving**: Server setup and endpoint definitions
- [x] **Hyperparameter Optimization**: Optuna integration framework
- [x] **Sample Data**: 10 conversation examples in JSONL format
- [x] **Testing Infrastructure**: pytest setup with Apple Silicon markers

### ❌ **Missing Core ML Operations (20% remaining)**

- [ ] **Model Loading**: Actual MLX model instantiation and tokenizer integration
- [ ] **Training Loop**: Real gradient computation and parameter updates
- [ ] **Text Generation**: MLX-based inference with sampling methods
- [ ] **LoRA Adapter Integration**: Adapter loading/saving/merging
- [ ] **Dataset Processing**: JSONL data loading and tokenization

## 🚀 Development Phases

### **Phase 1: Dependencies & Model Loading**

**Estimated Time:** 6-8 hours
**Priority:** Critical

#### 1.1 Add Missing Dependencies

```bash
# Add to pyproject.toml
# Note: mlx-lm already includes tokenizer functionality
# Only add transformers if mlx-lm tokenizers prove insufficient
uv add transformers>=4.35.0  # Only if needed for specific tokenizer features
uv add datasets>=2.14.0
# Avoid torch dependency - mlx-lm provides compatible tokenizers
# uv add torch>=2.0.0  # Heavy dependency - use only if absolutely required
```

#### 1.2 Implement Model Loading (`src/inference/engine.py`)

- [ ] Replace `NotImplementedError` in `load_model_and_tokenizer()`
- [ ] Use mlx_lm.utils.load() for HuggingFace model conversion to MLX format
- [ ] Implement filesystem caching for converted MLX models to avoid re-conversion
- [ ] Integrate mlx-lm native tokenizers (avoid transformers dependency when possible)
- [ ] Add model validation and compatibility checks

**Files to modify:**

- `src/inference/engine.py:454` - Remove NotImplementedError
- `src/lora/adapters.py` - Add model loading utilities

#### 1.3 Create Dataset Loader (`src/training/data_loader.py`)

- [ ] JSONL conversation parser
- [ ] Tokenization with chat templates
- [ ] MLX-compatible batch creation
- [ ] Data validation and preprocessing

### **Phase 2: Training Implementation**

**Estimated Time:** 8-10 hours
**Priority:** Critical

#### 2.1 Complete Training Loop (`src/training/trainer.py`)

- [ ] Implement `LoRATrainer.train()` method (currently placeholder)
- [ ] Add forward pass with loss computation
- [ ] Use mlx.core.grad() to create gradient functions for loss computation
- [ ] Implement parameter updates using mlx.optimizers (Adam, AdamW, SGD)
- [ ] Add gradient clipping and accumulation for stable training
- [ ] Add progress tracking and logging

#### 2.2 LoRA Layer Integration (`src/lora/layers.py`)

- [ ] Complete `merge_weights()` and `unmerge_weights()` methods
- [ ] Add adapter saving/loading functionality
- [ ] Use efficient mlx.core vectorized operations for LoRA weight merging
- [ ] Implement memory-efficient adapter parameter updates using mlx tensors
- [ ] Avoid Python loops in favor of mlx.core broadcast and einsum operations

#### 2.3 Fix CLI Training Command (`src/cli.py`)

- [ ] Replace demo message at line 80: `"⚠️ Model loading and training not implemented in this demo"`
- [ ] Add actual trainer instantiation and execution
- [ ] Implement progress reporting and error handling

### **Phase 3: Inference & Generation**

**Estimated Time:** 6-8 hours
**Priority:** High

#### 3.1 Core Text Generation (`src/inference/engine.py`)

- [ ] Implement basic `generate()` method with single-prompt, non-streaming generation
- [ ] Add fundamental sampling (temperature, top-p, top-k) using mlx operations
- [ ] Validate end-to-end generation pipeline works correctly

#### 3.2 Advanced Generation Features (Follow-up)

- [ ] Add streaming generation support for real-time output
- [ ] Implement memory-efficient batching for multiple prompts
- [ ] Optimize generation performance with MLX-specific optimizations

#### 3.3 CLI Generation Command (`src/cli.py`)

- [ ] Replace demo message at line 171: `"⚠️ Text generation not implemented in this demo"`
- [ ] Add actual model loading and text generation
- [ ] Implement output formatting and error handling

#### 3.4 FastAPI Serving (`src/inference/serving.py`)

- [ ] Replace demo message at line 175: `"Model loading completed (placeholder)"`
- [ ] Implement actual model loading in server startup
- [ ] Add request/response processing with real inference

### **Phase 4: Integration & Testing**

**Estimated Time:** 8-12 hours
**Priority:** Medium

#### 4.1 End-to-End Pipeline Testing

- [ ] Test full training pipeline with sample data
- [ ] Validate model saving and loading
- [ ] Verify inference server functionality
- [ ] Performance benchmarking on Apple Silicon

#### 4.2 Error Handling & Edge Cases

- [ ] Robust error handling for missing models/data
- [ ] Memory management for large models
- [ ] Graceful degradation when MLX unavailable

## 📁 Key Files to Modify

### **Critical Implementation Files**

| File | Current Status | Required Changes |
|------|---------------|------------------|
| `src/cli.py` | Demo mode placeholders | Replace lines 80, 144, 171 with real implementations |
| `src/inference/engine.py` | NotImplementedError at line 454 | Implement model loading and generation |
| `src/training/trainer.py` | Framework only | Complete training loop implementation |
| `src/lora/layers.py` | Partial implementation | Complete LoRA operations |
| `src/inference/serving.py` | Demo placeholders | Real FastAPI model integration |

### **Configuration Files**

| File | Status | Action |
|------|--------|---------|
| `pyproject.toml` | Missing deps | Add transformers, datasets, torch |
| `configs/default.yaml` | Complete | No changes needed |

## 🧪 Testing Strategy

### **Phase Testing Approach**

1. **Phase 1**: Unit tests for model loading and data processing
2. **Phase 2**: Training loop validation with synthetic data
3. **Phase 3**: Inference testing with saved models
4. **Phase 4**: Integration tests with real conversation data

### **Validation Commands**

```bash
# Test configuration validation (already works)
uv run efficientai-toolkit projects lora-finetuning-mlx validate

# Test training pipeline (to be implemented)
uv run efficientai-toolkit projects lora-finetuning-mlx train --epochs 1 --batch-size 1

# Test text generation (to be implemented)
uv run efficientai-toolkit projects lora-finetuning-mlx generate --prompt "Hello" --model-path ./output

# Test serving (to be implemented)
uv run efficientai-toolkit projects lora-finetuning-mlx serve --model-path ./output
```

## 📈 Success Metrics

### **Functional Requirements**

- [ ] Successfully train LoRA model on sample conversations
- [ ] Generate coherent text responses using trained model
- [ ] Serve model via FastAPI with sub-second response times
- [ ] Pass all existing tests (currently 56/56 passing)
- [ ] Memory usage under 8GB during training

### **Performance Targets**

- **Training Speed**: >100 tokens/second on Apple Silicon
- **Inference Speed**: >50 tokens/second for generation
- **Memory Efficiency**: <4GB for small models (117M parameters)
- **Model Size**: LoRA adapters <50MB

## 🔄 Development Workflow

### **Branch Strategy**

```bash
# Create feature branch for implementation
git checkout -b feature/implement-core-ml-operations

# Implement in phases
git add -A && git commit -m "Phase 1: Add dependencies and model loading"
git add -A && git commit -m "Phase 2: Implement training loop"
git add -A && git commit -m "Phase 3: Add inference and generation"
git add -A && git commit -m "Phase 4: Integration testing and cleanup"
```

### **Quality Gates**

- [ ] All existing tests continue to pass
- [ ] New functionality has comprehensive test coverage
- [ ] Code follows existing style (black, ruff, mypy)
- [ ] Documentation updated for new capabilities

## 📝 Progress Tracking

### **Phase 1 Progress** ⏳

- [ ] Dependencies added to pyproject.toml
- [ ] Model loading implemented
- [ ] Dataset loader created
- [ ] Unit tests passing

### **Phase 2 Progress** ⏳

- [ ] Training loop completed
- [ ] LoRA layers functional
- [ ] CLI training command working
- [ ] Training validation successful

### **Phase 3 Progress** ⏳

- [ ] Text generation implemented
- [ ] CLI generate command working
- [ ] FastAPI serving functional
- [ ] Inference tests passing

### **Phase 4 Progress** ⏳

- [ ] End-to-end pipeline working
- [ ] Performance benchmarks met
- [ ] Error handling robust
- [ ] Documentation complete

---

**Next Steps:** Begin Phase 1 by adding missing dependencies and implementing model loading functionality.

**Contact:** Update this document as development progresses to track completion status.
