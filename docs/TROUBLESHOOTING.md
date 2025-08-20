# Troubleshooting Guide

## Common Issues and Solutions

This guide provides solutions to the most frequently encountered issues when using the EfficientAI-MLX-Toolkit.

## Installation Issues

### MLX Framework Not Available

**Symptoms:**
```
ImportError: No module named 'mlx'
⚠️ MLX framework not detected. Falling back to CPU mode.
```

**Solutions:**

1. **Install MLX (Apple Silicon only):**
   ```bash
   uv add mlx
   # Or using pip
   pip install mlx
   ```

2. **Verify Apple Silicon:**
   ```bash
   uv run efficientai-toolkit info
   # Should show: ✅ Apple Silicon detected
   ```

3. **Check Python version:**
   ```bash
   python --version
   # MLX requires Python 3.8+, recommended 3.12+
   ```

4. **CPU fallback mode:**
   ```bash
   # Use CPU mode if MLX unavailable
   export EFFICIENTAI_FORCE_CPU=1
   uv run efficientai-toolkit projects lora-finetuning-mlx train
   ```

### Dependencies Installation Failed

**Symptoms:**
```
ERROR: Could not build wheels for [package]
uv sync failed with exit code 1
```

**Solutions:**

1. **Update uv:**
   ```bash
   pip install --upgrade uv
   ```

2. **Clear cache:**
   ```bash
   uv cache clean
   uv sync --reinstall
   ```

3. **Install system dependencies (macOS):**
   ```bash
   # Install Xcode command line tools
   xcode-select --install
   
   # Install Homebrew if needed
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   
   # Install required packages
   brew install cmake pkg-config
   ```

4. **Use pip fallback:**
   ```bash
   pip install -e .
   ```

### Permission Errors

**Symptoms:**
```
PermissionError: [Errno 13] Permission denied
```

**Solutions:**

1. **Check file permissions:**
   ```bash
   ls -la configs/
   chmod 644 configs/default.yaml
   ```

2. **Use user installation:**
   ```bash
   pip install --user -e .
   ```

3. **Check directory ownership:**
   ```bash
   sudo chown -R $USER:$USER .
   ```

## Configuration Issues

### Configuration File Not Found

**Symptoms:**
```
❌ Configuration file not found: configs/default.yaml
FileNotFoundError: [Errno 2] No such file or directory
```

**Solutions:**

1. **Check current directory:**
   ```bash
   pwd
   # Should be in: /path/to/EfficientAI-MLX-Toolkit
   cd /path/to/EfficientAI-MLX-Toolkit
   ```

2. **Use absolute paths:**
   ```bash
   uv run efficientai-toolkit projects lora-finetuning-mlx validate \
     --config /absolute/path/to/config.yaml
   ```

3. **Verify file exists:**
   ```bash
   ls -la projects/01_LoRA_Finetuning_MLX/configs/
   ```

4. **Generate default config:**
   ```bash
   uv run efficientai-toolkit projects lora-finetuning-mlx generate-config \
     --output configs/my_config.yaml
   ```

### YAML Parsing Errors

**Symptoms:**
```
yaml.parser.ParserError: expected <block end>, but found '<scalar>'
Error: Invalid YAML syntax
```

**Solutions:**

1. **Check indentation (use spaces, not tabs):**
   ```yaml
   # Correct
   lora:
     rank: 16
     alpha: 32
   
   # Incorrect (tabs)
   lora:
   	rank: 16
   	alpha: 32
   ```

2. **Verify quotation marks:**
   ```yaml
   # Correct
   training:
     model_name: "microsoft/DialoGPT-medium"
   
   # Incorrect (unmatched quotes)
   training:
     model_name: "microsoft/DialoGPT-medium
   ```

3. **Check list formatting:**
   ```yaml
   # Correct
   target_modules:
     - "q_proj"
     - "v_proj"
   
   # Incorrect
   target_modules: ["q_proj", "v_proj"  # Missing closing bracket
   ```

4. **Validate YAML online:**
   - Use online YAML validators to check syntax
   - Copy-paste your config to identify issues

### Parameter Validation Errors

**Symptoms:**
```
ValidationError: rank must be between 1 and 128
ValueError: learning_rate must be positive
```

**Solutions:**

1. **Check parameter bounds:**
   ```yaml
   lora:
     rank: 16        # Must be 1-128
     alpha: 32.0     # Must be positive float
     dropout: 0.1    # Must be 0.0-1.0
   
   training:
     learning_rate: 2e-4  # Must be 1e-6 to 1e-1
     batch_size: 2        # Must be positive integer
   ```

2. **Validate configuration:**
   ```bash
   uv run efficientai-toolkit projects lora-finetuning-mlx validate \
     --config configs/my_config.yaml
   ```

3. **Use default values:**
   ```bash
   # Start with working defaults
   cp projects/01_LoRA_Finetuning_MLX/configs/default.yaml my_config.yaml
   # Edit gradually
   ```

## Memory Issues

### Out of Memory Errors

**Symptoms:**
```
RuntimeError: MLX out of memory
CUDA out of memory. Tried to allocate 2.0 GB
MemoryError: Unable to allocate array
```

**Solutions:**

1. **Reduce batch size:**
   ```bash
   uv run efficientai-toolkit projects lora-finetuning-mlx train \
     --batch-size 1
   ```

2. **Lower LoRA rank:**
   ```bash
   uv run efficientai-toolkit projects lora-finetuning-mlx train \
     --rank 8 --alpha 16
   ```

3. **Set memory limits:**
   ```bash
   export MLX_MEMORY_LIMIT=8192  # 8GB limit
   uv run efficientai-toolkit projects lora-finetuning-mlx train
   ```

4. **Enable gradient accumulation:**
   ```yaml
   training:
     batch_size: 1
     gradient_accumulation_steps: 4  # Effective batch size: 4
   ```

5. **Monitor memory usage:**
   ```bash
   # macOS
   sudo powermetrics --samplers smc -n 1 | grep -i memory
   
   # Or use Activity Monitor
   # Or install htop: brew install htop
   ```

### Memory Leaks

**Symptoms:**
```
Memory usage keeps increasing during training
System becomes unresponsive
```

**Solutions:**

1. **Restart training periodically:**
   ```bash
   # Train in shorter segments
   uv run efficientai-toolkit projects lora-finetuning-mlx train \
     --epochs 2
   # Continue from checkpoint
   uv run efficientai-toolkit projects lora-finetuning-mlx train \
     --epochs 2 --resume-from-checkpoint outputs/checkpoint-1000
   ```

2. **Clear MLX cache:**
   ```python
   import mlx.core as mx
   mx.metal.clear_cache()
   ```

3. **Use memory profiling:**
   ```bash
   uv add memray
   uv run python -m memray run --live-remote train_script.py
   ```

## Training Issues

### Training Fails to Start

**Symptoms:**
```
RuntimeError: Model not found
ModuleNotFoundError: No module named 'transformers'
Error: Unable to load model
```

**Solutions:**

1. **Install missing dependencies:**
   ```bash
   uv sync --all-extras
   ```

2. **Check model availability:**
   ```python
   from transformers import AutoModel
   model = AutoModel.from_pretrained("microsoft/DialoGPT-medium")
   ```

3. **Use cache directory:**
   ```bash
   export HF_HOME=/path/to/huggingface/cache
   mkdir -p $HF_HOME
   ```

4. **Test model loading:**
   ```bash
   uv run python -c "
   from transformers import AutoTokenizer
   tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
   print('Model loading successful')
   "
   ```

### Slow Training Performance

**Symptoms:**
- Training takes much longer than expected
- Low GPU/MLX utilization
- System appears idle during training

**Solutions:**

1. **Enable MLX compilation:**
   ```yaml
   mlx:
     compile_model: true
     precision: "float16"
   ```

2. **Increase batch size:**
   ```bash
   uv run efficientai-toolkit projects lora-finetuning-mlx train \
     --batch-size 4  # If memory allows
   ```

3. **Check data loading:**
   ```yaml
   data:
     num_workers: 4       # Parallel data loading
     prefetch_factor: 2
   ```

4. **Profile training:**
   ```bash
   uv run python -m cProfile -o profile.prof train_script.py
   ```

5. **Monitor system resources:**
   ```bash
   # Check CPU usage
   top -pid $(pgrep -f python)
   
   # Check MLX utilization
   uv run python -c "
   import mlx.core as mx
   print('MLX Metal available:', mx.metal.is_available())
   print('Active memory:', mx.metal.get_active_memory() / 1024**2, 'MB')
   "
   ```

### NaN or Inf Loss Values

**Symptoms:**
```
Loss: nan
RuntimeError: Loss is NaN or Inf
Training unstable
```

**Solutions:**

1. **Reduce learning rate:**
   ```bash
   uv run efficientai-toolkit projects lora-finetuning-mlx train \
     --learning-rate 1e-5
   ```

2. **Enable gradient clipping:**
   ```yaml
   training:
     gradient_clip_norm: 1.0
     max_grad_norm: 1.0
   ```

3. **Add warmup steps:**
   ```yaml
   training:
     warmup_steps: 100
     scheduler: "linear"
   ```

4. **Check data quality:**
   ```python
   import json
   with open("data/samples/sample_conversations.jsonl") as f:
       for line in f:
           data = json.loads(line)
           if len(data.get("input", "")) == 0:
               print("Empty input found:", data)
   ```

## Inference Issues

### Generation Produces Poor Results

**Symptoms:**
- Generated text is repetitive
- Output is incoherent
- Generation stops early

**Solutions:**

1. **Adjust generation parameters:**
   ```bash
   uv run efficientai-toolkit projects lora-finetuning-mlx generate \
     --model-path outputs/model \
     --prompt "Hello" \
     --temperature 0.8 \
     --top-p 0.9 \
     --repetition-penalty 1.1
   ```

2. **Check model quality:**
   ```bash
   # Ensure training completed successfully
   ls -la outputs/lora-finetuned-model/
   ```

3. **Try different prompts:**
   ```bash
   # Test with various prompt styles
   uv run efficientai-toolkit projects lora-finetuning-mlx generate \
     --model-path outputs/model \
     --prompt "Human: Hello\nAssistant:" \
     --max-length 50
   ```

4. **Verify model compatibility:**
   ```python
   # Check if LoRA adapters match base model
   from src.inference.engine import InferenceEngine
   engine = InferenceEngine(model_path="outputs/model")
   print("Model loaded successfully")
   ```

### Serving API Errors

**Symptoms:**
```
Connection refused on port 8000
FastAPI server won't start
Internal server error 500
```

**Solutions:**

1. **Check port availability:**
   ```bash
   lsof -i :8000
   # Kill existing process if needed
   kill -9 <PID>
   ```

2. **Start server with debug:**
   ```bash
   uv run efficientai-toolkit projects lora-finetuning-mlx serve \
     --model-path outputs/model \
     --host 127.0.0.1 \
     --port 8001 \
     --debug
   ```

3. **Test API manually:**
   ```bash
   # Test health endpoint
   curl http://localhost:8000/health
   
   # Test generation
   curl -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Test", "max_length": 10}'
   ```

4. **Check server logs:**
   ```bash
   # Look for error messages in terminal output
   # Check system logs
   tail -f /var/log/system.log | grep python
   ```

## Testing Issues

### Tests Fail to Run

**Symptoms:**
```
ImportError during test collection
No tests found
pytest command not found
```

**Solutions:**

1. **Use unified CLI:**
   ```bash
   uv run efficientai-toolkit test lora-finetuning-mlx
   ```

2. **Check test discovery:**
   ```bash
   uv run pytest --collect-only tests/
   ```

3. **Install test dependencies:**
   ```bash
   uv add --group dev pytest pytest-cov pytest-asyncio
   ```

4. **Check test file naming:**
   ```bash
   # Test files must start with 'test_'
   ls tests/test_*.py
   ```

### Tests Pass Locally but Fail in CI

**Symptoms:**
- Tests work on local machine
- Fail in GitHub Actions or CI environment
- Hardware-specific failures

**Solutions:**

1. **Skip hardware-specific tests in CI:**
   ```python
   @pytest.mark.apple_silicon
   def test_mlx_feature():
       if not mx.metal.is_available():
           pytest.skip("Apple Silicon required")
   ```

2. **Mock external dependencies:**
   ```python
   @pytest.fixture
   def mock_mlx():
       with patch('mlx.core.metal.is_available', return_value=False):
           yield
   ```

3. **Set CI environment variables:**
   ```yaml
   # .github/workflows/test.yml
   env:
     EFFICIENTAI_TEST_MODE: 1
     MLX_MEMORY_LIMIT: 1024
   ```

## Performance Issues

### Optimization Runs Slowly

**Symptoms:**
- Hyperparameter optimization takes hours
- Each trial is very slow
- System becomes unresponsive

**Solutions:**

1. **Reduce search space:**
   ```bash
   uv run efficientai-toolkit projects lora-finetuning-mlx optimize \
     --model microsoft/DialoGPT-medium \
     --data /path/to/data.jsonl \
     --trials 5  # Instead of 20
   ```

2. **Use faster models:**
   ```bash
   # Use smaller model for optimization
   uv run efficientai-toolkit projects lora-finetuning-mlx optimize \
     --model microsoft/DialoGPT-small \
     --data /path/to/small_dataset.jsonl
   ```

3. **Limit training epochs in optimization:**
   ```yaml
   optimization:
     max_epochs: 2  # Quick evaluation
     early_stopping: true
   ```

4. **Parallel optimization:**
   ```bash
   # Run multiple optimization processes
   uv run efficientai-toolkit projects lora-finetuning-mlx optimize \
     --parallel-trials 2
   ```

## Debug Mode and Logging

### Enable Debug Mode

```bash
# Enable debug logging
export EFFICIENTAI_DEBUG=1

# Run with verbose output
uv run efficientai-toolkit projects lora-finetuning-mlx train --verbose

# Enable MLX debug mode
export MLX_DEBUG=1
```

### Custom Logging

```python
import logging
from utils import setup_logging

setup_logging(
    log_level="DEBUG",
    log_file=Path("debug.log"),
    enable_apple_silicon_tracking=True
)

logger = logging.getLogger(__name__)
logger.debug("Debug message")
```

## Getting Help

### Community Resources

1. **GitHub Issues**: Report bugs and request features
   - Include system information from `uv run efficientai-toolkit info`
   - Provide minimal reproduction example
   - Include error messages and stack traces

2. **Documentation**: Check comprehensive documentation
   - [CLI Reference](CLI_REFERENCE.md)
   - [Configuration Guide](CONFIGURATION.md)
   - [Testing Guide](TESTING.md)

3. **System Information**: Always include when reporting issues
   ```bash
   uv run efficientai-toolkit info > system_info.txt
   ```

### Creating Bug Reports

Include the following information:

1. **System details:**
   ```bash
   uv run efficientai-toolkit info
   python --version
   uname -a
   ```

2. **Error reproduction:**
   ```bash
   # Minimal command that reproduces the error
   uv run efficientai-toolkit projects lora-finetuning-mlx train --epochs 1
   ```

3. **Configuration files:**
   ```bash
   # Share relevant config (remove sensitive data)
   cat configs/default.yaml
   ```

4. **Full error messages:**
   ```bash
   # Include complete stack trace
   EFFICIENTAI_DEBUG=1 uv run [command] 2>&1 | tee error.log
   ```

---

Most issues can be resolved by following this guide. For persistent problems, don't hesitate to open a GitHub issue with detailed information.