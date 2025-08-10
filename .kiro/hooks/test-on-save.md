# Test on Save Hook

## Description
Automatically run tests when Python files are saved to ensure code quality and catch regressions early.

## Trigger
File save events for Python files (*.py)

## Actions
1. Run unit tests for the modified file
2. Run integration tests if shared utilities are modified
3. Check Apple Silicon optimizations if hardware-specific code is changed
4. Report test results and performance metrics

## Configuration
```yaml
name: "Test on Save"
trigger:
  event: "file_save"
  pattern: "**/*.py"
conditions:
  - file_exists: "tests/"
  - not_in_path: "__pycache__"
actions:
  - run_command: "uv run pytest tests/test_${filename_without_ext}.py -v"
  - if_shared_utils_modified:
      run_command: "uv run pytest tests/integration/ -v"
  - if_apple_silicon_code:
      run_command: "uv run pytest tests/test_apple_silicon.py -v"
notifications:
  success: "✅ Tests passed for ${filename}"
  failure: "❌ Tests failed for ${filename}"
```

## Expected Behavior
- Fast feedback on code changes
- Automatic detection of Apple Silicon optimization issues
- Integration test execution when shared components are modified
- Clear notifications about test status