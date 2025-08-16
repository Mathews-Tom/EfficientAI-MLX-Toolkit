"""
Documentation generator for DSPy Integration Framework.
"""

import inspect
import logging
from datetime import datetime
from pathlib import Path

import dspy

from ..exceptions import DSPyIntegrationError
from ..framework import DSPyFramework
from ..signatures import (
    ClientUpdateSignature,
    CLIPDomainAdaptationSignature,
    ContrastiveLossSignature,
    DiffusionOptimizationSignature,
    FederatedLearningSignature,
    LoRAOptimizationSignature,
    LoRATrainingSignature,
    SamplingScheduleSignature,
)

logger = logging.getLogger(__name__)


class DocumentationGenerator:
    """Automatic documentation generator for DSPy signatures and modules."""

    def __init__(self, output_dir: Path | None = None):
        """Initialize documentation generator."""
        self.output_dir = output_dir or Path("docs/dspy_toolkit")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Documentation templates
        self.templates = {
            "signature": self._get_signature_template(),
            "module": self._get_module_template(),
            "project": self._get_project_template(),
            "api": self._get_api_template(),
        }

        logger.info(
            "Documentation generator initialized with output dir: %s",
            self.output_dir,
        )

    def generate_signature_docs(
        self, signature_class: type[dspy.Signature]
    ) -> dict[str, str | int | float | bool]:
        """Generate documentation for a DSPy signature."""
        try:
            # Extract signature information
            signature_info = {
                "name": signature_class.__name__,
                "module": signature_class.__module__,
                "docstring": inspect.getdoc(signature_class) or "No description available",
                "fields": {},
                "examples": [],
                "created_at": datetime.now().isoformat(),
            }

            # Extract field information
            if hasattr(signature_class, "__annotations__"):
                for field_name, field_type in signature_class.__annotations__.items():
                    field_obj = getattr(signature_class, field_name, None)

                    field_info = {
                        "name": field_name,
                        "type": str(field_type),
                        "description": "",
                        "required": True,
                        "field_type": "unknown",
                    }

                    if isinstance(field_obj, dspy.InputField):
                        field_info["field_type"] = "input"
                        field_info["description"] = getattr(field_obj, "desc", "")
                    elif isinstance(field_obj, dspy.OutputField):
                        field_info["field_type"] = "output"
                        field_info["description"] = getattr(field_obj, "desc", "")

                    signature_info["fields"][field_name] = field_info

            # Generate usage examples
            signature_info["examples"] = self._generate_signature_examples(signature_class)

            return signature_info

        except Exception as e:
            logger.error(
                "Failed to generate signature docs for %s: %s",
                signature_class,
                e,
            )
            return {"error": str(e)}

    def generate_module_docs(
        self, module: dspy.Module, module_name: str
    ) -> dict[str, str | int | float | bool]:
        """Generate documentation for a DSPy module."""
        try:
            module_info = {
                "name": module_name,
                "type": type(module).__name__,
                "docstring": inspect.getdoc(module) or "No description available",
                "methods": {},
                "attributes": {},
                "signature": None,
                "examples": [],
                "created_at": datetime.now().isoformat(),
            }

            # Extract methods
            for method_name in dir(module):
                if not method_name.startswith("_"):
                    method = getattr(module, method_name)
                    if callable(method):
                        method_info = {
                            "name": method_name,
                            "docstring": inspect.getdoc(method) or "No description",
                            "signature": (
                                str(inspect.signature(method))
                                if hasattr(inspect, "signature")
                                else "Unknown"
                            ),
                        }
                        module_info["methods"][method_name] = method_info

            # Extract attributes
            for attr_name in dir(module):
                if not attr_name.startswith("_") and not callable(getattr(module, attr_name)):
                    attr_value = getattr(module, attr_name)
                    module_info["attributes"][attr_name] = {
                        "name": attr_name,
                        "type": type(attr_value).__name__,
                        "value": (
                            str(attr_value)[:100] + "..."
                            if len(str(attr_value)) > 100
                            else str(attr_value)
                        ),
                    }

            # Extract signature if available
            if hasattr(module, "signature"):
                signature_class = module.signature
                if inspect.isclass(signature_class) and issubclass(signature_class, dspy.Signature):
                    module_info["signature"] = self.generate_signature_docs(signature_class)

            # Generate usage examples
            module_info["examples"] = self._generate_module_examples(module, module_name)

            return module_info

        except Exception as e:
            logger.error(
                "Failed to generate module docs for %s: %s",
                module_name,
                e,
            )
            return {"error": str(e)}

    def generate_project_docs(
        self, framework: DSPyFramework, project_name: str
    ) -> dict[str, str | int | float | bool]:
        """Generate documentation for a project."""
        try:
            project_info = {
                "name": project_name,
                "description": f"Documentation for {project_name} project",
                "signatures": {},
                "modules": {},
                "configuration": {},
                "examples": [],
                "created_at": datetime.now().isoformat(),
            }

            # Get project signatures
            try:
                project_signatures = framework.signature_registry.get_project_signatures(
                    project_name
                )
                for sig_name, sig_class in project_signatures.items():
                    project_info["signatures"][sig_name] = self.generate_signature_docs(sig_class)
            except Exception as e:
                logger.warning(
                    "Failed to get signatures for project %s: %s",
                    project_name,
                    e,
                )

            # Get project modules
            try:
                all_modules = framework.module_manager.list_modules()
                project_modules = [m for m in all_modules if m.startswith(f"{project_name}_")]

                for module_name in project_modules:
                    module = framework.module_manager.get_module(module_name)
                    if module:
                        project_info["modules"][module_name] = self.generate_module_docs(
                            module, module_name
                        )
            except Exception as e:
                logger.warning(
                    "Failed to get modules for project %s: %s",
                    project_name,
                    e,
                )

            # Get project configuration
            try:
                from .config_manager import ConfigManager

                config_manager = ConfigManager()
                project_config = config_manager.load_project_config(project_name)
                project_info["configuration"] = project_config
            except Exception as e:
                logger.warning(
                    "Failed to get configuration for project %s: %s",
                    project_name,
                    e,
                )

            # Generate project examples
            project_info["examples"] = self._generate_project_examples(project_name, project_info)

            return project_info

        except Exception as e:
            logger.error("Failed to generate project docs for %s: %s", project_name, e)
            return {"error": str(e)}

    def generate_api_docs(self, framework: DSPyFramework) -> dict[str, str | int | float | bool]:
        """Generate API documentation for the framework."""
        try:
            api_info = {
                "title": "DSPy Integration Framework API",
                "version": "1.0.0",
                "description": "API documentation for DSPy Integration Framework",
                "endpoints": {},
                "schemas": {},
                "examples": [],
                "created_at": datetime.now().isoformat(),
            }

            # Document FastAPI endpoints
            try:

                # This is a simplified approach - in practice, you'd introspect the actual FastAPI app
                endpoints = {
                    "/health": {
                        "method": "GET",
                        "description": "Health check endpoint",
                        "response": "Health status information",
                    },
                    "/predict": {
                        "method": "POST",
                        "description": "General prediction endpoint",
                        "request": "DSPyRequest with inputs and module information",
                        "response": "DSPyResponse with outputs and metadata",
                    },
                    "/predict/{project_name}": {
                        "method": "POST",
                        "description": "Project-specific prediction endpoint",
                        "parameters": {"project_name": "Name of the project"},
                        "request": "DSPyRequest with inputs",
                        "response": "DSPyResponse with outputs and metadata",
                    },
                    "/modules": {
                        "method": "GET",
                        "description": "List available modules",
                        "response": "List of module names",
                    },
                    "/signatures/{project_name}": {
                        "method": "GET",
                        "description": "List signatures for a project",
                        "parameters": {"project_name": "Name of the project"},
                        "response": "List of signature names",
                    },
                    "/stats": {
                        "method": "GET",
                        "description": "Get framework statistics",
                        "response": "Framework statistics and metrics",
                    },
                }

                api_info["endpoints"] = endpoints

            except Exception as e:
                logger.warning(
                    "Failed to generate API endpoint docs: %s",
                    e,
                )

            # Document data schemas
            schemas = {
                "DSPyRequest": {
                    "type": "object",
                    "properties": {
                        "inputs": {
                            "type": "object",
                            "description": "Input parameters for DSPy module",
                        },
                        "module_name": {
                            "type": "string",
                            "description": "Specific module name to use",
                        },
                        "project_name": {
                            "type": "string",
                            "description": "Project name for module lookup",
                        },
                        "options": {
                            "type": "object",
                            "description": "Additional options",
                        },
                    },
                    "required": ["inputs"],
                },
                "DSPyResponse": {
                    "type": "object",
                    "properties": {
                        "outputs": {
                            "type": "object",
                            "description": "Output from DSPy module",
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Response metadata",
                        },
                        "performance": {
                            "type": "object",
                            "description": "Performance metrics",
                        },
                        "timestamp": {
                            "type": "number",
                            "description": "Response timestamp",
                        },
                    },
                    "required": ["outputs"],
                },
            }

            api_info["schemas"] = schemas

            # Generate API examples
            api_info["examples"] = self._generate_api_examples()

            return api_info

        except Exception as e:
            logger.error(
                "Failed to generate API docs: %s",
                e,
            )
            return {"error": str(e)}

    def generate_all_docs(self, framework: DSPyFramework) -> dict[str, str]:
        """Generate all documentation and save to files."""
        try:
            generated_files = {}

            # Generate API documentation
            api_docs = self.generate_api_docs(framework)
            api_file = self.output_dir / "api.md"
            self._write_markdown_file(api_file, self._format_api_docs(api_docs))
            generated_files["api"] = str(api_file)

            # Generate project documentation
            try:
                all_signatures = framework.signature_registry.get_all_signatures()
                projects = list(all_signatures.keys())

                for project_name in projects:
                    project_docs = self.generate_project_docs(framework, project_name)
                    project_file = self.output_dir / f"project_{project_name}.md"
                    self._write_markdown_file(project_file, self._format_project_docs(project_docs))
                    generated_files[f"project_{project_name}"] = str(project_file)

            except Exception as e:
                logger.warning(
                    "Failed to generate project docs: %s",
                    e,
                )

            # Generate signature documentation
            try:
                # Import all signature classes
                signature_classes = [
                    LoRAOptimizationSignature,
                    LoRATrainingSignature,
                    DiffusionOptimizationSignature,
                    SamplingScheduleSignature,
                    CLIPDomainAdaptationSignature,
                    ContrastiveLossSignature,
                    FederatedLearningSignature,
                    ClientUpdateSignature,
                ]

                signatures_docs = {}
                for sig_class in signature_classes:
                    sig_docs = self.generate_signature_docs(sig_class)
                    signatures_docs[sig_class.__name__] = sig_docs

                signatures_file = self.output_dir / "signatures.md"
                self._write_markdown_file(
                    signatures_file, self._format_signatures_docs(signatures_docs)
                )
                generated_files["signatures"] = str(signatures_file)

            except Exception as e:
                logger.warning(
                    "Failed to generate signature docs: %s",
                    e,
                )

            # Generate index file
            index_content = self._generate_index_content(generated_files)
            index_file = self.output_dir / "index.md"
            self._write_markdown_file(index_file, index_content)
            generated_files["index"] = str(index_file)

            logger.info(
                "Generated %d documentation files",
                len(generated_files),
            )
            return generated_files

        except Exception as e:
            logger.error(
                "Failed to generate all docs: %s",
                e,
            )
            raise DSPyIntegrationError(f"Documentation generation failed: {e}") from e

    def _generate_signature_examples(
        self, signature_class: type[dspy.Signature]
    ) -> list[dict[str, str | int | float | bool]]:
        """Generate usage examples for a signature."""
        examples = []

        try:
            # Generate basic usage example
            example = {
                "title": f"Basic usage of {signature_class.__name__}",
                "description": f"Example showing how to use {signature_class.__name__}",
                "code": f"""
import dspy
from dspy_toolkit.signatures import {signature_class.__name__}

# Create a module using the signature
module = dspy.ChainOfThought({signature_class.__name__})

# Example usage (replace with actual input fields)
result = module(input_field="example input")
print(result.output_field)
""",
                "language": "python",
            }
            examples.append(example)

        except Exception as e:
            logger.warning(
                "Failed to generate examples for %s: %s",
                signature_class.__name__,
                e,
            )

        return examples

    def _generate_module_examples(
        self, module: dspy.Module, module_name: str
    ) -> list[dict[str, str | int | float | bool]]:
        """Generate usage examples for a module."""
        examples = []

        try:
            example = {
                "title": f"Using {module_name}",
                "description": f"Example showing how to use the {module_name} module",
                "code": f"""
# Assuming the module is already loaded
result = {module_name}(input_parameter="example")
print(result)
""",
                "language": "python",
            }
            examples.append(example)

        except Exception as e:
            logger.warning(
                "Failed to generate examples for %s: %s",
                module_name,
                e,
            )

        return examples

    def _generate_project_examples(
        self, project_name: str, project_info: dict[str, str | int | float | bool]
    ) -> list[dict[str, str | int | float | bool]]:
        """Generate usage examples for a project."""
        examples = []

        try:
            example = {
                "title": f"Getting started with {project_name}",
                "description": f"Basic example for {project_name} project",
                "code": f"""
from dspy_toolkit import DSPyFramework, DSPyConfig

# Initialize framework
config = DSPyConfig(model_provider="mlx", model_name="mlx/mlx-7b")
framework = DSPyFramework(config)

# Register project signatures (if any)
# framework.register_project_signatures("{project_name}", signatures)

# Create and use modules
# module = framework.create_project_module("{project_name}", "module_name", "signature_name")
# result = module(input="example")
""",
                "language": "python",
            }
            examples.append(example)

        except Exception as e:
            logger.warning(
                "Failed to generate examples for %s: %s",
                project_name,
                e,
            )

        return examples

    def _generate_api_examples(self) -> list[dict[str, str | int | float | bool]]:
        """Generate API usage examples."""
        examples = [
            {
                "title": "Health Check",
                "description": "Check the health of the DSPy framework",
                "code": """
curl -X GET "http://localhost:8000/health"
""",
                "language": "bash",
            },
            {
                "title": "Make a Prediction",
                "description": "Send a prediction request to the API",
                "code": """
curl -X POST "http://localhost:8000/predict/my_project/my_module" \\
  -H "Content-Type: application/json" \\
  -d '{
    "inputs": {
      "input_field": "example input"
    },
    "options": {}
  }'
""",
                "language": "bash",
            },
            {
                "title": "Python Client Example",
                "description": "Using the API from Python",
                "code": """
import requests

# Make a prediction request
response = requests.post(
    "http://localhost:8000/predict/my_project/my_module",
    json={
        "inputs": {"input_field": "example input"},
        "options": {}
    }
)

result = response.json()
print(result["outputs"])
""",
                "language": "python",
            },
        ]

        return examples

    def _write_markdown_file(self, file_path: Path, content: str):
        """Write content to a markdown file."""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            logger.error(
                "Failed to write markdown file %s: %s",
                file_path,
                e,
            )
            raise DSPyIntegrationError(f"Markdown file writing failed: {e}") from e

    def _format_api_docs(self, api_docs: dict[str, str | int | float | bool]) -> str:
        """Format API documentation as markdown."""
        content = f"""# {api_docs['title']}

{api_docs['description']}

**Version:** {api_docs['version']}  
**Generated:** {api_docs['created_at']}

## Endpoints

"""

        for endpoint, info in api_docs.get("endpoints", {}).items():
            content += f"### {info['method']} {endpoint}\n\n"
            content += f"{info['description']}\n\n"

            if "parameters" in info:
                content += "**Parameters:**\n"
                for param, desc in info["parameters"].items():
                    content += f"- `{param}`: {desc}\n"
                content += "\n"

            if "request" in info:
                content += f"**Request:** {info['request']}\n\n"

            if "response" in info:
                content += f"**Response:** {info['response']}\n\n"

        # Add schemas
        content += "## Data Schemas\n\n"
        for schema_name, schema_info in api_docs.get("schemas", {}).items():
            content += f"### {schema_name}\n\n"
            content += f"**Type:** {schema_info['type']}\n\n"

            if "properties" in schema_info:
                content += "**Properties:**\n"
                for prop, prop_info in schema_info["properties"].items():
                    content += f"- `{prop}` ({prop_info['type']}): {prop_info['description']}\n"
                content += "\n"

            if "required" in schema_info:
                content += f"**Required:** {', '.join(schema_info['required'])}\n\n"

        # Add examples
        content += "## Examples\n\n"
        for example in api_docs.get("examples", []):
            content += f"### {example['title']}\n\n"
            content += f"{example['description']}\n\n"
            content += f"```{example['language']}\n{example['code']}\n```\n\n"

        return content

    def _format_project_docs(self, project_docs: dict[str, str | int | float | bool]) -> str:
        """Format project documentation as markdown."""
        content = f"""# {project_docs['name']} Project

{project_docs['description']}

**Generated:** {project_docs['created_at']}

## Signatures

"""

        for sig_name, sig_info in project_docs.get("signatures", {}).items():
            content += f"### {sig_name}\n\n"
            content += f"{sig_info.get('docstring', 'No description')}\n\n"

            if "fields" in sig_info:
                content += "**Fields:**\n"
                for field_name, field_info in sig_info["fields"].items():
                    field_type = field_info["field_type"]
                    desc = field_info["description"]
                    content += f"- `{field_name}` ({field_type}): {desc}\n"
                content += "\n"

        # Add modules
        content += "## Modules\n\n"
        for module_name, module_info in project_docs.get("modules", {}).items():
            content += f"### {module_name}\n\n"
            content += f"**Type:** {module_info['type']}\n\n"
            content += f"{module_info.get('docstring', 'No description')}\n\n"

        # Add examples
        content += "## Examples\n\n"
        for example in project_docs.get("examples", []):
            content += f"### {example['title']}\n\n"
            content += f"{example['description']}\n\n"
            content += f"```{example['language']}\n{example['code']}\n```\n\n"

        return content

    def _format_signatures_docs(self, signatures_docs: dict[str, str | int | float | bool]) -> str:
        """Format signatures documentation as markdown."""
        content = """# DSPy Signatures Reference

This document provides detailed information about all available DSPy signatures in the integration framework.

"""

        for sig_name, sig_info in signatures_docs.items():
            content += f"## {sig_name}\n\n"
            content += f"{sig_info.get('docstring', 'No description')}\n\n"

            if "fields" in sig_info:
                content += "### Fields\n\n"
                for field_name, field_info in sig_info["fields"].items():
                    field_type = field_info["field_type"]
                    desc = field_info["description"]
                    content += f"- **`{field_name}`** ({field_type}): {desc}\n"
                content += "\n"

            # Add examples
            if "examples" in sig_info:
                content += "### Examples\n\n"
                for example in sig_info["examples"]:
                    content += f"#### {example['title']}\n\n"
                    content += f"{example['description']}\n\n"
                    content += f"```{example['language']}\n{example['code']}\n```\n\n"

        return content

    def _generate_index_content(self, generated_files: dict[str, str]) -> str:
        """Generate index page content."""
        content = """# DSPy Integration Framework Documentation

Welcome to the DSPy Integration Framework documentation. This framework provides intelligent prompt optimization and workflow automation for Apple Silicon-optimized AI projects.

## Documentation Sections

"""

        for doc_type, file_path in generated_files.items():
            if doc_type != "index":
                file_name = Path(file_path).name
                content += f"- [{doc_type.replace('_', ' ').title()}]({file_name})\n"

        content += f"""

## Getting Started

1. Install the framework: `pip install dspy-integration-framework`
2. Initialize the framework with your configuration
3. Register your project signatures
4. Create and optimize DSPy modules
5. Deploy with FastAPI integration

## Key Features

- **Apple Silicon Optimization**: Native MLX integration for 3-5x performance improvement
- **Intelligent Automation**: Automated prompt optimization and workflow management
- **Production Ready**: FastAPI integration with monitoring and observability
- **Error Recovery**: Circuit breakers, retry handlers, and fallback mechanisms
- **Centralized Management**: Configuration and program sharing across projects

**Generated:** {datetime.now().isoformat()}
"""

        return content

    def _get_signature_template(self) -> str:
        """Get signature documentation template."""
        return """
# {name}

{description}

## Fields

{fields}

## Examples

{examples}
"""

    def _get_module_template(self) -> str:
        """Get module documentation template."""
        return """
# {name}

**Type:** {type}

{description}

## Methods

{methods}

## Examples

{examples}
"""

    def _get_project_template(self) -> str:
        """Get project documentation template."""
        return """
# {name} Project

{description}

## Signatures

{signatures}

## Modules

{modules}

## Configuration

{configuration}

## Examples

{examples}
"""

    def _get_api_template(self) -> str:
        """Get API documentation template."""
        return """
# API Documentation

{description}

## Endpoints

{endpoints}

## Schemas

{schemas}

## Examples

{examples}
"""
