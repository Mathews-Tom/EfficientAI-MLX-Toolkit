"""
Exception classes for DSPy Integration Framework.
"""

# Standard library imports
import logging
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger(__name__)


class DSPyIntegrationError(Exception):
    """Base exception for DSPy integration issues."""


class OptimizerFailureError(DSPyIntegrationError):
    """Raised when DSPy optimization fails."""


class SignatureValidationError(DSPyIntegrationError):
    """Raised when signature validation fails."""


class MLXProviderError(DSPyIntegrationError):
    """Raised when MLX LLM provider encounters issues."""


class HardwareCompatibilityError(DSPyIntegrationError):
    """Raised when hardware compatibility issues occur."""


class ModuleRegistrationError(DSPyIntegrationError):
    """Raised when module registration fails."""


def handle_dspy_errors(fallback_func: Callable = None):
    """
    Decorator for graceful DSPy error handling.

    Args:
        fallback_func: Optional fallback function to call on error
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except OptimizerFailureError as e:
                logger.warning("Optimization failed: %s, using unoptimized module", e)
                if fallback_func:
                    return fallback_func(*args, **kwargs)
                return args[0] if args else None  # Return original module
            except MLXProviderError as e:
                logger.warning("MLX provider failed: %s, falling back to default", e)
                if fallback_func:
                    return fallback_func(*args, **kwargs)
                raise DSPyIntegrationError("MLX provider failed") from e
            except Exception as e:
                logger.error("Unexpected DSPy error: %s", e)
                raise DSPyIntegrationError("DSPy integration failed") from e

        return wrapper

    return decorator


def handle_async_dspy_errors(fallback_func: Callable = None):
    """
    Async decorator for graceful DSPy error handling.

    Args:
        fallback_func: Optional async fallback function to call on error
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            try:
                return await func(*args, **kwargs)
            except OptimizerFailureError as e:
                logger.warning("Async optimization failed: %s", e)
                if fallback_func:
                    return await fallback_func(*args, **kwargs)
                return args[0] if args else None
            except MLXProviderError as e:
                logger.warning("Async MLX provider failed: %s", e)
                if fallback_func:
                    return await fallback_func(*args, **kwargs)
                raise DSPyIntegrationError("Async MLX provider failed") from e
            except Exception as e:
                logger.error("Unexpected async DSPy error: %s", e)
                raise DSPyIntegrationError("Async DSPy integration failed") from e

        return wrapper

    return decorator
