"""MLOps client module with MLFlow and BentoML integration."""

from mlops.client.mlflow_client import (
    MLFlowClient,
    MLFlowClientError,
    create_client,
)

# BentoML integration (optional)
try:
    from mlops.serving.bentoml.packager import package_model as package_model_with_bentoml
    from mlops.serving.bentoml.config import ModelFramework
    BENTOML_AVAILABLE = True
except ImportError:
    BENTOML_AVAILABLE = False
    package_model_with_bentoml = None
    ModelFramework = None

__all__ = [
    "MLFlowClient",
    "MLFlowClientError",
    "create_client",
    "package_model_with_bentoml",
    "ModelFramework",
    "BENTOML_AVAILABLE",
]
