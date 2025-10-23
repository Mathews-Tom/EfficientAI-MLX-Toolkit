"""MLOps client module with unified interface for all MLOps operations."""

# Unified MLOps client
from mlops.client.mlops_client import (
    MLOpsClient,
    MLOpsClientError,
    create_client,
)

# Individual component clients (for advanced usage)
from mlops.client.mlflow_client import (
    MLFlowClient,
    MLFlowClientError,
    create_client as create_mlflow_client,
)
from mlops.client.dvc_client import (
    DVCClient,
    DVCClientError,
    create_client as create_dvc_client,
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
    # Unified client (recommended)
    "MLOpsClient",
    "MLOpsClientError",
    "create_client",
    # Individual clients
    "MLFlowClient",
    "MLFlowClientError",
    "create_mlflow_client",
    "DVCClient",
    "DVCClientError",
    "create_dvc_client",
    # BentoML integration
    "package_model_with_bentoml",
    "ModelFramework",
    "BENTOML_AVAILABLE",
]
