"""Meta-learning algorithms for PEFT."""

from .reptile import ReptileLearner
from .maml import MAMLLearner, FOMAMLLearner
from .meta_sgd import MetaSGDLearner
from .orchestrator import (
    MetaLearningOrchestrator,
    MetaTrainingConfig,
    quick_meta_train,
)
from .evaluation import (
    FewShotEvaluator,
    CrossTaskEvaluator,
    BaselineComparator,
    comprehensive_evaluation,
)

__all__ = [
    "ReptileLearner",
    "MAMLLearner",
    "FOMAMLLearner",
    "MetaSGDLearner",
    "MetaLearningOrchestrator",
    "MetaTrainingConfig",
    "quick_meta_train",
    "FewShotEvaluator",
    "CrossTaskEvaluator",
    "BaselineComparator",
    "comprehensive_evaluation",
]
