"""Privacy budget tracking for federated learning."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PrivacyBudgetRecord:
    """Record of privacy budget consumption."""

    round_id: int
    epsilon: float
    delta: float
    cumulative_epsilon: float


class PrivacyBudgetTracker:
    """Tracks and manages privacy budget consumption."""

    def __init__(self, max_epsilon: float = 1.0, max_delta: float = 1e-5):
        """Initialize privacy budget tracker.

        Args:
            max_epsilon: Maximum allowed epsilon
            max_delta: Maximum allowed delta
        """
        self.max_epsilon = max_epsilon
        self.max_delta = max_delta
        self.records: list[PrivacyBudgetRecord] = []
        self.cumulative_epsilon = 0.0

    def record_round(
        self, round_id: int, epsilon: float, delta: float
    ) -> None:
        """Record privacy budget for a round.

        Args:
            round_id: Round identifier
            epsilon: Epsilon consumed this round
            delta: Delta consumed this round
        """
        self.cumulative_epsilon += epsilon

        record = PrivacyBudgetRecord(
            round_id=round_id,
            epsilon=epsilon,
            delta=delta,
            cumulative_epsilon=self.cumulative_epsilon,
        )

        self.records.append(record)

        logger.info(
            f"Round {round_id}: ε={epsilon:.4f}, "
            f"cumulative ε={self.cumulative_epsilon:.4f}"
        )

    def is_budget_exceeded(self) -> bool:
        """Check if privacy budget is exceeded.

        Returns:
            True if budget exceeded
        """
        return self.cumulative_epsilon >= self.max_epsilon

    def get_remaining_budget(self) -> float:
        """Get remaining privacy budget.

        Returns:
            Remaining epsilon
        """
        return max(0.0, self.max_epsilon - self.cumulative_epsilon)

    def get_budget_status(self) -> dict[str, Any]:
        """Get budget status.

        Returns:
            Dictionary containing budget information
        """
        return {
            "max_epsilon": self.max_epsilon,
            "cumulative_epsilon": self.cumulative_epsilon,
            "remaining_epsilon": self.get_remaining_budget(),
            "budget_exceeded": self.is_budget_exceeded(),
            "num_rounds": len(self.records),
        }
