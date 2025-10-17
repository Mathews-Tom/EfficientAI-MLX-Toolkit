"""Byzantine fault tolerance for federated aggregation."""

from __future__ import annotations

import logging

import mlx.core as mx

from federated.types import ModelUpdate

logger = logging.getLogger(__name__)


class ByzantineTolerantAggregator:
    """Byzantine-tolerant aggregation using robust statistics.

    Implements median-based and trimmed mean aggregation to handle
    malicious or faulty clients.
    """

    def __init__(
        self,
        strategy: str = "trimmed_mean",
        trim_ratio: float = 0.1,
    ):
        """Initialize Byzantine-tolerant aggregator.

        Args:
            strategy: Aggregation strategy (median, trimmed_mean)
            trim_ratio: Ratio of extreme values to trim
        """
        self.strategy = strategy
        self.trim_ratio = trim_ratio

    def aggregate(
        self, client_updates: list[ModelUpdate]
    ) -> dict[str, mx.array]:
        """Aggregate using Byzantine-tolerant strategy.

        Args:
            client_updates: List of client updates

        Returns:
            Aggregated parameters
        """
        if not client_updates:
            raise ValueError("No client updates provided")

        if self.strategy == "median":
            return self._aggregate_median(client_updates)
        elif self.strategy == "trimmed_mean":
            return self._aggregate_trimmed_mean(client_updates)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _aggregate_median(
        self, client_updates: list[ModelUpdate]
    ) -> dict[str, mx.array]:
        """Aggregate using coordinate-wise median.

        Args:
            client_updates: Client updates

        Returns:
            Median-aggregated parameters
        """
        param_names = list(client_updates[0].parameters.keys())
        aggregated = {}

        for name in param_names:
            # Stack parameters from all clients
            stacked = mx.stack(
                [update.parameters[name] for update in client_updates]
            )

            # Compute median
            aggregated[name] = mx.median(stacked, axis=0)

        logger.info(
            f"Aggregated {len(client_updates)} updates using median"
        )
        return aggregated

    def _aggregate_trimmed_mean(
        self, client_updates: list[ModelUpdate]
    ) -> dict[str, mx.array]:
        """Aggregate using trimmed mean (remove outliers).

        Args:
            client_updates: Client updates

        Returns:
            Trimmed mean aggregated parameters
        """
        param_names = list(client_updates[0].parameters.keys())
        aggregated = {}

        num_trim = int(len(client_updates) * self.trim_ratio)

        for name in param_names:
            # Stack parameters
            stacked = mx.stack(
                [update.parameters[name] for update in client_updates]
            )

            # Sort along client dimension
            sorted_params = mx.sort(stacked, axis=0)

            # Trim extreme values
            if num_trim > 0:
                trimmed = sorted_params[num_trim:-num_trim]
            else:
                trimmed = sorted_params

            # Compute mean
            aggregated[name] = mx.mean(trimmed, axis=0)

        logger.info(
            f"Aggregated {len(client_updates)} updates using trimmed mean "
            f"(trimmed {num_trim} from each end)"
        )
        return aggregated

    def detect_byzantine_clients(
        self, client_updates: list[ModelUpdate], threshold: float = 2.0
    ) -> list[str]:
        """Detect potential Byzantine clients based on parameter divergence.

        Args:
            client_updates: Client updates
            threshold: Z-score threshold for detection

        Returns:
            List of suspected Byzantine client IDs
        """
        if len(client_updates) < 3:
            return []  # Need at least 3 clients

        # Compute pairwise distances
        suspected = []

        for i, update_i in enumerate(client_updates):
            distances = []

            for j, update_j in enumerate(client_updates):
                if i != j:
                    dist = self._compute_parameter_distance(
                        update_i.parameters, update_j.parameters
                    )
                    distances.append(dist)

            # Check if this client is an outlier
            mean_dist = sum(distances) / len(distances)
            std_dist = (
                sum((d - mean_dist) ** 2 for d in distances) / len(distances)
            ) ** 0.5

            if std_dist > 0:
                z_score = (max(distances) - mean_dist) / std_dist
                if z_score > threshold:
                    suspected.append(update_i.client_id)

        if suspected:
            logger.warning(
                f"Detected {len(suspected)} suspected Byzantine clients: "
                f"{suspected}"
            )

        return suspected

    def _compute_parameter_distance(
        self, params1: dict[str, mx.array], params2: dict[str, mx.array]
    ) -> float:
        """Compute L2 distance between parameter sets.

        Args:
            params1: First parameter set
            params2: Second parameter set

        Returns:
            L2 distance
        """
        total_distance = 0.0

        for name in params1.keys():
            diff = params1[name] - params2[name]
            total_distance += float(mx.sum(diff ** 2))

        return total_distance ** 0.5
