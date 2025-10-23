"""Data Drift Detection

Evidently-based data drift detection for monitoring input data distribution changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from mlops.monitoring.evidently.config import DriftDetectionConfig

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Data drift detection report

    Attributes:
        timestamp: When drift detection was performed
        dataset_drift: Whether dataset-level drift was detected
        drift_share: Share of features with detected drift
        number_of_drifted_features: Number of features with drift
        total_features: Total number of features analyzed
        drifted_features: List of features with detected drift
        drift_scores: Drift scores for each feature
        metrics: Additional drift metrics
    """

    timestamp: datetime
    dataset_drift: bool
    drift_share: float
    number_of_drifted_features: int
    total_features: int
    drifted_features: list[str] = field(default_factory=list)
    drift_scores: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary

        Returns:
            Dictionary representation of drift report
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "dataset_drift": self.dataset_drift,
            "drift_share": self.drift_share,
            "number_of_drifted_features": self.number_of_drifted_features,
            "total_features": self.total_features,
            "drifted_features": self.drifted_features,
            "drift_scores": self.drift_scores,
            "metrics": self.metrics,
        }


class DriftDetector:
    """Data drift detector using Evidently

    This detector monitors input data distribution changes by comparing
    current data against a reference dataset. It identifies:
    - Dataset-level drift
    - Feature-level drift
    - Drift magnitude and affected features
    """

    def __init__(
        self,
        project_name: str,
        config: DriftDetectionConfig | None = None,
        workspace_path: Path | str | None = None,
    ):
        """Initialize drift detector

        Args:
            project_name: Name of the project being monitored
            config: Drift detection configuration
            workspace_path: Path to workspace for saving reports
        """
        self.project_name = project_name
        self.config = config or DriftDetectionConfig()
        self.workspace_path = (
            Path(workspace_path) if workspace_path else Path("mlops/monitoring/workspace")
        )
        self.workspace_path.mkdir(parents=True, exist_ok=True)

        # Reference data storage
        self._reference_data: pd.DataFrame | None = None

        logger.info(
            "Initialized DriftDetector for project: %s (workspace: %s)",
            project_name,
            self.workspace_path,
        )

    def set_reference_data(self, reference_data: pd.DataFrame) -> None:
        """Set reference data for drift detection

        Args:
            reference_data: Reference dataset (training data or initial production data)

        Raises:
            ValueError: If reference data is empty
        """
        if reference_data.empty:
            raise ValueError("Reference data cannot be empty")

        self._reference_data = reference_data.copy()
        logger.info(
            "Set reference data: %d rows, %d features",
            len(reference_data),
            len(reference_data.columns),
        )

    def detect_drift(self, current_data: pd.DataFrame) -> DriftReport:
        """Detect drift in current data compared to reference

        Args:
            current_data: Current data to check for drift

        Returns:
            DriftReport with detection results

        Raises:
            ValueError: If reference data not set or current data is empty
        """
        if self._reference_data is None:
            raise ValueError("Reference data not set. Call set_reference_data() first.")

        if current_data.empty:
            raise ValueError("Current data cannot be empty")

        try:
            logger.info("Running drift detection on %d rows", len(current_data))

            # Filter to numeric columns only for drift detection
            # Exclude binary target/prediction columns as they're often categorical
            numeric_cols = self._reference_data.select_dtypes(include=["number"]).columns.tolist()

            # Exclude common target/prediction columns
            exclude_cols = ["target", "prediction", "label", "pred"]
            numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

            # Create filtered dataframes
            ref_filtered = self._reference_data[numeric_cols]
            cur_filtered = current_data[numeric_cols]

            # Create Evidently data drift report
            report = Report(
                metrics=[
                    DataDriftPreset(
                        stattest=self.config.stattest,
                        stattest_threshold=self.config.stattest_threshold,
                        drift_share=self.config.drift_share,
                    )
                ]
            )

            # Run report
            report.run(
                reference_data=ref_filtered,
                current_data=cur_filtered,
            )

            # Extract results
            result = report.as_dict()
            drift_metrics = result["metrics"][0]["result"]

            # Get drifted features
            drifted_features = []
            drift_scores = {}

            if "drift_by_columns" in drift_metrics:
                for feature, feature_drift in drift_metrics["drift_by_columns"].items():
                    if feature_drift.get("drift_detected", False):
                        drifted_features.append(feature)
                    drift_scores[feature] = feature_drift.get("drift_score", 0.0)

            # Create drift report
            drift_report = DriftReport(
                timestamp=datetime.now(),
                dataset_drift=drift_metrics.get("dataset_drift", False),
                drift_share=drift_metrics.get("share_of_drifted_columns", 0.0),
                number_of_drifted_features=drift_metrics.get("number_of_drifted_columns", 0),
                total_features=drift_metrics.get("number_of_columns", 0),
                drifted_features=drifted_features,
                drift_scores=drift_scores,
                metrics=drift_metrics,
            )

            # Save report
            self._save_report(report, drift_report)

            logger.info(
                "Drift detection complete: dataset_drift=%s, drifted_features=%d/%d",
                drift_report.dataset_drift,
                drift_report.number_of_drifted_features,
                drift_report.total_features,
            )

            return drift_report

        except Exception as e:
            logger.error("Drift detection failed: %s", e)
            raise RuntimeError(f"Drift detection failed: {e}") from e

    def _save_report(self, evidently_report: Report, drift_report: DriftReport) -> None:
        """Save drift report to workspace

        Args:
            evidently_report: Evidently report object
            drift_report: Drift report dataclass
        """
        try:
            # Create project-specific directory
            project_dir = self.workspace_path / self.project_name / "drift"
            project_dir.mkdir(parents=True, exist_ok=True)

            # Save Evidently HTML report
            timestamp_str = drift_report.timestamp.strftime("%Y%m%d_%H%M%S")
            html_path = project_dir / f"drift_report_{timestamp_str}.html"
            evidently_report.save_html(str(html_path))

            logger.debug("Saved drift report to: %s", html_path)

        except Exception as e:
            logger.warning("Failed to save drift report: %s", e)

    def get_reference_data(self) -> pd.DataFrame | None:
        """Get current reference data

        Returns:
            Reference data DataFrame or None if not set
        """
        return self._reference_data.copy() if self._reference_data is not None else None
