"""
Knowledge Base Content Freshness and Update Tracking

This module provides tools for tracking content freshness, identifying stale
content, scheduling updates, and maintaining content lifecycle management.
"""

import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from indexer import KnowledgeBaseIndexer
from models import KnowledgeBaseEntry, KnowledgeBaseIndex

logger = logging.getLogger(__name__)


@dataclass
class ContentFreshnessInfo:
    """
    Information about content freshness for an entry.

    Attributes:
        entry_title: Title of the entry
        entry_path: Path to the entry file
        last_updated: When the entry was last updated
        days_since_update: Number of days since last update
        freshness_status: Status (fresh, aging, stale, critical)
        update_priority: Priority for updates (low, medium, high, urgent)
        suggested_review_date: When this entry should be reviewed next
        staleness_factors: Factors contributing to staleness
        update_recommendations: Specific recommendations for updates
    """

    entry_title: str
    entry_path: Path
    last_updated: datetime
    days_since_update: int
    freshness_status: str
    update_priority: str
    suggested_review_date: datetime
    staleness_factors: list[str]
    update_recommendations: list[str]


@dataclass
class UpdateSchedule:
    """
    Scheduled update information for entries.

    Attributes:
        entry_title: Title of the entry
        scheduled_date: When the update is scheduled
        update_type: Type of update (review, refresh, rewrite)
        assigned_to: Who is assigned to do the update
        notes: Additional notes about the update
        priority: Priority level
        estimated_effort: Estimated effort in hours
    """

    entry_title: str
    scheduled_date: datetime
    update_type: str
    assigned_to: str | None = None
    notes: str | None = None
    priority: str = "medium"
    estimated_effort: float | None = None


@dataclass
class FreshnessReport:
    """
    Comprehensive freshness report for the knowledge base.

    Attributes:
        generated_at: When the report was generated
        total_entries: Total number of entries analyzed
        freshness_breakdown: Count of entries by freshness status
        priority_breakdown: Count of entries by update priority
        stale_entries: list of stale entries needing attention
        upcoming_reviews: Entries scheduled for review soon
        recommendations: Overall recommendations for content maintenance
        update_schedule: Suggested update schedule
    """

    generated_at: datetime
    total_entries: int
    freshness_breakdown: dict[str, int]
    priority_breakdown: dict[str, int]
    stale_entries: list[ContentFreshnessInfo]
    upcoming_reviews: list[ContentFreshnessInfo]
    recommendations: list[str]
    update_schedule: list[UpdateSchedule]


class ContentFreshnessTracker:
    """
    System for tracking content freshness and managing update schedules.

    This class provides tools for identifying stale content, scheduling updates,
    and maintaining content lifecycle management.
    """

    def __init__(self, kb_path: Path, db_path: Path | None = None):
        self.kb_path = Path(kb_path)
        self.db_path = db_path or (self.kb_path / ".meta" / "freshness.db")

        # Load knowledge base index
        self.indexer = KnowledgeBaseIndexer(self.kb_path)
        self.index = self.indexer.load_index_from_file()
        if not self.index:
            logger.info("Building knowledge base index for freshness tracking")
            self.index = self.indexer.build_index()

        # Freshness thresholds (in days)
        self.freshness_thresholds = {
            "fresh": 30,  # 0-30 days: fresh
            "aging": 90,  # 31-90 days: aging
            "stale": 180,  # 91-180 days: stale
            "critical": 365,  # 181+ days: critical
        }

        # Priority thresholds
        self.priority_thresholds = {
            "low": 90,  # 0-90 days: low priority
            "medium": 180,  # 91-180 days: medium priority
            "high": 365,  # 181-365 days: high priority
            "urgent": float("inf"),  # 365+ days: urgent
        }

        # Initialize database
        self._init_database()

        # Statistics
        self.stats = {
            "entries_tracked": 0,
            "stale_entries_identified": 0,
            "updates_scheduled": 0,
            "reviews_completed": 0,
            "last_freshness_check": None,
        }

    def _init_database(self) -> None:
        """Initialize the freshness tracking database."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create freshness tracking table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS freshness_tracking (
                        entry_title TEXT PRIMARY KEY,
                        entry_path TEXT NOT NULL,
                        last_updated TEXT NOT NULL,
                        last_reviewed TEXT,
                        freshness_status TEXT NOT NULL,
                        update_priority TEXT NOT NULL,
                        staleness_factors TEXT,
                        next_review_date TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """
                )

                # Create update schedule table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS update_schedule (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        entry_title TEXT NOT NULL,
                        scheduled_date TEXT NOT NULL,
                        update_type TEXT NOT NULL,
                        assigned_to TEXT,
                        notes TEXT,
                        priority TEXT DEFAULT 'medium',
                        estimated_effort REAL,
                        completed BOOLEAN DEFAULT FALSE,
                        created_at TEXT NOT NULL
                    )
                """
                )

                # Create indexes
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_freshness_status ON freshness_tracking(freshness_status)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_update_priority ON freshness_tracking(update_priority)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_next_review ON freshness_tracking(next_review_date)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_scheduled_date ON update_schedule(scheduled_date)"
                )

                conn.commit()

        except Exception as e:
            logger.error(f"Failed to initialize freshness database: {e}")
            raise

    def analyze_content_freshness(self) -> FreshnessReport:
        """
        Analyze freshness of all knowledge base entries.

        Returns:
            FreshnessReport with comprehensive freshness analysis
        """
        logger.info("Analyzing content freshness")

        freshness_infos = []
        freshness_breakdown = {"fresh": 0, "aging": 0, "stale": 0, "critical": 0}
        priority_breakdown = {"low": 0, "medium": 0, "high": 0, "urgent": 0}

        try:
            for entry in self.index.entries:
                freshness_info = self._analyze_entry_freshness(entry)
                freshness_infos.append(freshness_info)

                # Update breakdowns
                freshness_breakdown[freshness_info.freshness_status] += 1
                priority_breakdown[freshness_info.update_priority] += 1

                # Store in database
                self._store_freshness_info(freshness_info)

            # Identify stale entries
            stale_entries = [
                info
                for info in freshness_infos
                if info.freshness_status in ["stale", "critical"]
            ]

            # Identify upcoming reviews
            upcoming_reviews = [
                info
                for info in freshness_infos
                if info.suggested_review_date <= datetime.now() + timedelta(days=7)
            ]

            # Generate recommendations
            recommendations = self._generate_freshness_recommendations(freshness_infos)

            # Generate update schedule
            update_schedule = self._generate_update_schedule(stale_entries)

            # Update statistics
            self.stats["entries_tracked"] = len(freshness_infos)
            self.stats["stale_entries_identified"] = len(stale_entries)
            self.stats["last_freshness_check"] = datetime.now().isoformat()

            return FreshnessReport(
                generated_at=datetime.now(),
                total_entries=len(freshness_infos),
                freshness_breakdown=freshness_breakdown,
                priority_breakdown=priority_breakdown,
                stale_entries=stale_entries,
                upcoming_reviews=upcoming_reviews,
                recommendations=recommendations,
                update_schedule=update_schedule,
            )

        except Exception as e:
            logger.error(f"Failed to analyze content freshness: {e}")
            return FreshnessReport(
                generated_at=datetime.now(),
                total_entries=0,
                freshness_breakdown={},
                priority_breakdown={},
                stale_entries=[],
                upcoming_reviews=[],
                recommendations=[],
                update_schedule=[],
            )

    def _analyze_entry_freshness(
        self, entry: KnowledgeBaseEntry
    ) -> ContentFreshnessInfo:
        """Analyze freshness for a single entry."""
        current_time = datetime.now()
        days_since_update = (current_time - entry.last_updated).days

        # Determine freshness status
        if days_since_update <= self.freshness_thresholds["fresh"]:
            freshness_status = "fresh"
        elif days_since_update <= self.freshness_thresholds["aging"]:
            freshness_status = "aging"
        elif days_since_update <= self.freshness_thresholds["stale"]:
            freshness_status = "stale"
        else:
            freshness_status = "critical"

        # Determine update priority
        if days_since_update <= self.priority_thresholds["low"]:
            update_priority = "low"
        elif days_since_update <= self.priority_thresholds["medium"]:
            update_priority = "medium"
        elif days_since_update <= self.priority_thresholds["high"]:
            update_priority = "high"
        else:
            update_priority = "urgent"

        # Identify staleness factors
        staleness_factors = self._identify_staleness_factors(entry, days_since_update)

        # Generate update recommendations
        update_recommendations = self._generate_entry_update_recommendations(
            entry, freshness_status, staleness_factors
        )

        # Calculate suggested review date
        if freshness_status == "fresh":
            review_days = 60
        elif freshness_status == "aging":
            review_days = 30
        elif freshness_status == "stale":
            review_days = 14
        else:  # critical
            review_days = 7

        suggested_review_date = current_time + timedelta(days=review_days)

        return ContentFreshnessInfo(
            entry_title=entry.title,
            entry_path=entry.content_path,
            last_updated=entry.last_updated,
            days_since_update=days_since_update,
            freshness_status=freshness_status,
            update_priority=update_priority,
            suggested_review_date=suggested_review_date,
            staleness_factors=staleness_factors,
            update_recommendations=update_recommendations,
        )

    def _identify_staleness_factors(
        self, entry: KnowledgeBaseEntry, days_since_update: int
    ) -> list[str]:
        """Identify factors that contribute to content staleness."""
        factors = []

        # Age factor
        if days_since_update > 365:
            factors.append(f"Very old content ({days_since_update} days)")
        elif days_since_update > 180:
            factors.append(f"Old content ({days_since_update} days)")

        # Technology-specific factors
        if "mlx" in entry.tags:
            factors.append("MLX framework evolves rapidly - may need updates")

        if "apple-silicon" in entry.tags:
            factors.append("Apple Silicon ecosystem changes frequently")

        # Category-specific factors
        if entry.category == "troubleshooting":
            factors.append(
                "Troubleshooting entries may become outdated as issues are resolved"
            )
        elif entry.category == "performance":
            factors.append(
                "Performance optimizations may be superseded by new techniques"
            )

        # Check for version-specific content
        try:
            content = entry.get_content()
            version_patterns = [r"version\s+\d+\.\d+", r"v\d+\.\d+", r"python\s+3\.\d+"]
            for pattern in version_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    factors.append(
                        "Contains version-specific information that may be outdated"
                    )
                    break
        except Exception:
            pass

        return factors

    def _generate_entry_update_recommendations(
        self,
        entry: KnowledgeBaseEntry,
        freshness_status: str,
        staleness_factors: list[str],
    ) -> list[str]:
        """Generate specific update recommendations for an entry."""
        recommendations = []

        if freshness_status == "critical":
            recommendations.append(
                "Urgent review needed - content may be significantly outdated"
            )
            recommendations.append(
                "Verify all code examples still work with current versions"
            )
            recommendations.append("Check if the problem/solution is still relevant")
        elif freshness_status == "stale":
            recommendations.append("Review content for accuracy and relevance")
            recommendations.append("Update any version-specific information")
            recommendations.append("Test code examples with current dependencies")
        elif freshness_status == "aging":
            recommendations.append("Light review to ensure content is still accurate")
            recommendations.append(
                "Consider adding recent developments or improvements"
            )

        # Factor-specific recommendations
        if "MLX framework evolves rapidly" in staleness_factors:
            recommendations.append("Check MLX documentation for API changes")
            recommendations.append("Test code examples with latest MLX version")

        if "version-specific information" in staleness_factors:
            recommendations.append(
                "Update version numbers and compatibility information"
            )

        if entry.category == "troubleshooting":
            recommendations.append("Verify the issue still exists in current versions")
            recommendations.append("Check if there are better solutions available now")

        return recommendations

    def _generate_freshness_recommendations(
        self, freshness_infos: list[ContentFreshnessInfo]
    ) -> list[str]:
        """Generate overall recommendations for content maintenance."""
        recommendations = []

        # Count entries by status
        status_counts = {}
        for info in freshness_infos:
            status_counts[info.freshness_status] = (
                status_counts.get(info.freshness_status, 0) + 1
            )

        total_entries = len(freshness_infos)

        # Critical entries
        critical_count = status_counts.get("critical", 0)
        if critical_count > 0:
            recommendations.append(
                f"URGENT: {critical_count} entries are critically outdated and need immediate attention"
            )

        # Stale entries
        stale_count = status_counts.get("stale", 0)
        if stale_count > 0:
            recommendations.append(
                f"{stale_count} entries are stale and should be reviewed soon"
            )

        # Overall health assessment
        outdated_count = critical_count + stale_count
        if outdated_count > total_entries * 0.3:
            recommendations.append(
                "Knowledge base has significant outdated content - consider a comprehensive review"
            )
        elif outdated_count > total_entries * 0.1:
            recommendations.append(
                "Knowledge base has moderate outdated content - regular maintenance recommended"
            )
        else:
            recommendations.append(
                "Knowledge base is in good shape - continue regular maintenance"
            )

        # Category-specific recommendations
        category_issues = {}
        for info in freshness_infos:
            if info.freshness_status in ["stale", "critical"]:
                category = info.entry_path.parent.name
                category_issues[category] = category_issues.get(category, 0) + 1

        for category, count in category_issues.items():
            if count > 2:
                recommendations.append(
                    f"Category '{category}' has {count} outdated entries - consider focused review"
                )

        return recommendations

    def _generate_update_schedule(
        self, stale_entries: list[ContentFreshnessInfo]
    ) -> list[UpdateSchedule]:
        """Generate suggested update schedule for stale entries."""
        schedule = []
        current_date = datetime.now()

        # Sort by priority (urgent first)
        priority_order = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
        sorted_entries = sorted(
            stale_entries,
            key=lambda x: (
                priority_order.get(x.update_priority, 4),
                x.days_since_update,
            ),
            reverse=True,
        )

        # Schedule updates
        schedule_date = current_date
        for i, entry in enumerate(sorted_entries):
            # Determine update type
            if entry.freshness_status == "critical":
                update_type = "rewrite"
                effort = 2.0
            elif entry.freshness_status == "stale":
                update_type = "refresh"
                effort = 1.0
            else:
                update_type = "review"
                effort = 0.5

            # Space out updates
            if entry.update_priority == "urgent":
                days_offset = i  # Immediate to next few days
            elif entry.update_priority == "high":
                days_offset = i + 3  # Start after urgent items
            else:
                days_offset = i + 7  # Start after high priority items

            schedule_date = current_date + timedelta(days=days_offset)

            schedule.append(
                UpdateSchedule(
                    entry_title=entry.entry_title,
                    scheduled_date=schedule_date,
                    update_type=update_type,
                    priority=entry.update_priority,
                    estimated_effort=effort,
                    notes=f"Factors: {', '.join(entry.staleness_factors[:2])}",
                )
            )

        return schedule

    def _store_freshness_info(self, info: ContentFreshnessInfo) -> None:
        """Store freshness information in the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO freshness_tracking (
                        entry_title, entry_path, last_updated, freshness_status,
                        update_priority, staleness_factors, next_review_date,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        info.entry_title,
                        str(info.entry_path),
                        info.last_updated.isoformat(),
                        info.freshness_status,
                        info.update_priority,
                        json.dumps(info.staleness_factors),
                        info.suggested_review_date.isoformat(),
                        datetime.now().isoformat(),
                        datetime.now().isoformat(),
                    ),
                )

                conn.commit()

        except Exception as e:
            logger.error(f"Failed to store freshness info for {info.entry_title}: {e}")

    def schedule_update(self, schedule: UpdateSchedule) -> bool:
        """
        Schedule an update for an entry.

        Args:
            schedule: UpdateSchedule object with update details

        Returns:
            True if successfully scheduled, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO update_schedule (
                        entry_title, scheduled_date, update_type, assigned_to,
                        notes, priority, estimated_effort, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        schedule.entry_title,
                        schedule.scheduled_date.isoformat(),
                        schedule.update_type,
                        schedule.assigned_to,
                        schedule.notes,
                        schedule.priority,
                        schedule.estimated_effort,
                        datetime.now().isoformat(),
                    ),
                )

                conn.commit()
                self.stats["updates_scheduled"] += 1
                logger.info(
                    f"Scheduled {schedule.update_type} for {schedule.entry_title}"
                )
                return True

        except Exception as e:
            logger.error(f"Failed to schedule update for {schedule.entry_title}: {e}")
            return False

    def get_scheduled_updates(
        self, days_ahead: int = 30, completed: bool = False
    ) -> list[UpdateSchedule]:
        """
        Get scheduled updates within the specified time range.

        Args:
            days_ahead: Number of days ahead to look for scheduled updates
            completed: Whether to include completed updates

        Returns:
            List of UpdateSchedule objects
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                end_date = datetime.now() + timedelta(days=days_ahead)

                cursor.execute(
                    """
                    SELECT entry_title, scheduled_date, update_type, assigned_to,
                            notes, priority, estimated_effort
                    FROM update_schedule
                    WHERE scheduled_date <= ? AND completed = ?
                    ORDER BY scheduled_date ASC
                    """,
                    (end_date.isoformat(), completed),
                )

                schedules = []
                for row in cursor.fetchall():
                    schedules.append(
                        UpdateSchedule(
                            entry_title=row[0],
                            scheduled_date=datetime.fromisoformat(row[1]),
                            update_type=row[2],
                            assigned_to=row[3],
                            notes=row[4],
                            priority=row[5],
                            estimated_effort=row[6],
                        )
                    )

                return schedules

        except Exception as e:
            logger.error(f"Failed to get scheduled updates: {e}")
            return []

    def mark_update_completed(self, entry_title: str, update_type: str) -> bool:
        """
        Mark a scheduled update as completed.

        Args:
            entry_title: Title of the entry that was updated
            update_type: Type of update that was completed

        Returns:
            True if successfully marked as completed, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    UPDATE update_schedule
                    SET completed = TRUE
                    WHERE entry_title = ? AND update_type = ? AND completed = FALSE
                    """,
                    (entry_title, update_type),
                )

                if cursor.rowcount > 0:
                    conn.commit()
                    self.stats["reviews_completed"] += 1
                    logger.info(f"Marked {update_type} for {entry_title} as completed")
                    return True
                else:
                    logger.warning(f"No pending {update_type} found for {entry_title}")
                    return False

        except Exception as e:
            logger.error(f"Failed to mark update as completed: {e}")
            return False

    def get_entries_due_for_review(
        self, days_ahead: int = 7
    ) -> list[ContentFreshnessInfo]:
        """
        Get entries that are due for review within the specified time range.

        Args:
            days_ahead: Number of days ahead to look for due reviews

        Returns:
            List of ContentFreshnessInfo objects due for review
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                end_date = datetime.now() + timedelta(days=days_ahead)

                cursor.execute(
                    """
                    SELECT entry_title, entry_path, last_updated, freshness_status,
                            update_priority, staleness_factors, next_review_date
                    FROM freshness_tracking
                    WHERE next_review_date <= ?
                    ORDER BY next_review_date ASC
                    """,
                    (end_date.isoformat(),),
                )

                due_entries = []
                for row in cursor.fetchall():
                    last_updated = datetime.fromisoformat(row[2])
                    next_review = datetime.fromisoformat(row[6])
                    days_since_update = (datetime.now() - last_updated).days

                    due_entries.append(
                        ContentFreshnessInfo(
                            entry_title=row[0],
                            entry_path=Path(row[1]),
                            last_updated=last_updated,
                            days_since_update=days_since_update,
                            freshness_status=row[3],
                            update_priority=row[4],
                            suggested_review_date=next_review,
                            staleness_factors=json.loads(row[5]) if row[5] else [],
                            update_recommendations=[],  # Will be generated on demand
                        )
                    )

                return due_entries

        except Exception as e:
            logger.error(f"Failed to get entries due for review: {e}")
            return []

    def update_entry_freshness(self, entry_title: str) -> bool:
        """
        Update the freshness information for a specific entry after it has been modified.

        Args:
            entry_title: Title of the entry that was updated

        Returns:
            True if successfully updated, False otherwise
        """
        try:
            # Find the entry in the index
            entry = None
            for e in self.index.entries:
                if e.title == entry_title:
                    entry = e
                    break

            if not entry:
                logger.error(f"Entry '{entry_title}' not found in index")
                return False

            # Re-analyze freshness
            freshness_info = self._analyze_entry_freshness(entry)

            # Store updated information
            self._store_freshness_info(freshness_info)

            logger.info(f"Updated freshness information for '{entry_title}'")
            return True

        except Exception as e:
            logger.error(f"Failed to update entry freshness for {entry_title}: {e}")
            return False

    def generate_freshness_report_json(self, output_path: Path | None = None) -> Path:
        """
        Generate a JSON report of content freshness.

        Args:
            output_path: Optional path for the output file

        Returns:
            Path to the generated report file
        """
        if not output_path:
            output_path = self.kb_path / ".meta" / "freshness_report.json"

        try:
            report = self.analyze_content_freshness()

            # Convert to serializable format
            report_data = {
                "generated_at": report.generated_at.isoformat(),
                "total_entries": report.total_entries,
                "freshness_breakdown": report.freshness_breakdown,
                "priority_breakdown": report.priority_breakdown,
                "stale_entries": [
                    {
                        "title": entry.entry_title,
                        "path": str(entry.entry_path),
                        "days_since_update": entry.days_since_update,
                        "status": entry.freshness_status,
                        "priority": entry.update_priority,
                        "factors": entry.staleness_factors,
                        "recommendations": entry.update_recommendations,
                    }
                    for entry in report.stale_entries
                ],
                "upcoming_reviews": [
                    {
                        "title": entry.entry_title,
                        "review_date": entry.suggested_review_date.isoformat(),
                        "status": entry.freshness_status,
                    }
                    for entry in report.upcoming_reviews
                ],
                "recommendations": report.recommendations,
                "update_schedule": [
                    {
                        "title": schedule.entry_title,
                        "scheduled_date": schedule.scheduled_date.isoformat(),
                        "update_type": schedule.update_type,
                        "priority": schedule.priority,
                        "estimated_effort": schedule.estimated_effort,
                        "notes": schedule.notes,
                    }
                    for schedule in report.update_schedule
                ],
                "statistics": self.stats,
            }

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(report_data, f, indent=2)

            logger.info(f"Generated freshness report: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to generate freshness report: {e}")
            raise

    def cleanup_old_schedules(self, days_old: int = 90) -> int:
        """
        Clean up old completed update schedules.

        Args:
            days_old: Remove completed schedules older than this many days

        Returns:
            Number of schedules removed
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cutoff_date = datetime.now() - timedelta(days=days_old)

                cursor.execute(
                    """
                    DELETE FROM update_schedule
                    WHERE completed = TRUE AND created_at < ?
                    """,
                    (cutoff_date.isoformat(),),
                )

                removed_count = cursor.rowcount
                conn.commit()

                logger.info(f"Cleaned up {removed_count} old completed schedules")
                return removed_count

        except Exception as e:
            logger.error(f"Failed to cleanup old schedules: {e}")
            return 0

    def get_statistics(self) -> dict[str, Any]:
        """Get freshness tracking statistics."""
        return self.stats.copy()


# Convenience functions for common operations


def check_knowledge_base_freshness(kb_path: Path) -> FreshnessReport:
    """
    Quick function to check knowledge base freshness.

    Args:
        kb_path: Path to the knowledge base directory

    Returns:
        FreshnessReport with analysis results
    """
    tracker = ContentFreshnessTracker(kb_path)
    return tracker.analyze_content_freshness()


def schedule_stale_content_updates(kb_path: Path) -> list[UpdateSchedule]:
    """
    Quick function to schedule updates for stale content.

    Args:
        kb_path: Path to the knowledge base directory

    Returns:
        List of scheduled updates
    """
    tracker = ContentFreshnessTracker(kb_path)
    report = tracker.analyze_content_freshness()

    # Schedule updates for stale entries
    for schedule in report.update_schedule:
        tracker.schedule_update(schedule)

    return report.update_schedule


def get_entries_needing_attention(
    kb_path: Path, days_ahead: int = 7
) -> list[ContentFreshnessInfo]:
    """
    Quick function to get entries needing attention soon.

    Args:
        kb_path: Path to the knowledge base directory
        days_ahead: Number of days ahead to look

    Returns:
        List of entries needing attention
    """
    tracker = ContentFreshnessTracker(kb_path)
    return tracker.get_entries_due_for_review(days_ahead)
