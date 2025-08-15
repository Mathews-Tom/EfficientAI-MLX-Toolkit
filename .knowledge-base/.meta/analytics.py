"""
Knowledge Base Usage Analytics and Tracking

This module provides comprehensive analytics and tracking capabilities for
knowledge base usage, including entry access tracking, search analytics,
and usage pattern analysis.
"""

import json
import logging
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from models import KnowledgeBaseEntry, KnowledgeBaseIndex

logger = logging.getLogger(__name__)


@dataclass
class UsageEvent:
    """
    Represents a single usage event in the knowledge base.

    Attributes:
        event_type: Type of event (view, search, create, update)
        entry_title: Title of the entry involved (if applicable)
        user_id: Identifier for the user (optional, privacy-conscious)
        timestamp: When the event occurred
        context: Additional context about the event
        metadata: Additional metadata as key-value pairs
    """

    event_type: str
    entry_title: str | None = None
    user_id: str | None = None
    timestamp: datetime = None
    context: str | None = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AnalyticsReport:
    """
    Comprehensive analytics report for the knowledge base.

    Attributes:
        report_period: Time period covered by this report
        total_events: Total number of events in the period
        popular_entries: Most accessed entries
        popular_searches: Most common search queries
        category_usage: Usage breakdown by category
        tag_usage: Usage breakdown by tags
        user_activity: User activity patterns
        knowledge_gaps: Identified knowledge gaps
        recommendations: Recommendations based on analytics
    """

    report_period: tuple[datetime, datetime]
    total_events: int
    popular_entries: list[tuple[str, int]]
    popular_searches: list[tuple[str, int]]
    category_usage: dict[str, int]
    tag_usage: dict[str, int]
    user_activity: dict[str, int]
    knowledge_gaps: list[str]
    recommendations: list[str]


class KnowledgeBaseAnalytics:
    """
    Analytics and tracking system for knowledge base usage.

    This class provides comprehensive tracking of knowledge base usage,
    including entry access, search patterns, and user behavior analysis.
    """

    def __init__(
        self, kb_path: Path, db_path: Path | None = None, privacy_mode: bool = True
    ):
        self.kb_path = Path(kb_path)
        self.db_path = db_path or (self.kb_path / ".meta" / "analytics.db")
        self.privacy_mode = privacy_mode

        # Initialize database
        self._init_database()

        # In-memory cache for recent events
        self.recent_events: list[UsageEvent] = []
        self.cache_size = 1000

        # Statistics
        self.stats = {
            "total_events_tracked": 0,
            "unique_entries_accessed": 0,
            "unique_searches_performed": 0,
            "active_users": 0,
            "tracking_start_date": None,
        }

        # Load existing statistics
        self._load_statistics()

    def _init_database(self) -> None:
        """Initialize the analytics database."""
        try:
            # Ensure directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create events table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_type TEXT NOT NULL,
                        entry_title TEXT,
                        user_id TEXT,
                        timestamp TEXT NOT NULL,
                        context TEXT,
                        metadata TEXT
                    )
                """
                )

                # Create indexes for performance
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_event_type ON events(event_type)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_entry_title ON events(entry_title)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_user_id ON events(user_id)"
                )

                # Create aggregated statistics table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS statistics (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """
                )

                conn.commit()

        except Exception as e:
            logger.error(f"Failed to initialize analytics database: {e}")
            raise

    def track_event(
        self,
        event_type: str,
        entry_title: str | None = None,
        user_id: str | None = None,
        context: str | None = None,
        **metadata,
    ) -> None:
        """
        Track a usage event.

        Args:
            event_type: Type of event (view, search, create, update, etc.)
            entry_title: Title of the entry involved
            user_id: User identifier (anonymized if privacy_mode is True)
            context: Additional context about the event
            **metadata: Additional metadata as keyword arguments
        """
        try:
            # Anonymize user_id if privacy mode is enabled
            if self.privacy_mode and user_id:
                user_id = self._anonymize_user_id(user_id)

            # Create event
            event = UsageEvent(
                event_type=event_type,
                entry_title=entry_title,
                user_id=user_id,
                context=context,
                metadata=metadata,
            )

            # Add to cache
            self.recent_events.append(event)
            if len(self.recent_events) > self.cache_size:
                self.recent_events.pop(0)

            # Store in database
            self._store_event(event)

            # Update statistics
            self.stats["total_events_tracked"] += 1
            if not self.stats["tracking_start_date"]:
                self.stats["tracking_start_date"] = datetime.now().isoformat()

            logger.debug(f"Tracked event: {event_type} for {entry_title or 'N/A'}")

        except Exception as e:
            logger.error(f"Failed to track event: {e}")

    def track_entry_view(
        self,
        entry: KnowledgeBaseEntry,
        user_id: str | None = None,
        context: str | None = None,
    ) -> None:
        """Track when an entry is viewed."""
        self.track_event(
            event_type="view",
            entry_title=entry.title,
            user_id=user_id,
            context=context,
            category=entry.category,
            tags=entry.tags,
            difficulty=entry.difficulty,
        )

        # Update entry usage count
        entry.update_usage()

    def track_search(
        self,
        query: str,
        results_count: int,
        user_id: str | None = None,
        filters: dict | None = None,
    ) -> None:
        """Track search queries and results."""
        self.track_event(
            event_type="search",
            user_id=user_id,
            context=f"Query: {query}",
            query=query,
            results_count=results_count,
            filters=filters or {},
        )

    def track_entry_creation(
        self, entry: KnowledgeBaseEntry, user_id: str | None = None
    ) -> None:
        """Track when a new entry is created."""
        self.track_event(
            event_type="create",
            entry_title=entry.title,
            user_id=user_id,
            context="Entry created",
            category=entry.category,
            tags=entry.tags,
            difficulty=entry.difficulty,
        )

    def track_entry_update(
        self,
        entry: KnowledgeBaseEntry,
        user_id: str | None = None,
        changes: list[str] | None = None,
    ) -> None:
        """Track when an entry is updated."""
        self.track_event(
            event_type="update",
            entry_title=entry.title,
            user_id=user_id,
            context="Entry updated",
            changes=changes or [],
        )

    def _store_event(self, event: UsageEvent) -> None:
        """Store an event in the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO events (event_type, entry_title, user_id, timestamp, context, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        event.event_type,
                        event.entry_title,
                        event.user_id,
                        event.timestamp.isoformat(),
                        event.context,
                        json.dumps(event.metadata),
                    ),
                )
                conn.commit()

        except Exception as e:
            logger.error(f"Failed to store event in database: {e}")

    def _anonymize_user_id(self, user_id: str) -> str:
        """Anonymize user ID for privacy."""
        import hashlib

        return hashlib.sha256(user_id.encode()).hexdigest()[:16]

    def get_popular_entries(
        self, days: int = 30, limit: int = 10
    ) -> list[tuple[str, int]]:
        """
        Get the most popular entries based on view events.

        Args:
            days: Number of days to look back
            limit: Maximum number of entries to return

        Returns:
            List of (entry_title, view_count) tuples
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT entry_title, COUNT(*) as view_count
                    FROM events
                    WHERE event_type = 'view' 
                    AND entry_title IS NOT NULL
                    AND timestamp >= ?
                    GROUP BY entry_title
                    ORDER BY view_count DESC
                    LIMIT ?
                """,
                    (cutoff_date.isoformat(), limit),
                )

                return cursor.fetchall()

        except Exception as e:
            logger.error(f"Failed to get popular entries: {e}")
            return []

    def get_popular_searches(
        self, days: int = 30, limit: int = 10
    ) -> list[tuple[str, int]]:
        """
        Get the most popular search queries.

        Args:
            days: Number of days to look back
            limit: Maximum number of queries to return

        Returns:
            List of (query, search_count) tuples
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT json_extract(metadata, '$.query') as query, COUNT(*) as search_count
                    FROM events
                    WHERE event_type = 'search'
                    AND json_extract(metadata, '$.query') IS NOT NULL
                    AND timestamp >= ?
                    GROUP BY query
                    ORDER BY search_count DESC
                    LIMIT ?
                """,
                    (cutoff_date.isoformat(), limit),
                )

                return cursor.fetchall()

        except Exception as e:
            logger.error(f"Failed to get popular searches: {e}")
            return []

    def get_category_usage(self, days: int = 30) -> dict[str, int]:
        """Get usage statistics by category."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT json_extract(metadata, '$.category') as category, COUNT(*) as usage_count
                    FROM events
                    WHERE event_type = 'view'
                    AND json_extract(metadata, '$.category') IS NOT NULL
                    AND timestamp >= ?
                    GROUP BY category
                    ORDER BY usage_count DESC
                """,
                    (cutoff_date.isoformat(),),
                )

                return dict(cursor.fetchall())

        except Exception as e:
            logger.error(f"Failed to get category usage: {e}")
            return {}

    def get_tag_usage(self, days: int = 30, limit: int = 20) -> dict[str, int]:
        """Get usage statistics by tags."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            tag_counts = Counter()

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT json_extract(metadata, '$.tags') as tags
                    FROM events
                    WHERE event_type = 'view'
                    AND json_extract(metadata, '$.tags') IS NOT NULL
                    AND timestamp >= ?
                """,
                    (cutoff_date.isoformat(),),
                )

                for (tags_json,) in cursor.fetchall():
                    try:
                        tags = json.loads(tags_json)
                        if isinstance(tags, list):
                            for tag in tags:
                                tag_counts[tag] += 1
                    except (json.JSONDecodeError, TypeError):
                        continue

            return dict(tag_counts.most_common(limit))

        except Exception as e:
            logger.error(f"Failed to get tag usage: {e}")
            return {}

    def get_user_activity(self, days: int = 30) -> dict[str, int]:
        """Get user activity statistics."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT user_id, COUNT(*) as activity_count
                    FROM events
                    WHERE user_id IS NOT NULL
                    AND timestamp >= ?
                    GROUP BY user_id
                    ORDER BY activity_count DESC
                """,
                    (cutoff_date.isoformat(),),
                )

                results = cursor.fetchall()

                # Return anonymized results
                return {f"user_{i+1}": count for i, (_, count) in enumerate(results)}

        except Exception as e:
            logger.error(f"Failed to get user activity: {e}")
            return {}

    def identify_knowledge_gaps(
        self, index: KnowledgeBaseIndex, days: int = 30
    ) -> list[str]:
        """
        Identify potential knowledge gaps based on search patterns.

        Args:
            index: Knowledge base index for comparison
            days: Number of days to analyze

        Returns:
            List of potential knowledge gaps
        """
        gaps = []

        try:
            # Get searches that returned few or no results
            cutoff_date = datetime.now() - timedelta(days=days)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT json_extract(metadata, '$.query') as query, 
                            AVG(json_extract(metadata, '$.results_count')) as avg_results
                    FROM events
                    WHERE event_type = 'search'
                    AND json_extract(metadata, '$.query') IS NOT NULL
                    AND timestamp >= ?
                    GROUP BY query
                    HAVING avg_results < 2
                    ORDER BY COUNT(*) DESC
                    LIMIT 10
                """,
                    (cutoff_date.isoformat(),),
                )

                for query, avg_results in cursor.fetchall():
                    if query and avg_results < 2:
                        gaps.append(
                            f"Low results for search: '{query}' (avg: {avg_results:.1f} results)"
                        )

            # Identify categories with low content but high search interest
            popular_searches = self.get_popular_searches(days=days, limit=20)
            existing_categories = set(index.categories.keys())

            for query, count in popular_searches:
                # Simple heuristic: if search term doesn't match existing categories
                query_lower = query.lower()
                if not any(
                    cat in query_lower or query_lower in cat
                    for cat in existing_categories
                ):
                    gaps.append(
                        f"Potential new category needed: '{query}' ({count} searches)"
                    )

        except Exception as e:
            logger.error(f"Failed to identify knowledge gaps: {e}")

        return gaps

    def generate_analytics_report(
        self, index: KnowledgeBaseIndex, days: int = 30
    ) -> AnalyticsReport:
        """
        Generate a comprehensive analytics report.

        Args:
            index: Knowledge base index
            days: Number of days to include in the report

        Returns:
            AnalyticsReport with comprehensive analytics
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Get total events in period
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM events 
                    WHERE timestamp >= ? AND timestamp <= ?
                """,
                    (start_date.isoformat(), end_date.isoformat()),
                )
                total_events = cursor.fetchone()[0]

            # Gather all analytics
            popular_entries = self.get_popular_entries(days=days)
            popular_searches = self.get_popular_searches(days=days)
            category_usage = self.get_category_usage(days=days)
            tag_usage = self.get_tag_usage(days=days)
            user_activity = self.get_user_activity(days=days)
            knowledge_gaps = self.identify_knowledge_gaps(index, days=days)

            # Generate recommendations
            recommendations = self._generate_recommendations(
                popular_entries, popular_searches, category_usage, knowledge_gaps
            )

            return AnalyticsReport(
                report_period=(start_date, end_date),
                total_events=total_events,
                popular_entries=popular_entries,
                popular_searches=popular_searches,
                category_usage=category_usage,
                tag_usage=tag_usage,
                user_activity=user_activity,
                knowledge_gaps=knowledge_gaps,
                recommendations=recommendations,
            )

        except Exception as e:
            logger.error(f"Failed to generate analytics report: {e}")
            return AnalyticsReport(
                report_period=(start_date, end_date),
                total_events=0,
                popular_entries=[],
                popular_searches=[],
                category_usage={},
                tag_usage={},
                user_activity={},
                knowledge_gaps=[],
                recommendations=[],
            )

    def _generate_recommendations(
        self,
        popular_entries: list[tuple[str, int]],
        popular_searches: list[tuple[str, int]],
        category_usage: dict[str, int],
        knowledge_gaps: list[str],
    ) -> list[str]:
        """Generate recommendations based on analytics."""
        recommendations = []

        # Recommend updating popular entries
        if popular_entries:
            top_entry = popular_entries[0]
            recommendations.append(
                f"Consider updating '{top_entry[0]}' - it's your most popular entry ({top_entry[1]} views)"
            )

        # Recommend creating content for popular searches
        if popular_searches:
            top_search = popular_searches[0]
            recommendations.append(
                f"Consider creating content for '{top_search[0]}' - it's frequently searched ({top_search[1]} times)"
            )

        # Recommend focusing on popular categories
        if category_usage:
            top_category = max(category_usage.items(), key=lambda x: x[1])
            recommendations.append(
                f"Focus on '{top_category[0]}' category - it has the highest usage ({top_category[1]} views)"
            )

        # Recommend addressing knowledge gaps
        if knowledge_gaps:
            recommendations.append(
                "Address identified knowledge gaps to improve search satisfaction"
            )

        return recommendations

    def _load_statistics(self) -> None:
        """Load existing statistics from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT key, value FROM statistics")

                for key, value in cursor.fetchall():
                    try:
                        # Try to parse as JSON first, then as string
                        parsed_value = json.loads(value)
                        self.stats[key] = parsed_value
                    except json.JSONDecodeError:
                        self.stats[key] = value

        except Exception as e:
            logger.debug(f"Could not load existing statistics: {e}")

    def _save_statistics(self) -> None:
        """Save current statistics to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                for key, value in self.stats.items():
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO statistics (key, value, updated_at)
                        VALUES (?, ?, ?)
                    """,
                        (key, json.dumps(value), datetime.now().isoformat()),
                    )

                conn.commit()

        except Exception as e:
            logger.error(f"Failed to save statistics: {e}")

    def export_analytics_data(
        self, output_path: Path, days: int = 30, format: str = "json"
    ) -> None:
        """
        Export analytics data to file.

        Args:
            output_path: Path to save the exported data
            days: Number of days to include
            format: Export format (json, csv)
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT event_type, entry_title, timestamp, context, metadata
                    FROM events
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                """,
                    (cutoff_date.isoformat(),),
                )

                events = cursor.fetchall()

            if format.lower() == "json":
                export_data = {
                    "export_date": datetime.now().isoformat(),
                    "period_days": days,
                    "total_events": len(events),
                    "events": [
                        {
                            "event_type": event[0],
                            "entry_title": event[1],
                            "timestamp": event[2],
                            "context": event[3],
                            "metadata": json.loads(event[4]) if event[4] else {},
                        }
                        for event in events
                    ],
                }

                with open(output_path, "w") as f:
                    json.dump(export_data, f, indent=2, default=str)

            elif format.lower() == "csv":
                import csv

                with open(output_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "event_type",
                            "entry_title",
                            "timestamp",
                            "context",
                            "metadata",
                        ]
                    )
                    writer.writerows(events)

            logger.info(f"Exported {len(events)} events to {output_path}")

        except Exception as e:
            logger.error(f"Failed to export analytics data: {e}")

    def get_analytics_stats(self) -> dict[str, Any]:
        """Get analytics system statistics."""
        # Update statistics before returning
        self._save_statistics()

        return {
            **self.stats,
            "database_path": str(self.db_path),
            "privacy_mode": self.privacy_mode,
            "cache_size": len(self.recent_events),
            "database_size_mb": (
                self.db_path.stat().st_size / (1024 * 1024)
                if self.db_path.exists()
                else 0
            ),
        }


def create_analytics_tracker(kb_path: Path, **kwargs) -> KnowledgeBaseAnalytics:
    """
    Factory function to create an analytics tracker.

    Args:
        kb_path: Path to knowledge base directory
        **kwargs: Additional arguments for KnowledgeBaseAnalytics

    Returns:
        Configured analytics tracker
    """
    return KnowledgeBaseAnalytics(kb_path, **kwargs)


def main():
    """
    Command-line interface for analytics system.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Knowledge Base Analytics")
    parser.add_argument("kb_path", help="Path to knowledge base directory")
    parser.add_argument(
        "--report", action="store_true", help="Generate analytics report"
    )
    parser.add_argument(
        "--days", type=int, default=30, help="Number of days to analyze"
    )
    parser.add_argument("--export", help="Export analytics data to file")
    parser.add_argument("--format", default="json", help="Export format (json, csv)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")

    try:
        # Create analytics tracker
        analytics = create_analytics_tracker(Path(args.kb_path))

        if args.report:
            # Load knowledge base index for report
            from indexer import KnowledgeBaseIndexer

            indexer = KnowledgeBaseIndexer(Path(args.kb_path))
            index = indexer.load_index_from_file()

            if not index:
                print("No index found. Building index...")
                index = indexer.build_index()

            # Generate report
            print(f"📊 Generating analytics report for last {args.days} days...")
            report = analytics.generate_analytics_report(index, days=args.days)

            print(
                f"\n📈 Analytics Report ({report.report_period[0].strftime('%Y-%m-%d')} to {report.report_period[1].strftime('%Y-%m-%d')}):"
            )
            print(f"   Total events: {report.total_events}")

            if report.popular_entries:
                print("\n🔥 Popular Entries:")
                for i, (title, count) in enumerate(report.popular_entries[:5], 1):
                    print(f"   {i}. {title} ({count} views)")

            if report.popular_searches:
                print("\n🔍 Popular Searches:")
                for i, (query, count) in enumerate(report.popular_searches[:5], 1):
                    print(f"   {i}. '{query}' ({count} searches)")

            if report.category_usage:
                print("\n📁 Category Usage:")
                for category, count in sorted(
                    report.category_usage.items(), key=lambda x: x[1], reverse=True
                )[:5]:
                    print(f"   {category}: {count} views")

            if report.knowledge_gaps:
                print("\n🔍 Knowledge Gaps:")
                for gap in report.knowledge_gaps[:3]:
                    print(f"   • {gap}")

            if report.recommendations:
                print("\n💡 Recommendations:")
                for rec in report.recommendations:
                    print(f"   • {rec}")

        elif args.export:
            print("📤 Exporting analytics data...")
            analytics.export_analytics_data(
                Path(args.export), days=args.days, format=args.format
            )
            print(f"✅ Data exported to {args.export}")

        else:
            # Show basic statistics
            stats = analytics.get_analytics_stats()
            print("📊 Analytics Statistics:")
            print(f"   Total events tracked: {stats['total_events_tracked']}")
            print(f"   Database size: {stats['database_size_mb']:.2f} MB")
            print(f"   Privacy mode: {stats['privacy_mode']}")
            print(f"   Tracking since: {stats.get('tracking_start_date', 'N/A')}")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
